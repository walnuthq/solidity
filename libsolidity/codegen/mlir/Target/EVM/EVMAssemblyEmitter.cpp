/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0

#include "EVMAssemblyEmitter.h"

#include "EVMOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#pragma GCC diagnostic pop

#include <libevmasm/Instruction.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace solidity;
using namespace solidity::evmasm;

namespace
{

/// Thrown for any construct the emitter cannot lower; carries the message that
/// the driver reports as the reason the object did not reach the asm stage.
struct EmitError
{
	std::string message;
};

[[noreturn]] void fail(std::string const& _message)
{
	throw EmitError{_message};
}

/// The EVM word slot size, in bytes.
constexpr uint64_t kWord = 32;

/// Largest frame this backend will save with an unrolled copy when the target
/// EVM version has no MCOPY.
constexpr uint64_t kMaxUnrolledFrameWords = 64;

/// Where solc's free memory area begins; below it is scratch and the free
/// memory pointer.
constexpr uint64_t kFreeMemoryStart = 0x80;

/// How many slots the stack cache may hold. Past DUP16 a value is out of reach
/// anyway, and the operands of the instruction being built need room above it.
constexpr size_t kMaxCachedStack = 8;

u256 toU256(llvm::APInt const& _value)
{
	llvm::APInt widened = _value.zextOrTrunc(256);
	llvm::SmallString<80> digits;
	widened.toStringUnsigned(digits, 16);
	return u256("0x" + std::string(digits.c_str()));
}

/// Per-function frame: where each SSA value lives, and where the calling
/// convention expects arguments and results.
struct Frame
{
	uint64_t base = 0;
	uint64_t size = 0;
	std::vector<uint64_t> argSlots;
	std::vector<uint64_t> resultSlots;
	llvm::DenseMap<mlir::Value, uint64_t> valueSlots;
};

class Emitter
{
public:
	Emitter(solidity::mlirgen::EVMAssemblyOptions const& _options):
		m_options(_options),
		m_assembly(std::make_shared<Assembly>(_options.evmVersion, _options.creation, _options.name))
	{
	}

	std::shared_ptr<Assembly> run(mlir::ModuleOp _module)
	{
		m_module = _module;
		registerSegments();
		collectFunctions(_module);
		findRecursion();
		layoutFrames();

		// Tags have to exist before any call or branch can reference them.
		for (mlir::func::FuncOp func: m_functions)
		{
			m_functionTags.emplace(func.getName().str(), m_assembly->newTag());
			for (mlir::Block& block: func.getBody())
				m_blockTags.emplace(&block, m_assembly->newTag());
		}

		// Emitted at offset zero, ahead of the entry tag, so it runs once on
		// every entry into the object and falls straight through.
		if (!m_recursive.empty())
		{
			push(u256(m_saveBase));
			storeToAddress(m_savePointer);
		}

		// The object entry has to sit at offset zero; everything else follows.
		for (mlir::func::FuncOp func: m_functions)
			if (isEntry(func))
				emitFunction(func);
		for (mlir::func::FuncOp func: m_functions)
			if (!isEntry(func))
				emitFunction(func);

		return m_assembly;
	}

private:
	bool isEntry(mlir::func::FuncOp _func) const { return _func.getName() == "__entry"; }

	/// Nested objects become sub-assemblies and data segments become data
	/// items; both are then addressable by name from `dataoffset`/`datasize`.
	void registerSegments()
	{
		for (solidity::mlirgen::EVMSubObject const& sub: m_options.subObjects)
		{
			if (!sub.assembly)
				fail("sub-object '" + sub.name + "' was not emitted");
			AssemblyItem const item = m_assembly->newSub(sub.assembly);
			m_subObjects.emplace(sub.name, SubAssemblyID(item.data()));
		}
		for (solidity::mlirgen::EVMDataSegment const& segment: m_options.dataSegments)
			m_dataSegments.emplace(segment.name, m_assembly->newData(segment.data));
	}

	void collectFunctions(mlir::ModuleOp _module)
	{
		for (mlir::Operation& op: _module.getBody()->getOperations())
		{
			auto func = llvm::dyn_cast<mlir::func::FuncOp>(&op);
			if (!func)
				fail("unexpected module-level op '" + op.getName().getStringRef().str() + "'");
			if (func.isExternal())
				fail("declaration without a body: " + func.getName().str());
			m_functions.push_back(func);
			m_byName[func.getName().str()] = func;
		}
		// A contract with no functions is still a contract: its runtime is code
		// that does nothing, which is what an empty object assembles to. It has
		// to be assemblable because another contract may name it - through
		// `type(C).runtimeCode` - and nest it as a sub-object.
	}

	/// Frames are addressed absolutely, so two live activations of one function
	/// would share slots. Find the functions that can re-enter themselves; a
	/// call into one of those has its frame saved and restored around the call.
	void findRecursion()
	{
		std::map<std::string, std::vector<std::string>> callees;
		for (mlir::func::FuncOp func: m_functions)
		{
			std::vector<std::string>& out = callees[func.getName().str()];
			func.walk([&](mlir::func::CallOp _call) { out.push_back(_call.getCallee().str()); });
		}

		// A function needs saving exactly when it can reach itself.
		for (mlir::func::FuncOp func: m_functions)
		{
			std::string const start = func.getName().str();
			std::set<std::string> seen;
			std::vector<std::string> worklist = callees[start];
			while (!worklist.empty())
			{
				std::string const current = worklist.back();
				worklist.pop_back();
				if (current == start)
				{
					m_recursive.insert(start);
					break;
				}
				if (!m_byName.count(current) || !seen.insert(current).second)
					continue;
				for (std::string const& next: callees[current])
					worklist.push_back(next);
			}
		}
	}

	/// Decides where the frames live.
	///
	/// `memoryguard(x)` is the start of the contract's heap: memory below it is
	/// reserved, memory above it is allocated from. Frames belong in reserved
	/// space, so they go at x and the guard is rewritten to x + their size,
	/// which moves the heap above them. Putting them *at* the heap start
	/// instead - or leaving them at a fixed high address - is what makes every
	/// value touch pay to expand memory out to them, enough to exceed the 2300
	/// gas a plain `send` forwards.
	///
	/// Saved frames of a recursive call chain grow without a static bound, so
	/// nothing can be placed above them; those objects keep the fixed base.
	/// True when nothing in the object can occupy or observe memory, so the
	/// frames may sit wherever is cheapest. A read of a statically empty range
	/// touches nothing and does not count; anything else is assumed to.
	bool objectIgnoresMemory() const
	{
		static std::set<std::string> const touchesMemory = {
			"mload", "mstore", "mstore8", "mcopy", "msize", "calldatacopy", "codecopy",
			"returndatacopy", "extcodecopy", "datacopy", "keccak256", "log0", "log1", "log2",
			"log3", "log4", "create", "create2", "call", "callcode", "delegatecall", "staticcall"};

		bool clean = true;
		for (mlir::func::FuncOp func: m_functions)
			func.walk([&](mlir::Operation* op) {
				if (!op->getDialect() || op->getDialect()->getNamespace() != "evm")
					return;
				std::string const mnemonic = op->getName().stripDialect().str();
				if (touchesMemory.count(mnemonic))
					clean = false;
				else if (mnemonic == "return" || mnemonic == "revert")
				{
					// return(p, 0) and revert(p, 0) read nothing.
					std::optional<llvm::APInt> const length = constantOf(op->getOperand(1));
					if (!length || !length->isZero())
						clean = false;
				}
			});
		return clean;
	}

	void chooseFrameBase()
	{
		if (!m_recursive.empty())
		{
			m_frameBase = m_options.frameBase;
			return;
		}
		if (auto guard = m_module->getAttrOfType<mlir::IntegerAttr>("evm.memory_guard"))
		{
			m_frameBase = guard.getValue().getLimitedValue();
			m_relocatedFrames = true;
			return;
		}
		// No declared heap. If the object cannot tell where memory is used, the
		// frames still have to go somewhere cheap: a fallback that only has to
		// answer within the 2300 gas a `send` forwards cannot afford to expand
		// memory out to a fixed high address first.
		m_frameBase = objectIgnoresMemory() ? kFreeMemoryStart : m_options.frameBase;
	}

	void layoutFrames()
	{
		chooseFrameBase();
		uint64_t next = m_frameBase;
		for (mlir::func::FuncOp func: m_functions)
		{
			Frame frame;
			frame.base = next;
			uint64_t slot = 0;

			mlir::Block& entry = func.getBody().front();
			for (mlir::BlockArgument arg: entry.getArguments())
			{
				frame.valueSlots[arg] = slot;
				frame.argSlots.push_back(slot);
				++slot;
			}
			for (unsigned i = 0; i < func.getNumResults(); ++i)
				frame.resultSlots.push_back(slot++);

			for (mlir::Block& block: func.getBody())
			{
				if (&block != &entry)
					for (mlir::BlockArgument arg: block.getArguments())
						frame.valueSlots[arg] = slot++;
				for (mlir::Operation& op: block.getOperations())
				{
					if (llvm::isa<mlir::arith::ConstantOp>(&op))
						continue; // rematerialised at each use
					if (op.getNumResults() == 1 && isRematerializable(op.getResult(0)))
						continue; // likewise
					for (mlir::Value result: op.getResults())
						frame.valueSlots[result] = slot++;
				}
			}

			frame.size = slot;
			next += slot * kWord;
			m_frames[func.getName().str()] = std::move(frame);
		}

		// Saved frames of recursive activations grow upwards from just past the
		// static frames, addressed through one pointer word.
		m_savePointer = next;
		m_saveBase = next + kWord;
		// Everything this backend owns ends here, so this is where the
		// contract's own heap may start.
		m_heapBase = m_saveBase;
	}

	Frame const& frameOf(mlir::func::FuncOp _func) const { return m_frames.at(_func.getName().str()); }

	uint64_t addressOf(mlir::Value _value) const
	{
		auto it = m_frame->valueSlots.find(_value);
		if (it == m_frame->valueSlots.end())
			fail("value has no frame slot");
		return m_frame->base + it->second * kWord;
	}

	static std::optional<llvm::APInt> constantOf(mlir::Value _value)
	{
		auto constant = _value.getDefiningOp<mlir::arith::ConstantOp>();
		if (!constant)
			return std::nullopt;
		auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
		if (!attr)
			fail("non-integer arith.constant");
		return attr.getValue();
	}

	//===------------------------------------------------------------------===//
	// The stack model
	//
	// m_stack mirrors the physical slots this block has pushed above its own
	// baseline, back() being the top; a slot holds the value it carries, or
	// null when it is a machine word with no SSA identity. Because every value
	// is also written to its frame slot, the model is only ever a cache: a
	// lookup that misses, or lands out of DUP reach, falls back to a reload and
	// nothing depends on the model being complete.
	//===------------------------------------------------------------------===//

	void modelPush(mlir::Value _value = {}) { m_stack.push_back(_value); }

	void modelPop(unsigned _count)
	{
		if (_count > m_stack.size())
			fail("stack model underflow");
		m_stack.resize(m_stack.size() - _count);
	}

	/// True for a value produced by a single-byte opcode that takes nothing and
	/// reads nothing - `caller`, `calldatasize`, the block fields. Giving one a
	/// frame slot costs a store and a reload, eight bytes, to avoid re-emitting
	/// one. They are rematerialised at each use, like constants.
	static bool isRematerializable(mlir::Value _value)
	{
		mlir::Operation* definition = _value.getDefiningOp();
		if (!definition || definition->getNumOperands() != 0 || definition->getNumResults() != 1)
			return false;
		if (!definition->getDialect() || definition->getDialect()->getNamespace() != "evm")
			return false;
		if (!mlir::isMemoryEffectFree(definition))
			return false;
		// Only the ones that really are one opcode: dataoffset, immutables and
		// linker symbols are relocations and can be far larger than a reload.
		std::string name = definition->getName().stripDialect().str();
		for (char& c: name)
			c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
		return c_instructions.count(name) != 0;
	}

	void push(u256 const& _value)
	{
		m_assembly->append(AssemblyItem(_value));
		modelPush();
	}

	void op(Instruction _instruction)
	{
		m_assembly->append(_instruction);
		InstructionInfo const info = instructionInfo(_instruction, m_options.evmVersion);
		modelPop(static_cast<unsigned>(info.args));
		for (int i = 0; i < info.ret; ++i)
			modelPush();
	}

	/// DUP and SWAP move identities around rather than producing new ones, so
	/// they are modelled by hand instead of through the generic effect.
	void emitDup(unsigned _depth)
	{
		solAssert(_depth >= 1 && _depth <= 16);
		m_assembly->append(dupInstruction(_depth));
		modelPush(m_stack[m_stack.size() - _depth]);
	}

	void emitSwap(unsigned _depth)
	{
		solAssert(_depth >= 1 && _depth <= 16);
		m_assembly->append(swapInstruction(_depth));
		std::swap(m_stack.back(), m_stack[m_stack.size() - 1 - _depth]);
	}

	/// Depth of the shallowest live copy of @a _value, if one is in DUP reach.
	std::optional<unsigned> residentDepth(mlir::Value _value) const
	{
		for (size_t i = m_stack.size(); i > 0; --i)
			if (m_stack[i - 1] == _value)
			{
				unsigned const depth = static_cast<unsigned>(m_stack.size() - i) + 1;
				return depth <= 16 ? std::optional<unsigned>(depth) : std::nullopt;
			}
		return std::nullopt;
	}

	/// Returns the stack to the block's baseline. Anything cached on it is also
	/// in its frame slot, so dropping the cache loses nothing; what it protects
	/// is the contract that a jump, and a callee, see the stack they expect.
	void flushStack()
	{
		while (!m_stack.empty())
			op(Instruction::POP);
	}

	/// Assembly tracks one net stack height across the whole item stream, which
	/// only describes straight-line code: a function's return JUMP consumes an
	/// address pushed by a caller that is somewhere else entirely in the
	/// stream, so the running count drifts and eventually trips its own
	/// underflow assertion. Re-anchor it wherever this backend's own invariant
	/// says the stack is back at rest.
	void anchorStackHeight() { m_assembly->setDeposit(1); }

	/// Leaves @a _value on the stack: as a literal, as a copy of a slot the
	/// stack already holds, or failing both by reloading it from its frame.
	void pushValue(mlir::Value _value)
	{
		if (std::optional<llvm::APInt> constant = constantOf(_value))
		{
			push(toU256(*constant));
			return;
		}
		if (std::optional<unsigned> const depth = residentDepth(_value))
		{
			emitDup(*depth);
			return;
		}
		if (isRematerializable(_value))
		{
			std::string name = _value.getDefiningOp()->getName().stripDialect().str();
			for (char& c: name)
				c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
			op(c_instructions.find(name)->second);
			m_stack.back() = _value;
			return;
		}
		push(u256(addressOf(_value)));
		op(Instruction::MLOAD);
		m_stack.back() = _value;
	}

	/// Consumes the stack top, writing it to @a _address.
	void storeToAddress(uint64_t _address)
	{
		push(u256(_address));
		op(Instruction::MSTORE);
	}

	void storeResult(mlir::Value _value) { storeToAddress(addressOf(_value)); }

	/// True for ops whose operands this backend materialises with pushOperands,
	/// i.e. the ones a stack-resident value can be handed to directly. Calls,
	/// branches and returns arrange the stack themselves.
	static bool takesPlainOperands(mlir::Operation& _op)
	{
		if (llvm::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp, mlir::func::CallOp, mlir::func::ReturnOp>(&_op))
			return false;
		if (llvm::isa<mlir::arith::CmpIOp, mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, mlir::arith::TruncIOp>(&_op))
			return true;
		if (arithInstruction(_op))
			return true;
		if (!_op.getDialect() || _op.getDialect()->getNamespace() != "evm")
			return false;
		// The ops that are not a single opcode build their own operand order.
		std::string const mnemonic = _op.getName().stripDialect().str();
		return mnemonic != "dataoffset" && mnemonic != "datasize" && mnemonic != "loadimmutable"
			&& mnemonic != "linkersymbol" && mnemonic != "program";
	}

	/// Where a value sits among an op's operands, or none if it is not one.
	static std::optional<unsigned> operandIndex(mlir::Operation& _op, mlir::Value _value)
	{
		for (unsigned i = 0; i < _op.getNumOperands(); ++i)
			if (_op.getOperand(i) == _value)
				return i;
		return std::nullopt;
	}

	/// Whether a freshly computed result can be left on the stack for the very
	/// next instruction rather than written to its slot and read straight back.
	///
	/// Operands are pushed back to front, so a value already on the stack ends
	/// up underneath everything pushed after it: it can only serve as the
	/// deepest operand, or as the lower of a pair, which one SWAP1 corrects.
	/// That covers the expression-tree edges Yul emits in quantity.
	bool canStayOnStack(mlir::Value _result, mlir::Operation& _definition) const
	{
		if (!_result.hasOneUse())
			return false;
		mlir::Operation* user = *_result.getUsers().begin();
		if (user != _definition.getNextNode() || !takesPlainOperands(*user))
			return false;
		std::optional<unsigned> const index = operandIndex(*user, _result);
		if (!index)
			return false;
		unsigned const count = user->getNumOperands();
		return *index + 1 == count || (count == 2 && *index == 0);
	}

	/// Whether caching @a _value would pay off: only instructions that take
	/// their operands the ordinary way can read it off the stack, and the cache
	/// is dropped before every call and every branch anyway.
	static bool hasCacheableUseInBlock(mlir::Value _value, mlir::Block* _block)
	{
		for (mlir::Operation* user: _value.getUsers())
			if (user->getBlock() == _block && takesPlainOperands(*user))
				return true;
		return false;
	}

	/// EVM pops the first operand first, so operands are pushed back to front.
	void pushOperands(mlir::Operation& _op)
	{
		int const count = static_cast<int>(_op.getNumOperands());

		// The previous instruction may have left its result on the stack; by
		// construction it is one of this op's operands and in a position that
		// costs at most one swap to reach.
		if (m_stackResident && count > 0)
		{
			mlir::Value const resident = m_stackResident;
			m_stackResident = nullptr;
			if (_op.getOperand(static_cast<unsigned>(count) - 1) == resident)
			{
				for (int i = count - 2; i >= 0; --i)
					pushValue(_op.getOperand(static_cast<unsigned>(i)));
				return;
			}
			if (count == 2 && _op.getOperand(0) == resident)
			{
				pushValue(_op.getOperand(1));
				emitSwap(1);
				return;
			}
			fail("a stack-resident value was not consumed by the next instruction");
		}

		for (int i = count - 1; i >= 0; --i)
			pushValue(_op.getOperand(static_cast<unsigned>(i)));
	}

	/// Commits the single result now on the stack: either kept there for the
	/// next instruction, or written to its frame slot.
	void finishResult(mlir::Operation& _op)
	{
		mlir::Value const result = _op.getResult(0);
		if (result.use_empty())
		{
			op(Instruction::POP); // nothing reads it, so it needs no slot
			return;
		}

		m_stack.back() = result;
		if (canStayOnStack(result, _op))
		{
			// Consumed by the very next instruction, so it never needs a slot.
			m_stackResident = result;
			return;
		}

		// Keeping a copy costs one DUP now and one POP at the flush, and saves
		// a reload at every later use in this block.
		bool const keepCached = hasCacheableUseInBlock(result, _op.getBlock())
			&& m_stack.size() < kMaxCachedStack;
		if (keepCached)
			emitDup(1);
		storeResult(result);
	}

	/// Copies @a _words words between a static frame address and the save
	/// region the pointer word currently points at.
	void copyFrame(uint64_t _frameAddress, uint64_t _words, bool _toSaveArea)
	{
		uint64_t const bytes = _words * kWord;
		if (m_options.evmVersion.hasMcopy())
		{
			push(u256(bytes));                      // length
			if (_toSaveArea)
			{
				push(u256(_frameAddress));          // source
				push(u256(m_savePointer));
				op(Instruction::MLOAD);             // destination
			}
			else
			{
				push(u256(m_savePointer));
				op(Instruction::MLOAD);             // source
				push(u256(_frameAddress));          // destination
			}
			op(Instruction::MCOPY);
			return;
		}

		// Without MCOPY the copy is unrolled, which is only reasonable for the
		// small frames recursive helpers tend to have.
		if (_words > kMaxUnrolledFrameWords)
			fail(
				"recursive function needs a " + std::to_string(_words)
				+ "-word frame copy, which requires an EVM version with MCOPY");
		for (uint64_t index = 0; index < _words; ++index)
		{
			uint64_t const offset = index * kWord;
			if (_toSaveArea)
			{
				push(u256(_frameAddress + offset));
				op(Instruction::MLOAD); // value
				push(u256(offset));
				push(u256(m_savePointer));
				op(Instruction::MLOAD);
				op(Instruction::ADD); // destination
			}
			else
			{
				push(u256(offset));
				push(u256(m_savePointer));
				op(Instruction::MLOAD);
				op(Instruction::ADD);
				op(Instruction::MLOAD);            // value
				push(u256(_frameAddress + offset)); // destination
			}
			op(Instruction::MSTORE);
		}
	}

	void adjustSavePointer(uint64_t _bytes, bool _grow)
	{
		push(u256(_bytes));
		push(u256(m_savePointer));
		op(Instruction::MLOAD);
		op(_grow ? Instruction::ADD : Instruction::SUB);
		storeToAddress(m_savePointer);
	}

	void emitFunction(mlir::func::FuncOp _func)
	{
		m_frame = &frameOf(_func);
		m_currentFunc = _func;

		std::vector<mlir::Block*> blocks;
		for (mlir::Block& block: _func.getBody())
			blocks.push_back(&block);

		for (size_t i = 0; i < blocks.size(); ++i)
		{
			mlir::Block* block = blocks[i];
			if (i == 0)
				m_assembly->append(m_functionTags.at(_func.getName().str()));
			m_assembly->append(m_blockTags.at(block));
			anchorStackHeight();
			m_next = (i + 1 < blocks.size()) ? blocks[i + 1] : nullptr;
			emitBlock(*block);
		}
		m_frame = nullptr;
	}

	void emitBlock(mlir::Block& _block)
	{
		m_stackResident = nullptr;
		m_stack.clear();
		for (mlir::Operation& op: _block.getOperations())
			emitOp(op);
		// Nothing may outlive the block: a value is only kept on the stack when
		// the very next instruction consumes it.
		if (m_stackResident)
			fail("a stack-resident value survived to the end of its block");
	}

	void emitOp(mlir::Operation& _op)
	{
		if (llvm::isa<mlir::arith::ConstantOp>(&_op))
			return; // rematerialised at each use
		if (_op.getNumResults() == 1 && isRematerializable(_op.getResult(0)))
			return; // likewise

		if (auto branch = llvm::dyn_cast<mlir::cf::BranchOp>(&_op))
			return emitBranch(branch);
		if (auto condBranch = llvm::dyn_cast<mlir::cf::CondBranchOp>(&_op))
			return emitCondBranch(condBranch);
		if (auto call = llvm::dyn_cast<mlir::func::CallOp>(&_op))
			return emitCall(call);
		if (auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(&_op))
			return emitReturn(ret);
		if (auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&_op))
			return emitCompare(cmp);
		if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(&_op))
			return emitSelect(select);

		if (std::optional<Instruction> instruction = arithInstruction(_op))
		{
			pushOperands(_op);
			op(*instruction);
			finishResult(_op);
			return;
		}

		// Width casts between the i1 that arith.cmpi produces and the i256 word
		// are representation-free: the EVM comparison already yields 0 or 1.
		if (llvm::isa<mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, mlir::arith::TruncIOp>(&_op))
		{
			pushOperands(_op);
			finishResult(_op);
			return;
		}

		if (_op.getDialect() && _op.getDialect()->getNamespace() == "evm")
			return emitEvmOp(_op);

		fail("no EVM lowering for op '" + _op.getName().getStringRef().str() + "'");
	}

	static std::optional<Instruction> arithInstruction(mlir::Operation& _op)
	{
		if (llvm::isa<mlir::arith::AddIOp>(&_op))
			return Instruction::ADD;
		if (llvm::isa<mlir::arith::SubIOp>(&_op))
			return Instruction::SUB;
		if (llvm::isa<mlir::arith::MulIOp>(&_op))
			return Instruction::MUL;
		if (llvm::isa<mlir::arith::AndIOp>(&_op))
			return Instruction::AND;
		if (llvm::isa<mlir::arith::OrIOp>(&_op))
			return Instruction::OR;
		if (llvm::isa<mlir::arith::XOrIOp>(&_op))
			return Instruction::XOR;
		return std::nullopt;
	}

	void emitCompare(mlir::arith::CmpIOp _cmp)
	{
		Instruction instruction = Instruction::EQ;
		bool negate = false;
		switch (_cmp.getPredicate())
		{
		case mlir::arith::CmpIPredicate::eq: instruction = Instruction::EQ; break;
		case mlir::arith::CmpIPredicate::ne: instruction = Instruction::EQ; negate = true; break;
		case mlir::arith::CmpIPredicate::ult: instruction = Instruction::LT; break;
		case mlir::arith::CmpIPredicate::ugt: instruction = Instruction::GT; break;
		case mlir::arith::CmpIPredicate::ule: instruction = Instruction::GT; negate = true; break;
		case mlir::arith::CmpIPredicate::uge: instruction = Instruction::LT; negate = true; break;
		case mlir::arith::CmpIPredicate::slt: instruction = Instruction::SLT; break;
		case mlir::arith::CmpIPredicate::sgt: instruction = Instruction::SGT; break;
		case mlir::arith::CmpIPredicate::sle: instruction = Instruction::SGT; negate = true; break;
		case mlir::arith::CmpIPredicate::sge: instruction = Instruction::SLT; negate = true; break;
		}
		pushOperands(*_cmp.getOperation());
		op(instruction);
		if (negate)
			op(Instruction::ISZERO);
		finishResult(*_cmp.getOperation());
	}

	static std::string stringAttr(mlir::Operation& _op, char const* _name)
	{
		auto attr = _op.getAttrOfType<mlir::StringAttr>(_name);
		if (!attr)
			fail(std::string("op '") + _op.getName().getStringRef().str() + "' is missing its '" + _name
				 + "' attribute");
		return attr.getValue().str();
	}

	/// A path naming an object inside one of our own sub-objects. Sub-assembly
	/// ids are handed out in registration order, so the position of each name
	/// in its parent's list is its id, and the chain encodes to a single id.
	void emitNestedSegmentQuery(
		mlir::Operation& _op, std::string const& _child, std::string const& _rest, bool _wantSize)
	{
		if (_rest.find('.') != std::string::npos)
			fail("segment path '" + _child + "." + _rest + "' nests deeper than this target resolves");

		std::vector<SubAssemblyID> path;
		for (size_t index = 0; index < m_options.subObjects.size(); ++index)
			if (m_options.subObjects[index].name == _child)
			{
				path.emplace_back(static_cast<SubAssemblyID::ValueType>(index));
				std::vector<std::string> const& grandchildren = m_options.subObjects[index].children;
				for (size_t inner = 0; inner < grandchildren.size(); ++inner)
					if (grandchildren[inner] == _rest)
					{
						path.emplace_back(static_cast<SubAssemblyID::ValueType>(inner));
						break;
					}
				break;
			}
		if (path.size() != 2)
			fail("object references unknown segment '" + _child + "." + _rest + "'");

		SubAssemblyID const encoded = m_assembly->encodeSubPath(path);
		if (_wantSize)
			m_assembly->pushSubroutineSize(encoded);
		else
			m_assembly->pushSubroutineOffset(encoded);
		modelPush();
		finishResult(_op);
	}

	/// `dataoffset`/`datasize` of a nested object or data segment. Offsets and
	/// sizes are only known once the whole assembly is laid out, so both become
	/// relocations rather than computed constants.
	void emitSegmentQuery(mlir::Operation& _op, bool _wantSize)
	{
		std::string segment = stringAttr(_op, "segment");

		// Segment names are paths relative to the enclosing object, so they may
		// be qualified with the name of the object doing the referring.
		std::string const ownPrefix = m_options.name + ".";
		if (segment.rfind(ownPrefix, 0) == 0)
			segment = segment.substr(ownPrefix.size());
		if (size_t const dot = segment.find('.'); dot != std::string::npos)
			return emitNestedSegmentQuery(_op, segment.substr(0, dot), segment.substr(dot + 1), _wantSize);

		if (auto sub = m_subObjects.find(segment); sub != m_subObjects.end())
		{
			if (_wantSize)
				m_assembly->pushSubroutineSize(sub->second);
			else
				m_assembly->pushSubroutineOffset(sub->second);
			modelPush();
		}
		else if (auto data = m_dataSegments.find(segment); data != m_dataSegments.end())
		{
			// A data item has no address of its own: `dataoffset` pushes the
			// item itself and `datasize` its length, which is known already.
			if (_wantSize)
				push(u256(m_assembly->data(solidity::util::h256(data->second.data())).size()));
			else
			{
				m_assembly->append(data->second);
				modelPush();
			}
		}
		else if (segment == m_options.name)
		{
			// An object may name itself: its offset is the origin it is already
			// addressed from, and its size is the whole assembly's.
			if (_wantSize)
			{
				m_assembly->appendProgramSize();
				modelPush();
			}
			else
				push(u256(0));
		}
		else
			fail("object references unknown segment '" + segment + "'");

		finishResult(_op);
	}

	/// The EVM has no select, but the condition is an i1 - so it is 0 or 1, and
	/// `b xor ((a xor b) * c)` picks between the arms without branching.
	/// Canonicalization produces these when it turns a diamond into a value.
	void emitSelect(mlir::arith::SelectOp _select)
	{
		if (!_select.getCondition().getType().isInteger(1))
			fail("arith.select with a non-i1 condition");

		pushValue(_select.getFalseValue());
		pushValue(_select.getTrueValue());
		op(Instruction::XOR);
		pushValue(_select.getCondition());
		op(Instruction::MUL);
		pushValue(_select.getFalseValue());
		op(Instruction::XOR);
		finishResult(*_select.getOperation());
	}

	void emitEvmOp(mlir::Operation& _op)
	{
		std::string mnemonic = _op.getName().stripDialect().str();

		if (mnemonic == "dataoffset" || mnemonic == "datasize")
			return emitSegmentQuery(_op, mnemonic == "datasize");
		if (mnemonic == "memoryguard")
		{
			// Hand the contract a heap that starts above our frames, or the
			// boundary unchanged when we did not move in below it.
			auto size = _op.getAttrOfType<mlir::IntegerAttr>("size");
			if (!size)
				fail("evm.memoryguard is missing its size attribute");
			push(m_relocatedFrames ? u256(m_heapBase) : toU256(size.getValue()));
			finishResult(_op);
			return;
		}
		if (mnemonic == "linkersymbol")
		{
			m_assembly->appendLibraryAddress(stringAttr(_op, "symbol"));
			modelPush();
			finishResult(_op);
			return;
		}
		if (mnemonic == "loadimmutable")
		{
			m_assembly->appendImmutable(stringAttr(_op, "immutable_name"));
			modelPush();
			finishResult(_op);
			return;
		}
		if (mnemonic == "setimmutable")
		{
			// Yul passes (offset, "name", value); the string is an attribute
			// here, so the remaining operands land offset-on-top, which is the
			// order AssignImmutable consumes them in.
			pushOperands(_op);
			m_assembly->appendImmutableAssignment(stringAttr(_op, "immutable_name"));
			modelPop(2); // consumes the offset and the value
			return;
		}
		if (mnemonic == "program")
			fail("evm.program is not emitted by this pipeline");

		// Copying a segment is a CODECOPY once its offset is a relocation.
		std::string name = (mnemonic == "datacopy") ? "CODECOPY" : mnemonic;
		for (char& c: name)
			c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
		auto it = c_instructions.find(name);
		if (it == c_instructions.end())
			fail("no EVM opcode named '" + name + "' for op 'evm." + mnemonic + "'");

		pushOperands(_op);
		op(it->second);

		if (_op.getNumResults() == 1)
			finishResult(_op);
		else if (_op.getNumResults() > 1)
			fail("op 'evm." + mnemonic + "' has more than one result");
	}

	/// Stores the values a successor's block arguments expect.
	///
	/// Every operand is read onto the stack before any slot is written. A
	/// branch back to its own block passes the block's arguments as operands,
	/// so storing eagerly would clobber a value a later operand still needs.
	void passBlockArguments(std::vector<std::pair<mlir::Block*, mlir::ValueRange>> const& _edges)
	{
		std::vector<uint64_t> destinations;
		for (auto const& [target, operands]: _edges)
			for (unsigned i = 0; i < operands.size(); ++i)
			{
				pushValue(operands[i]);
				destinations.push_back(addressOf(target->getArgument(i)));
			}
		// The last value pushed is on top, so unwind in reverse.
		for (size_t i = destinations.size(); i > 0; --i)
			storeToAddress(destinations[i - 1]);
	}

	void emitBranch(mlir::cf::BranchOp _branch)
	{
		flushStack();
		passBlockArguments({{_branch.getDest(), _branch.getDestOperands()}});
		if (_branch.getDest() == m_next)
			return; // falls through into the next block
		m_assembly->appendJump(m_blockTags.at(_branch.getDest()));
	}

	void emitCondBranch(mlir::cf::CondBranchOp _branch)
	{
		// JUMPI consumes only the condition, so anything cached underneath it
		// would survive into the successor, which expects a bare stack.
		flushStack();

		// Canonicalization merges the arms of a diamond, which can leave both
		// edges pointing at one block with different arguments. There is no
		// branch left to take then - the arguments themselves are the choice.
		if (_branch.getTrueDest() == _branch.getFalseDest())
		{
			emitMergedBranch(_branch);
			return;
		}

		passBlockArguments(
			{{_branch.getTrueDest(), _branch.getTrueDestOperands()},
			 {_branch.getFalseDest(), _branch.getFalseDestOperands()}});

		pushValue(_branch.getCondition());
		m_assembly->appendJumpI(m_blockTags.at(_branch.getTrueDest()));
		modelPop(1); // JUMPI consumed the condition
		if (_branch.getFalseDest() == m_next)
			return;
		m_assembly->appendJump(m_blockTags.at(_branch.getFalseDest()));
	}

	/// A conditional branch whose arms are the same block: pick each argument
	/// with the branch-free select and then jump unconditionally.
	void emitMergedBranch(mlir::cf::CondBranchOp _branch)
	{
		mlir::Block* target = _branch.getTrueDest();
		mlir::ValueRange const onTrue = _branch.getTrueDestOperands();
		mlir::ValueRange const onFalse = _branch.getFalseDestOperands();

		std::vector<uint64_t> destinations;
		for (unsigned i = 0; i < onTrue.size(); ++i)
		{
			if (onTrue[i] == onFalse[i])
				pushValue(onTrue[i]);
			else
			{
				pushValue(onFalse[i]);
				pushValue(onTrue[i]);
				op(Instruction::XOR);
				pushValue(_branch.getCondition());
				op(Instruction::MUL);
				pushValue(onFalse[i]);
				op(Instruction::XOR);
			}
			destinations.push_back(addressOf(target->getArgument(i)));
		}
		for (size_t i = destinations.size(); i > 0; --i)
			storeToAddress(destinations[i - 1]);

		if (target != m_next)
			m_assembly->appendJump(m_blockTags.at(target));
	}

	void emitCall(mlir::func::CallOp _call)
	{
		// The callee runs with our stack underneath it and cannot be asked to
		// preserve a cache it knows nothing about; dropping it here also keeps
		// the depth of a call chain independent of what each frame had cached.
		flushStack();

		std::string callee = _call.getCallee().str();
		auto target = m_byName.find(callee);
		if (target == m_byName.end())
			fail("call to unknown function '" + callee + "'");
		Frame const& calleeFrame = m_frames.at(callee);

		if (_call.getNumOperands() != calleeFrame.argSlots.size())
			fail("argument count mismatch calling '" + callee + "'");

		// A function that can re-enter itself would overwrite the live frame of
		// the activation below it, so the caller banks it first.
		bool const savesFrame = m_recursive.count(callee) != 0 && calleeFrame.size > 0;
		if (savesFrame)
		{
			copyFrame(calleeFrame.base, calleeFrame.size, /*toSaveArea=*/true);
			adjustSavePointer(calleeFrame.size * kWord, /*grow=*/true);
		}

		// In a self-call the argument slots being written belong to the frame
		// the arguments are read from, so read every one before writing any.
		for (unsigned i = 0; i < _call.getNumOperands(); ++i)
			pushValue(_call.getOperand(i));
		for (unsigned i = _call.getNumOperands(); i > 0; --i)
			storeToAddress(calleeFrame.base + calleeFrame.argSlots[i - 1] * kWord);

		AssemblyItem returnTag = m_assembly->newTag();
		m_assembly->append(returnTag.pushTag());
		m_assembly->appendJump(m_functionTags.at(callee));
		m_assembly->append(returnTag);
		anchorStackHeight(); // the callee consumed the return address

		// Results are read out of the callee's frame before the restore puts it
		// back, but they are parked on the stack rather than in a slot: in a
		// self-call the destination slot lives in the frame being restored and
		// would be overwritten again.
		for (unsigned i = 0; i < _call.getNumResults(); ++i)
		{
			push(u256(calleeFrame.base + calleeFrame.resultSlots[i] * kWord));
			op(Instruction::MLOAD);
		}

		if (savesFrame)
		{
			adjustSavePointer(calleeFrame.size * kWord, /*grow=*/false);
			copyFrame(calleeFrame.base, calleeFrame.size, /*toSaveArea=*/false);
		}

		for (unsigned i = _call.getNumResults(); i > 0; --i)
			storeResult(_call.getResult(i - 1));
	}

	void emitReturn(mlir::func::ReturnOp _return)
	{
		flushStack();
		for (unsigned i = 0; i < _return.getNumOperands(); ++i)
		{
			pushValue(_return.getOperand(i));
			storeToAddress(m_frame->base + m_frame->resultSlots[i] * kWord);
		}

		// The object entry is not called, so there is no return address to
		// jump back to - falling off the end of the object code halts.
		// The return address was pushed by the caller, below this frame's
		// baseline, so the jump consuming it is invisible to the model.
		m_assembly->append(isEntry(m_currentFunc) ? Instruction::STOP : Instruction::JUMP);
	}

	solidity::mlirgen::EVMAssemblyOptions m_options;
	std::shared_ptr<Assembly> m_assembly;

	std::vector<mlir::func::FuncOp> m_functions;
	std::map<std::string, mlir::func::FuncOp> m_byName;
	std::map<std::string, Frame> m_frames;
	std::map<std::string, AssemblyItem> m_functionTags;
	std::map<mlir::Block*, AssemblyItem> m_blockTags;
	std::map<std::string, SubAssemblyID> m_subObjects;
	std::map<std::string, AssemblyItem> m_dataSegments;
	std::set<std::string> m_recursive;
	uint64_t m_savePointer = 0;
	uint64_t m_saveBase = 0;

	mlir::ModuleOp m_module;
	/// Where the frames live, and where the contract's heap may start above them.
	uint64_t m_frameBase = 0;
	uint64_t m_heapBase = 0;
	bool m_relocatedFrames = false;
	/// Physical slots this block pushed above its baseline; back() is the top.
	std::vector<mlir::Value> m_stack;
	/// Result of the previous instruction, left on the stack for the next one.
	mlir::Value m_stackResident;
	Frame const* m_frame = nullptr;
	mlir::func::FuncOp m_currentFunc;
	mlir::Block* m_next = nullptr;
};

} // anonymous namespace

std::shared_ptr<Assembly> solidity::mlirgen::emitEVMAssembly(
	mlir::ModuleOp _module,
	EVMAssemblyOptions const& _options,
	std::string& _error)
{
	try
	{
		Emitter emitter(_options);
		return emitter.run(_module);
	}
	catch (EmitError const& _e)
	{
		_error = _e.message;
		return nullptr;
	}
	catch (std::exception const& _e)
	{
		_error = std::string("assembly emission failed: ") + _e.what();
		return nullptr;
	}
}
