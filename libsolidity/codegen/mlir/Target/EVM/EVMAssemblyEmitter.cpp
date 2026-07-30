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
		if (m_functions.empty())
			fail("module contains no functions");
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

	void layoutFrames()
	{
		uint64_t next = m_options.frameBase;
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

	void push(u256 const& _value) { m_assembly->append(AssemblyItem(_value)); }
	void op(Instruction _instruction) { m_assembly->append(_instruction); }

	/// Assembly tracks one net stack height across the whole item stream, which
	/// only describes straight-line code: a function's return JUMP consumes an
	/// address pushed by a caller that is somewhere else entirely in the
	/// stream, so the running count drifts and eventually trips its own
	/// underflow assertion. Re-anchor it wherever this backend's own invariant
	/// says the stack is back at rest.
	void anchorStackHeight() { m_assembly->setDeposit(1); }

	/// Leaves @a _value on the stack.
	void pushValue(mlir::Value _value)
	{
		if (std::optional<llvm::APInt> constant = constantOf(_value))
		{
			push(toU256(*constant));
			return;
		}
		push(u256(addressOf(_value)));
		op(Instruction::MLOAD);
	}

	/// Consumes the stack top, writing it to @a _address.
	void storeToAddress(uint64_t _address)
	{
		push(u256(_address));
		op(Instruction::MSTORE);
	}

	void storeResult(mlir::Value _value) { storeToAddress(addressOf(_value)); }

	/// EVM pops the first operand first, so operands are pushed back to front.
	void pushOperands(mlir::Operation& _op)
	{
		for (int i = static_cast<int>(_op.getNumOperands()) - 1; i >= 0; --i)
			pushValue(_op.getOperand(static_cast<unsigned>(i)));
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
		for (mlir::Operation& op: _block.getOperations())
			emitOp(op);
	}

	void emitOp(mlir::Operation& _op)
	{
		if (llvm::isa<mlir::arith::ConstantOp>(&_op))
			return; // rematerialised at each use

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

		if (std::optional<Instruction> instruction = arithInstruction(_op))
		{
			pushOperands(_op);
			op(*instruction);
			storeResult(_op.getResult(0));
			return;
		}

		// Width casts between the i1 that arith.cmpi produces and the i256 word
		// are representation-free: the EVM comparison already yields 0 or 1.
		if (llvm::isa<mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, mlir::arith::TruncIOp>(&_op))
		{
			pushValue(_op.getOperand(0));
			storeResult(_op.getResult(0));
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
		pushValue(_cmp.getRhs());
		pushValue(_cmp.getLhs());
		op(instruction);
		if (negate)
			op(Instruction::ISZERO);
		storeResult(_cmp.getResult());
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
		storeResult(_op.getResult(0));
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
		}
		else if (auto data = m_dataSegments.find(segment); data != m_dataSegments.end())
		{
			// A data item has no address of its own: `dataoffset` pushes the
			// item itself and `datasize` its length, which is known already.
			if (_wantSize)
				push(u256(m_assembly->data(solidity::util::h256(data->second.data())).size()));
			else
				m_assembly->append(data->second);
		}
		else if (segment == m_options.name)
		{
			// An object may name itself: its offset is the origin it is already
			// addressed from, and its size is the whole assembly's.
			if (_wantSize)
				m_assembly->appendProgramSize();
			else
				push(u256(0));
		}
		else
			fail("object references unknown segment '" + segment + "'");

		storeResult(_op.getResult(0));
	}

	void emitEvmOp(mlir::Operation& _op)
	{
		std::string mnemonic = _op.getName().stripDialect().str();

		if (mnemonic == "dataoffset" || mnemonic == "datasize")
			return emitSegmentQuery(_op, mnemonic == "datasize");
		if (mnemonic == "linkersymbol")
		{
			m_assembly->appendLibraryAddress(stringAttr(_op, "symbol"));
			storeResult(_op.getResult(0));
			return;
		}
		if (mnemonic == "loadimmutable")
		{
			m_assembly->appendImmutable(stringAttr(_op, "immutable_name"));
			storeResult(_op.getResult(0));
			return;
		}
		if (mnemonic == "setimmutable")
		{
			// Yul passes (offset, "name", value); the string is an attribute
			// here, so the remaining operands land offset-on-top, which is the
			// order AssignImmutable consumes them in.
			pushOperands(_op);
			m_assembly->appendImmutableAssignment(stringAttr(_op, "immutable_name"));
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
			storeResult(_op.getResult(0));
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
		passBlockArguments({{_branch.getDest(), _branch.getDestOperands()}});
		if (_branch.getDest() == m_next)
			return; // falls through into the next block
		m_assembly->appendJump(m_blockTags.at(_branch.getDest()));
	}

	void emitCondBranch(mlir::cf::CondBranchOp _branch)
	{
		// Both edges write their target's argument slots unconditionally, which
		// is only sound while the targets are distinct.
		if (_branch.getTrueDest() == _branch.getFalseDest()
			&& _branch.getTrueDestOperands() != _branch.getFalseDestOperands())
			fail("conditional branch with a shared destination and differing arguments");

		passBlockArguments(
			{{_branch.getTrueDest(), _branch.getTrueDestOperands()},
			 {_branch.getFalseDest(), _branch.getFalseDestOperands()}});

		pushValue(_branch.getCondition());
		m_assembly->appendJumpI(m_blockTags.at(_branch.getTrueDest()));
		if (_branch.getFalseDest() == m_next)
			return;
		m_assembly->appendJump(m_blockTags.at(_branch.getFalseDest()));
	}

	void emitCall(mlir::func::CallOp _call)
	{
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
		for (unsigned i = 0; i < _return.getNumOperands(); ++i)
		{
			pushValue(_return.getOperand(i));
			storeToAddress(m_frame->base + m_frame->resultSlots[i] * kWord);
		}

		// The object entry is not called, so there is no return address to
		// jump back to - falling off the end of the object code halts.
		if (isEntry(m_currentFunc))
			op(Instruction::STOP);
		else
			op(Instruction::JUMP);
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
