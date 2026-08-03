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
/**
 * SolToYul conversion implementation.
 *
 * The conversion walks sol.contract containers, assigns sequential storage
 * slots to sol.state_var declarations, and converts each sol.func body:
 *
 *  - value representation: everything becomes an i256 word; narrowing
 *    conversions mask (unsigned/address), sign-extend (signed) or normalize
 *    to 0/1 (bool);
 *  - arithmetic maps 1:1 onto yul builtins (unchecked EVM semantics -
 *    checked-arithmetic overflow guards are a follow-up, matching the old
 *    pipeline's behavior);
 *  - sol.cmp's string predicate maps onto lt/gt/slt/sgt/eq (+iszero for the
 *    negated/or-equal forms);
 *  - sol.if with an else region lowers to two exclusive yul.ifs over the
 *    pre-computed condition (Yul has no else);
 *  - sol.for's init region is hoisted, the last value-producing op of the
 *    condition region becomes yul.condition;
 *  - sol.return -> yul.leave, sol.require -> if iszero(c) revert(0,0),
 *    sol.revert -> yul.revert(0,0) (message ABI encoding is a follow-up).
 */

#include "SolToYul.h"

#include "SolidityDialect.h"
#include "SolidityOps.h"
#include "YulDialect.h"
#include "YulOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

#include <libsolutil/FunctionSelector.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

using mlir::failed;
using mlir::succeeded;

namespace
{

struct ConversionError
{
	std::string message;
};

/// Name the creation half uses to address the runtime half, and the name
/// sol2evm registers that half's assembly under.
constexpr char kRuntimeObjectName[] = "runtime";

/// Set by the conversion, read by whoever assembles the object. A single
/// conversion runs at a time, so this needs no more than file scope.
std::set<std::string>& referencedContracts()
{
	static std::set<std::string> objects;
	return objects;
}

class SolToYulConverter
{
public:
	SolToYulConverter(mlir::MLIRContext& _ctx, bool _creation): m_builder(&_ctx), m_creation(_creation) {}

	mlir::OwningOpRef<mlir::ModuleOp> run(mlir::ModuleOp _src, std::string& _error)
	{
		mlir::OwningOpRef<mlir::ModuleOp> dst = mlir::ModuleOp::create(m_builder.getUnknownLoc());
		try
		{
			// Every contract's functions have to be known before any body is
			// converted: a call now names the contract that defines it, and
			// `Base.value` is reachable from `Derived` even though it lives in
			// a different sol.contract.
			for (mlir::Operation& op: _src.getBody()->getOperations())
				if (auto contract = llvm::dyn_cast<mlir::solidity::ContractOp>(&op))
					for (mlir::Operation& inner: contract.getBody().front().getOperations())
						if (auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&inner))
						{
							auto kindAttr = func->getAttrOfType<mlir::StringAttr>("kind");
							std::string const bare = (kindAttr && kindAttr.getValue() == "constructor")
								? std::string("constructor")
								: func.getSymName().str();
							m_declaredFunctions.insert(
								bare.find('.') != std::string::npos
									? bare
									: contract.getName().str() + "." + bare);
						}

			for (mlir::Operation& op: _src.getBody()->getOperations())
			{
				if (auto contract = llvm::dyn_cast<mlir::solidity::ContractOp>(&op))
					convertContract(contract, *dst);
				else if (auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&op))
				{
					m_builder.setInsertionPointToEnd(dst->getBody());
					convertFunction(func);
				}
				else
					fail("unsupported top-level operation: " + op.getName().getStringRef().str());
			}
		}
		catch (ConversionError const& _e)
		{
			_error = _e.message;
			return nullptr;
		}
		if (failed(mlir::verify(*dst)))
		{
			_error = "converted module failed MLIR verification";
			return nullptr;
		}
		return dst;
	}

private:
	mlir::OpBuilder m_builder;
	llvm::DenseMap<mlir::Value, mlir::Value> m_map;
	llvm::StringMap<uint64_t> m_storageSlots;
	bool m_usesMemory = false;
	bool m_needsBytesHelpers = false;
	/// Objects this contract's code names, which have to be nested in it.
	std::set<std::string>& m_referencedObjects = referencedContracts();

	/// `string` and `bytes` do not fit a word, so they live in memory and are
	/// named by a pointer to [length][data...].
	static bool isDynamic(mlir::Type _type)
	{
		return llvm::isa<mlir::solidity::StringType, mlir::solidity::DynamicBytesType>(_type);
	}
	llvm::StringSet<> m_declaredFunctions;
	llvm::StringMap<unsigned> m_constructorArity;
	std::string m_contractName;
	bool m_creation = false;

	[[noreturn]] static void fail(std::string _message) { throw ConversionError{std::move(_message)}; }

	mlir::Location loc() { return m_builder.getUnknownLoc(); }
	mlir::IntegerType wordType() { return m_builder.getIntegerType(256); }

	mlir::Value mapped(mlir::Value _v)
	{
		auto it = m_map.find(_v);
		if (it == m_map.end())
			fail("use of an unmapped value during sol->yul conversion");
		return it->second;
	}

	mlir::Value wordConstant(llvm::APInt _value)
	{
		return m_builder.create<mlir::yul::ConstOp>(loc(), mlir::IntegerAttr::get(wordType(), std::move(_value)));
	}

	mlir::Value wordConstant(uint64_t _value) { return wordConstant(llvm::APInt(256, _value)); }

	//===------------------------------------------------------------------===//
	// Structure
	//===------------------------------------------------------------------===//

	/// Yul has one flat namespace, but a file's contracts all land in one
	/// module and nothing stops two of them defining `value`. Qualifying the
	/// symbol by its contract is what keeps them apart; the ABI name is taken
	/// from the unqualified part, so selectors are unaffected.
	std::string qualified(llvm::StringRef _name) const { return m_contractName + "." + _name.str(); }

	void convertContract(mlir::solidity::ContractOp _contract, mlir::ModuleOp _dst)
	{
		m_contractName = _contract.getName().str();
		// Storage layout: sequential slots in declaration order (the real
		// pipeline uses ContractType::linearizedStateVariables(); the
		// generator emits state_var ops in that order).
		uint64_t nextSlot = 0;
		for (mlir::Operation& op: _contract.getBody().front().getOperations())
			if (auto stateVar = llvm::dyn_cast<mlir::solidity::StateVarOp>(&op))
				m_storageSlots[stateVar.getName()] = nextSlot++;

		// A call can name a function this contract does not contain - one that
		// is inherited but never emitted here, or reached through `super`.
		// Knowing the set up front is what lets the call lowering refuse
		// instead of building a reference that nothing resolves.
		for (mlir::Operation& op: _contract.getBody().front().getOperations())
			if (auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&op))
				m_declaredFunctions.insert(qualified(func.getSymName()));

		for (mlir::Operation& op: _contract.getBody().front().getOperations())
		{
			if (llvm::isa<mlir::solidity::StateVarOp>(&op))
				continue; // layout only; initial values are constructor work
			if (auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&op))
			{
				m_builder.setInsertionPointToEnd(_dst.getBody());
				convertFunction(func);
				continue;
			}
			fail("unsupported op in contract body: " + op.getName().getStringRef().str());
		}

		if (m_creation)
			emitCreation(_contract, _dst);
		else
			emitDispatcher(_contract, _dst);

		// Emitted last because it is the conversion of the bodies above that
		// says whether anything needs them.
		if (m_needsBytesHelpers)
		{
			emitLoadBytesHelper(_dst);
			emitStoreBytesHelper(_dst);
		}

		// The free memory pointer has to hold the start of the heap before the
		// first allocation, so this goes at the very front of the object -
		// which is only known to be needed once the bodies are converted. A
		// contract that never allocates does not pay for it.
		if (m_usesMemory)
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(_dst.getBody());
			initialiseFreeMemoryPointer();
		}
	}

	/// The creation half. State variables with an initialiser are stored to
	/// their slots, the constructor body runs if there is one, and then the
	/// runtime half is copied out of the object's data and returned - which is
	/// what makes it the deployed code.
	void emitCreation(mlir::solidity::ContractOp _contract, mlir::ModuleOp _dst)
	{
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToEnd(_dst.getBody());

		// The generator emits every initialiser as a function rather than as a
		// literal on the declaration, because only a plain integer fits in the
		// latter - `keccak256("x")` and everything else read zero.
		std::string const initializerName = qualified("init");
		if (m_declaredFunctions.contains(initializerName))
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), initializerName),
				mlir::ValueRange{});

		// A constructor body was converted as a function; call it.
		//
		// Only a parameterless one. Constructor arguments arrive appended to
		// the creation code and need the ABI decoding that does not exist yet,
		// and calling with the wrong arity builds a call the verifier rejects -
		// which would cost the whole creation half.
		std::string const constructorName = qualified("constructor");
		auto arity = m_constructorArity.find(constructorName);
		if (m_declaredFunctions.contains(constructorName))
		{
			if (arity != m_constructorArity.end() && arity->second > 0)
				fail("constructor takes arguments, which the creation code cannot decode yet");
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), constructorName),
				mlir::ValueRange{});
		}

		mlir::Value size = m_builder.create<mlir::yul::DataSizeOp>(loc(), kRuntimeObjectName);
		mlir::Value offset = m_builder.create<mlir::yul::DataOffsetOp>(loc(), kRuntimeObjectName);
		mlir::Value zero = wordConstant(uint64_t(0));
		m_builder.create<mlir::yul::DataCopyOp>(loc(), zero, offset, size);
		m_builder.create<mlir::yul::ReturnOp>(loc(), zero, size);
	}

	void convertFunction(mlir::solidity::FunctionOp _func)
	{
		mlir::FunctionType solType = _func.getFunctionType();
		llvm::SmallVector<mlir::Type, 4> paramTypes(solType.getNumInputs(), wordType());
		llvm::SmallVector<mlir::Type, 2> resultTypes(solType.getNumResults(), wordType());
		auto yulType = mlir::FunctionType::get(m_builder.getContext(), paramTypes, resultTypes);

		std::string const kind
			= _func->getAttrOfType<mlir::StringAttr>("kind") ? _func->getAttrOfType<mlir::StringAttr>("kind").str() : "";
		if (kind == "constructor")
		{
			// Constructors reach the module under a name of their own, since
			// the generator leaves sym_name empty for them.
			mlir::FunctionType const solType0 = _func.getFunctionType();
			auto ctorType = mlir::FunctionType::get(
				m_builder.getContext(),
				llvm::SmallVector<mlir::Type, 4>(solType0.getNumInputs(), wordType()),
				llvm::SmallVector<mlir::Type, 2>(solType0.getNumResults(), wordType()));
			m_constructorArity[qualified("constructor")] = solType0.getNumInputs();
			auto ctor = m_builder.create<mlir::yul::FuncOp>(loc(), qualified("constructor"), ctorType);
			mlir::Block* ctorBody = &ctor.getBody().emplaceBlock();
			mlir::OpBuilder::InsertionGuard ctorGuard(m_builder);
			m_builder.setInsertionPointToStart(ctorBody);
			bool ctorTerminated = false;
			if (!_func.getBody().empty())
			{
				mlir::Block& srcEntry0 = _func.getBody().front();
				for (unsigned i = 0; i < srcEntry0.getNumArguments(); ++i)
					m_map[srcEntry0.getArgument(i)] = ctorBody->addArgument(wordType(), loc());
				ctorTerminated = convertBlockOps(srcEntry0);
			}
			if (!ctorTerminated)
				m_builder.create<mlir::yul::LeaveOp>(loc(), mlir::ValueRange{});
			return;
		}

		bool const preQualified = _func.getSymName().contains('.');
		auto yulFunc = m_builder.create<mlir::yul::FuncOp>(
			loc(),
			(preQualified || m_contractName.empty()) ? _func.getSymName().str() : qualified(_func.getSymName()),
			yulType);
		mlir::Block* body = &yulFunc.getBody().emplaceBlock();

		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(body);

		bool terminated = false;
		if (!_func.getBody().empty())
		{
			mlir::Block& srcEntry = _func.getBody().front();
			for (unsigned i = 0; i < srcEntry.getNumArguments(); ++i)
				m_map[srcEntry.getArgument(i)] = body->addArgument(wordType(), loc());
			terminated = convertBlockOps(srcEntry);
		}
		if (!terminated)
		{
			// Fell off the end: leave with zero-initialized results.
			llvm::SmallVector<mlir::Value, 2> zeros;
			for (unsigned i = 0; i < resultTypes.size(); ++i)
				zeros.push_back(wordConstant(uint64_t(0)));
			m_builder.create<mlir::yul::LeaveOp>(loc(), zeros);
		}
	}

	//===------------------------------------------------------------------===//
	// Statements / operations
	//===------------------------------------------------------------------===//

	/// @returns true if the block ended in a terminator.
	bool convertBlockOps(mlir::Block& _src)
	{
		for (mlir::Operation& op: _src.getOperations())
			if (convertOp(op))
				return true;
		return false;
	}

	bool convertOp(mlir::Operation& _op)
	{
		if (auto constant = llvm::dyn_cast<mlir::solidity::ConstantOp>(&_op))
		{
			m_map[constant.getResult()] = convertConstant(constant);
			return false;
		}
		if (auto length = llvm::dyn_cast<mlir::solidity::MemoryLengthOp>(&_op))
		{
			m_map[length.getResult()] = m_builder.create<mlir::yul::MLoadOp>(loc(), mapped(length.getValue()));
			return false;
		}
		if (auto code = llvm::dyn_cast<mlir::solidity::ContractCodeOp>(&_op))
		{
			// The named contract is nested as a sub-object, so its code is data
			// here: its size is known when the object is assembled, and copying
			// it into memory makes it an ordinary `bytes` value.
			// A dotted name is a path to a nested object, so the sub-object
			// this refers to is named after the contract and holds its runtime
			// code directly.
			if (code.getCreation())
				fail("type(C).creationCode");
			std::string const object = code.getContractName().str();
			m_referencedObjects.insert(object);
			m_usesMemory = true;

			mlir::Value size = m_builder.create<mlir::yul::DataSizeOp>(loc(), object);
			mlir::Value pointer = allocate(
				m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(size)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, size);
			m_builder.create<mlir::yul::DataCopyOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::DataOffsetOp>(loc(), object),
				size);
			m_map[code.getResult()] = pointer;
			return false;
		}
		if (auto literal = llvm::dyn_cast<mlir::solidity::StringLiteralOp>(&_op))
		{
			m_map[literal.getResult()] = materialiseLiteral(literal.getValue());
			m_usesMemory = true;
			return false;
		}
		if (auto load = llvm::dyn_cast<mlir::solidity::LoadStateVarOp>(&_op))
		{
			mlir::Value slot = wordConstant(storageSlot(load.getVarName()));
			// A dynamic value does not fit a slot: the slot holds its length
			// and, past 31 bytes, only a pointer to where the data starts.
			if (isDynamic(load.getResult().getType()))
			{
				m_needsBytesHelpers = m_usesMemory = true;
				m_map[load.getResult()] = m_builder.create<mlir::yul::FuncCallOp>(
					loc(),
					mlir::TypeRange{wordType()},
					mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
					mlir::ValueRange{slot})->getResult(0);
				return false;
			}
			m_map[load.getResult()] = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::StoreStateVarOp>(&_op))
		{
			mlir::Value slot = wordConstant(storageSlot(store.getVarName()));
			if (isDynamic(store.getValue().getType()))
			{
				m_needsBytesHelpers = m_usesMemory = true;
				m_builder.create<mlir::yul::FuncCallOp>(
					loc(),
					mlir::TypeRange{},
					mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
					mlir::ValueRange{slot, mapped(store.getValue())});
				return false;
			}
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, mapped(store.getValue()));
			return false;
		}
		if (convertBinaryArith(_op))
			return false;
		if (auto notOp = llvm::dyn_cast<mlir::solidity::NotOp>(&_op))
		{
			// sol.not on bool is logical negation; on integers bitwise.
			if (llvm::isa<mlir::solidity::BoolType>(notOp.getOperand().getType()))
				m_map[notOp.getResult()] = m_builder.create<mlir::yul::IsZeroOp>(loc(), mapped(notOp.getOperand()));
			else
				m_map[notOp.getResult()] = m_builder.create<mlir::yul::NotOp>(loc(), mapped(notOp.getOperand()));
			return false;
		}
		if (auto cmp = llvm::dyn_cast<mlir::solidity::CmpOp>(&_op))
		{
			m_map[cmp.getResult()] = convertCmp(cmp);
			return false;
		}
		if (auto convert = llvm::dyn_cast<mlir::solidity::ConvertOp>(&_op))
		{
			m_map[convert.getOutput()] = representationConvert(mapped(convert.getInput()), convert.getOutput().getType());
			return false;
		}
		if (auto toI1 = llvm::dyn_cast<mlir::solidity::ToI1Op>(&_op))
		{
			// Everything is a word here, so narrowing a bool to i1 is only a
			// change of type - the value is already 0 or 1.
			m_map[toI1.getResult()] = mapped(toI1.getOperand());
			return false;
		}
		if (auto whileOp = llvm::dyn_cast<mlir::scf::WhileOp>(&_op))
		{
			convertWhile(whileOp);
			return false;
		}
		if (auto ifOp = llvm::dyn_cast<mlir::solidity::IfOp>(&_op))
		{
			convertIf(ifOp);
			return false;
		}
		if (auto forOp = llvm::dyn_cast<mlir::solidity::ForOp>(&_op))
		{
			convertFor(forOp);
			return false;
		}
		if (auto access = llvm::dyn_cast<mlir::solidity::MappingAccessOp>(&_op))
		{
			mlir::Value slot = mappingSlot(storageSlot(access.getVarName()), access.getKeys());
			m_map[access.getResult()] = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::MappingStoreOp>(&_op))
		{
			// The operands are the keys followed by the value; only the count
			// of keys says where the split is.
			unsigned const keyCount = static_cast<unsigned>(store.getNumKeys());
			if (store.getOperands().size() != keyCount + 1)
				fail("mapping_store operand count disagrees with its numKeys attribute");
			mlir::Value slot
				= mappingSlot(storageSlot(store.getVarName()), store.getOperands().take_front(keyCount));
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, mapped(store.getOperands().back()));
			return false;
		}
		if (auto call = llvm::dyn_cast<mlir::solidity::FunctionCallOp>(&_op))
		{
			// Internal calls map straight onto yul.func_call; only the callee
			// attribute differs, a plain string here and a symbol reference
			// there. Everything is a word at the yul rung, so the result types
			// come from the arity rather than from the Solidity types.
			// A call names a function of the contract being converted; anything
			// else - inherited but not emitted here, or reached through
			// `super` - has no symbol to bind to.
			// The generator names another contract's function in full; an
			// unqualified name is one of this contract's own.
			std::string const callee = call.getCallee().contains('.')
				? call.getCallee().str()
				: qualified(call.getCallee());
			if (!m_declaredFunctions.contains(callee))
				fail("call to a function this contract does not define: " + call.getCallee().str());

			llvm::SmallVector<mlir::Value, 4> arguments;
			for (mlir::Value argument: call.getArgs())
				arguments.push_back(mapped(argument));

			llvm::SmallVector<mlir::Type, 2> resultTypes(call.getNumResults(), wordType());
			auto lowered = m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				resultTypes,
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), callee),
				arguments);
			for (unsigned i = 0; i < call.getNumResults(); ++i)
				m_map[call.getResult(i)] = lowered.getResult(i);
			return false;
		}
		if (auto ret = llvm::dyn_cast<mlir::solidity::ReturnOp>(&_op))
		{
			llvm::SmallVector<mlir::Value, 2> results;
			for (mlir::Value operand: ret.getOperands())
				results.push_back(mapped(operand));
			m_builder.create<mlir::yul::LeaveOp>(loc(), results);
			return true;
		}
		if (auto require = llvm::dyn_cast<mlir::solidity::RequireOp>(&_op))
		{
			// require(c): if iszero(c) { revert(0, 0) }.
			// TODO: ABI-encode the message into revert data (Error(string)).
			mlir::Value failed = m_builder.create<mlir::yul::IsZeroOp>(loc(), mapped(require.getCondition()));
			emitRevertIf(failed);
			return false;
		}
		if (auto assertOp = llvm::dyn_cast<mlir::solidity::AssertOp>(&_op))
		{
			// assert(c): if iszero(c) { invalid-style panic }. Scaffold uses
			// revert(0,0); Panic(0x01) ABI encoding is a follow-up.
			mlir::Value failed = m_builder.create<mlir::yul::IsZeroOp>(loc(), mapped(assertOp.getCondition()));
			emitRevertIf(failed);
			return false;
		}
		if (llvm::isa<mlir::solidity::RevertOp>(&_op))
		{
			mlir::Value zero = wordConstant(uint64_t(0));
			m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
			return true;
		}
		fail("unsupported sol op in sol->yul conversion: " + _op.getName().getStringRef().str());
	}

	/// Solidity keeps the free memory pointer at 0x40 and starts the heap at
	/// 0x80, leaving 0x00-0x3f as scratch (which the mapping slot derivation
	/// already uses) and 0x60 as the zero slot. Nothing here allocated before,
	/// because nothing needed memory that outlived an expression.
	static constexpr uint64_t kFreeMemoryPointer = 0x40;
	static constexpr uint64_t kHeapStart = 0x80;

	void initialiseFreeMemoryPointer()
	{
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), wordConstant(kFreeMemoryPointer), wordConstant(kHeapStart));
	}

	/// Bumps the free memory pointer by `_size` and yields the old value.
	/// Solidity never frees, so this is the whole allocator.
	mlir::Value allocate(mlir::Value _size)
	{
		mlir::Value pointer = m_builder.create<mlir::yul::MLoadOp>(loc(), wordConstant(kFreeMemoryPointer));
		mlir::Value next = m_builder.create<mlir::yul::AddOp>(loc(), pointer, _size);
		m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(kFreeMemoryPointer), next);
		return pointer;
	}

	/// `_value` rounded up to a whole number of words.
	mlir::Value roundedUp(mlir::Value _value)
	{
		mlir::Value padded = m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(31)));
		return m_builder.create<mlir::yul::AndOp>(
			loc(), padded, wordConstant(~llvm::APInt(256, 31)));
	}

	/// A string or bytes literal written into fresh memory as Solidity holds
	/// them: the length in the first word, the bytes after it. The length is
	/// known here, so the copy is unrolled and needs no loop.
	mlir::Value materialiseLiteral(llvm::StringRef _text)
	{
		uint64_t const length = _text.size();
		uint64_t const padded = (length + 31) / 32 * 32;
		mlir::Value pointer = allocate(wordConstant(32 + padded));
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, wordConstant(length));

		for (uint64_t offset = 0; offset < padded; offset += 32)
		{
			llvm::APInt word(256, 0);
			for (uint64_t i = 0; i < 32; ++i)
			{
				word <<= 8;
				if (offset + i < length)
					word |= llvm::APInt(256, static_cast<uint8_t>(_text[offset + i]));
			}
			mlir::Value at = m_builder.create<mlir::yul::AddOp>(
				loc(), pointer, wordConstant(32 + offset));
			m_builder.create<mlir::yul::MStoreOp>(loc(), at, wordConstant(word));
		}
		return pointer;
	}

	/// Solidity stores `bytes` and `string` two ways in one slot. Up to 31 bytes
	/// the data sits in the slot itself, left-aligned, with `2 * length` in the
	/// lowest byte; from 32 bytes the slot holds `2 * length + 1` and the data
	/// starts at keccak256(slot). The low bit tells the two apart.
	///
	/// These are emitted once per module as ordinary Yul functions rather than
	/// inline, because both ends of every dynamic access need them.
	static constexpr char const* kLoadBytesHelper = "$loadStorageBytes";
	static constexpr char const* kStoreBytesHelper = "$storeStorageBytes";

	/// slot -> memory pointer to [length][data...]
	void emitLoadBytesHelper(mlir::ModuleOp _dst)
	{
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		// A definition, not a statement: it goes at the front so it cannot land
		// after the return that ends the object's code.
		m_builder.setInsertionPointToStart(_dst.getBody());

		auto func = m_builder.create<mlir::yul::FuncOp>(
			loc(), kLoadBytesHelper, m_builder.getFunctionType({wordType()}, {wordType()}));
		mlir::Block& body = func.getBody().emplaceBlock();
		body.addArgument(wordType(), loc());
		m_builder.setInsertionPointToEnd(&body);

		mlir::Value slot = body.getArgument(0);
		mlir::Value packed = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
		mlir::Value isLong = m_builder.create<mlir::yul::AndOp>(loc(), packed, wordConstant(uint64_t(1)));

		// length = isLong ? (packed - 1) / 2 : (packed & 0xff) / 2, branch-free.
		mlir::Value shortLength = m_builder.create<mlir::yul::AndOp>(loc(), packed, wordConstant(uint64_t(0xff)));
		mlir::Value chosen = m_builder.create<mlir::yul::MulOp>(loc(), isLong, packed);
		mlir::Value notLong = m_builder.create<mlir::yul::IsZeroOp>(loc(), isLong);
		mlir::Value shortPart = m_builder.create<mlir::yul::MulOp>(loc(), notLong, shortLength);
		mlir::Value combined = m_builder.create<mlir::yul::AddOp>(loc(), chosen, shortPart);
		mlir::Value length = m_builder.create<mlir::yul::ShrOp>(loc(), wordConstant(uint64_t(1)), combined);

		mlir::Value pointer = allocate(
			m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(length)));
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
		mlir::Value data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));

		// Short: the slot word is the data, but it also carries the length in its
		// lowest byte, so only the top `length` bytes of it belong to the value.
		auto shortCase = m_builder.create<mlir::yul::IfOp>(loc(), notLong);
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&shortCase.getThenRegion().emplaceBlock());
			mlir::Value bits = m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(8)));
			mlir::Value keep = m_builder.create<mlir::yul::SubOp>(loc(), wordConstant(uint64_t(256)), bits);
			mlir::Value masked = m_builder.create<mlir::yul::ShlOp>(
				loc(), keep, m_builder.create<mlir::yul::ShrOp>(loc(), keep, packed));
			m_builder.create<mlir::yul::MStoreOp>(loc(), data, masked);
		}

		// Long: whole words from keccak256(slot) onwards.
		auto longCase = m_builder.create<mlir::yul::IfOp>(loc(), isLong);
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&longCase.getThenRegion().emplaceBlock());
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), slot);
			mlir::Value base = m_builder.create<mlir::yul::Keccak256Op>(
				loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(32)));
			emitWordLoop(length, [&](mlir::Value _index) {
				mlir::Value from = m_builder.create<mlir::yul::AddOp>(
					loc(), base, m_builder.create<mlir::yul::ShrOp>(loc(), wordConstant(uint64_t(5)), _index));
				mlir::Value word = m_builder.create<mlir::yul::SLoadOp>(loc(), from);
				mlir::Value to = m_builder.create<mlir::yul::AddOp>(loc(), data, _index);
				m_builder.create<mlir::yul::MStoreOp>(loc(), to, word);
			});
		}

		m_builder.create<mlir::yul::LeaveOp>(loc(), mlir::ValueRange{pointer});
	}

	/// slot, memory pointer to [length][data...] -> stored
	void emitStoreBytesHelper(mlir::ModuleOp _dst)
	{
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(_dst.getBody());

		auto func = m_builder.create<mlir::yul::FuncOp>(
			loc(), kStoreBytesHelper, m_builder.getFunctionType({wordType(), wordType()}, {}));
		mlir::Block& body = func.getBody().emplaceBlock();
		body.addArgument(wordType(), loc());
		body.addArgument(wordType(), loc());
		m_builder.setInsertionPointToEnd(&body);

		mlir::Value slot = body.getArgument(0);
		mlir::Value pointer = body.getArgument(1);
		mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
		mlir::Value data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		mlir::Value isLong = m_builder.create<mlir::yul::LtOp>(loc(), wordConstant(uint64_t(31)), length);

		auto shortCase = m_builder.create<mlir::yul::IfOp>(
			loc(), m_builder.create<mlir::yul::IsZeroOp>(loc(), isLong));
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&shortCase.getThenRegion().emplaceBlock());
			// Keep only the bytes that belong to the value, then put 2*length
			// in the byte the data cannot reach.
			mlir::Value word = m_builder.create<mlir::yul::MLoadOp>(loc(), data);
			mlir::Value bits = m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(8)));
			mlir::Value keep = m_builder.create<mlir::yul::SubOp>(loc(), wordConstant(uint64_t(256)), bits);
			mlir::Value masked = m_builder.create<mlir::yul::ShlOp>(
				loc(), keep, m_builder.create<mlir::yul::ShrOp>(loc(), keep, word));
			mlir::Value tag = m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(2)));
			m_builder.create<mlir::yul::SStoreOp>(
				loc(), slot, m_builder.create<mlir::yul::OrOp>(loc(), masked, tag));
		}

		auto longCase = m_builder.create<mlir::yul::IfOp>(loc(), isLong);
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&longCase.getThenRegion().emplaceBlock());
			mlir::Value tag = m_builder.create<mlir::yul::AddOp>(
				loc(),
				m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(2))),
				wordConstant(uint64_t(1)));
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, tag);
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), slot);
			mlir::Value base = m_builder.create<mlir::yul::Keccak256Op>(
				loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(32)));
			emitWordLoop(length, [&](mlir::Value _index) {
				mlir::Value word = m_builder.create<mlir::yul::MLoadOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(loc(), data, _index));
				mlir::Value to = m_builder.create<mlir::yul::AddOp>(
					loc(), base, m_builder.create<mlir::yul::ShrOp>(loc(), wordConstant(uint64_t(5)), _index));
				m_builder.create<mlir::yul::SStoreOp>(loc(), to, word);
			});
		}

		m_builder.create<mlir::yul::LeaveOp>(loc(), mlir::ValueRange{});
	}

	/// `for (i = 0; i < roundUp32(length); i += 32) _body(i)`.
	template<typename Body>
	void emitWordLoop(mlir::Value _length, Body _body)
	{
		mlir::Value limit = roundedUp(_length);
		auto var = m_builder.create<mlir::yul::VarOp>(loc(), wordConstant(uint64_t(0)));

		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());
			mlir::Value index = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), var);
			m_builder.create<mlir::yul::ConditionOp>(
				loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, limit));
		}
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());
			_body(m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), var));
		}
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getPostRegion().emplaceBlock());
			mlir::Value index = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), var);
			m_builder.create<mlir::yul::AssignOp>(
				loc(), var, m_builder.create<mlir::yul::AddOp>(loc(), index, wordConstant(uint64_t(32))));
		}
	}

	void emitRevertIf(mlir::Value _condition)
	{
		auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), _condition);
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
		mlir::Value zero = wordConstant(uint64_t(0));
		m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
	}

	/// Slot of `mapping[key]`, which Solidity defines as
	/// keccak256(key . slot) over the two words written to scratch memory at
	/// 0x00 and 0x20 - the region the language reserves for exactly this. A
	/// nested mapping applies the rule once per key, the result of one round
	/// becoming the base slot of the next.
	mlir::Value mappingSlot(uint64_t _baseSlot, mlir::ValueRange _keys)
	{
		mlir::Value slot = wordConstant(_baseSlot);
		for (mlir::Value key: _keys)
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), mapped(key));
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(32)), slot);
			slot = m_builder.create<mlir::yul::Keccak256Op>(
				loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(64)));
		}
		return slot;
	}

	/// The ABI name of a type, or nothing when it is one this dispatcher cannot
	/// describe. Only elementary types are handled: anything living in memory
	/// needs encoding that does not exist yet, and guessing at its name would
	/// produce a selector that silently answers the wrong call.
	static std::optional<std::string> abiTypeName(mlir::Type _type)
	{
		if (auto uintType = llvm::dyn_cast<mlir::solidity::UIntType>(_type))
			return "uint" + std::to_string(uintType.getBitWidth());
		if (auto intType = llvm::dyn_cast<mlir::solidity::IntType>(_type))
			return "int" + std::to_string(intType.getBitWidth());
		if (llvm::isa<mlir::solidity::AddressType>(_type))
			return std::string("address");
		if (llvm::isa<mlir::solidity::BoolType>(_type))
			return std::string("bool");
		if (auto bytesType = llvm::dyn_cast<mlir::solidity::BytesType>(_type))
			return "bytes" + std::to_string(bytesType.getSize());
		if (llvm::isa<mlir::solidity::StringType>(_type))
			return std::string("string");
		if (llvm::isa<mlir::solidity::DynamicBytesType>(_type))
			return std::string("bytes");
		return std::nullopt;
	}

	/// The canonical ABI signature, when every type in it can be named.
	static std::optional<std::string> abiSignature(mlir::solidity::FunctionOp _func)
	{
		std::string signature = _func.getSymName().str() + "(";
		bool first = true;
		for (mlir::Type type: _func.getArgumentTypes())
		{
			std::optional<std::string> const name = abiTypeName(type);
			if (!name)
				return std::nullopt;
			signature += (first ? "" : ",") + *name;
			first = false;
		}
		return signature + ")";
	}

	/// Emits the external entry point at module scope, which YulToEVM turns
	/// into the object's `@__entry`: read the selector, and for each public
	/// function whose signature can be spelled, decode its arguments straight
	/// out of calldata, call it and return the result. Anything unmatched
	/// reverts, as a contract without a fallback does.
	///
	/// Only single-word arguments and results are dispatched, because that is
	/// exactly the set needing no memory encoding. Functions outside it are
	/// left undispatched rather than dispatched wrongly - unreachable, but not
	/// answering to a selector that means something else.
	void emitDispatcher(mlir::solidity::ContractOp _contract, mlir::ModuleOp _dst)
	{
		std::vector<std::pair<uint32_t, mlir::solidity::FunctionOp>> entries;
		for (mlir::Operation& op: _contract.getBody().front().getOperations())
		{
			auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&op);
			if (!func)
				continue;
			if (func.getSymName().contains('.'))
				continue; // a shadowed base implementation, reachable only by explicit call
			// Constructors, receive and fallback are not reached by a selector.
			// The constructor in particular has no name, so building an entry
			// for it yields a call to nothing.
			if (func->getAttrOfType<mlir::StringAttr>("kind"))
				continue;
			if (func.getVisibility() && *func.getVisibility() != "public" && *func.getVisibility() != "external")
				continue;
			bool describable = true;
			for (mlir::Type type: func.getResultTypes())
				describable = describable && abiTypeName(type).has_value();
			if (!describable)
				continue;
			if (std::optional<std::string> const signature = abiSignature(func))
				entries.emplace_back(solidity::util::selectorFromSignatureU32(*signature), func);
		}
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToEnd(_dst.getBody());

		// A contract nothing can be called on still has to reject the call.
		// Returning without emitting anything left an object that accepts
		// every call and answers nothing, where Solidity reverts.
		if (entries.empty())
		{
			mlir::Value zero = wordConstant(uint64_t(0));
			m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
			return;
		}

		mlir::Value word = m_builder.create<mlir::yul::CallDataLoadOp>(loc(), wordConstant(uint64_t(0)));
		mlir::Value selector = m_builder.create<mlir::yul::ShrOp>(loc(), wordConstant(uint64_t(224)), word);

		for (auto& [value, func]: entries)
		{
			mlir::Value matches = m_builder.create<mlir::yul::EqOp>(loc(), selector, wordConstant(value));
			auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), matches);
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());

			// Calldata short of the declared arguments has to revert. Reading
			// past the end yields zeros rather than faulting, so without this
			// the call quietly succeeds on arguments nobody supplied - which is
			// what the differential against solc caught.
			size_t const argumentCount = func.getArgumentTypes().size();
			if (argumentCount > 0)
			{
				mlir::Value size = m_builder.create<mlir::yul::CallDataSizeOp>(loc());
				mlir::Value tooShort = m_builder.create<mlir::yul::LtOp>(
					loc(), size, wordConstant(uint64_t(4 + 32 * argumentCount)));
				emitRevertIf(tooShort);
			}

			llvm::SmallVector<mlir::Value, 4> arguments;
			for (unsigned i = 0; i < argumentCount; ++i)
				arguments.push_back(
					m_builder.create<mlir::yul::CallDataLoadOp>(loc(), wordConstant(uint64_t(4 + 32 * i))));

			llvm::SmallVector<mlir::Type, 1> resultTypes(func.getResultTypes().size(), wordType());
			auto called = m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				resultTypes,
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), qualified(func.getSymName())),
				arguments);

			// One word per result, in order. Only a single result was encoded
			// before and everything else returned empty data, so a function
			// returning a tuple answered nothing at all.
			mlir::Value zero = wordConstant(uint64_t(0));
			unsigned const resultCount = called->getNumResults();
			mlir::ArrayRef<mlir::Type> const declared = func.getResultTypes();
			bool const dynamic = resultCount == 1 && declared.size() == 1 && isDynamic(declared[0]);

			if (dynamic)
			{
				// A dynamic result is a memory pointer to [length][data...].
				// The ABI wants an offset to that pair, so the encoding is one
				// word of offset followed by the pair itself, padded to a whole
				// number of words.
				m_usesMemory = true;
				mlir::Value pointer = called->getResult(0);
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
				mlir::Value payload = m_builder.create<mlir::yul::AddOp>(
					loc(), wordConstant(uint64_t(32)), roundedUp(length));
				mlir::Value total = m_builder.create<mlir::yul::AddOp>(
					loc(), wordConstant(uint64_t(32)), payload);

				mlir::Value head = allocate(total);
				m_builder.create<mlir::yul::MStoreOp>(loc(), head, wordConstant(uint64_t(32)));
				m_builder.create<mlir::yul::MCopyOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), head, wordConstant(uint64_t(32))),
					pointer,
					payload);
				m_builder.create<mlir::yul::ReturnOp>(loc(), head, total);
			}
			else
			{
				for (unsigned i = 0; i < resultCount; ++i)
					m_builder.create<mlir::yul::MStoreOp>(
						loc(), wordConstant(uint64_t(32 * i)), called->getResult(i));
				m_builder.create<mlir::yul::ReturnOp>(loc(), zero, wordConstant(uint64_t(32 * resultCount)));
			}
		}

		mlir::Value zero = wordConstant(uint64_t(0));
		m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
	}

	uint64_t storageSlot(llvm::StringRef _name)
	{
		auto it = m_storageSlots.find(_name);
		if (it == m_storageSlots.end())
			fail("reference to unknown state variable '" + _name.str() + "'");
		return it->second;
	}

	//===------------------------------------------------------------------===//
	// Values
	//===------------------------------------------------------------------===//

	mlir::Value convertConstant(mlir::solidity::ConstantOp _op)
	{
		mlir::Attribute value = _op.getValue();
		// BoolAttr IS-A IntegerAttr: check it first to avoid sign-extending
		// a 1-bit true into -1.
		if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(value))
			return wordConstant(uint64_t(boolAttr.getValue() ? 1 : 0));
		if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value))
		{
			llvm::APInt raw = intAttr.getValue();
			bool isSigned = llvm::isa<mlir::solidity::IntType>(_op.getResult().getType());
			llvm::APInt wide = isSigned ? raw.sextOrTrunc(256) : raw.zextOrTrunc(256);
			return wordConstant(std::move(wide));
		}
		fail("unsupported sol.constant attribute kind");
	}

	bool convertBinaryArith(mlir::Operation& _op)
	{
		// sol shift ops carry ($value, $shift); yul builtins take
		// (shift, value) - swap during conversion.
		mlir::Value result;
		if (auto shl = llvm::dyn_cast<mlir::solidity::ShlOp>(&_op))
			result = m_builder.create<mlir::yul::ShlOp>(loc(), mapped(shl.getShift()), mapped(shl.getValue()));
		else if (auto shr = llvm::dyn_cast<mlir::solidity::ShrOp>(&_op))
			result = m_builder.create<mlir::yul::ShrOp>(loc(), mapped(shr.getShift()), mapped(shr.getValue()));
		else if (auto sar = llvm::dyn_cast<mlir::solidity::SarOp>(&_op))
			result = m_builder.create<mlir::yul::SarOp>(loc(), mapped(sar.getShift()), mapped(sar.getValue()));
		else if (auto add = llvm::dyn_cast<mlir::solidity::AddOp>(&_op))
			result = m_builder.create<mlir::yul::AddOp>(loc(), mapped(add.getLhs()), mapped(add.getRhs()));
		else if (auto sub = llvm::dyn_cast<mlir::solidity::SubOp>(&_op))
			result = m_builder.create<mlir::yul::SubOp>(loc(), mapped(sub.getLhs()), mapped(sub.getRhs()));
		else if (auto mul = llvm::dyn_cast<mlir::solidity::MulOp>(&_op))
			result = m_builder.create<mlir::yul::MulOp>(loc(), mapped(mul.getLhs()), mapped(mul.getRhs()));
		else if (auto div = llvm::dyn_cast<mlir::solidity::DivOp>(&_op))
		{
			bool isSigned = llvm::isa<mlir::solidity::IntType>(div.getLhs().getType());
			if (isSigned)
				result = m_builder.create<mlir::yul::SDivOp>(loc(), mapped(div.getLhs()), mapped(div.getRhs()));
			else
				result = m_builder.create<mlir::yul::DivOp>(loc(), mapped(div.getLhs()), mapped(div.getRhs()));
		}
		else if (auto mod = llvm::dyn_cast<mlir::solidity::ModOp>(&_op))
		{
			bool isSigned = llvm::isa<mlir::solidity::IntType>(mod.getLhs().getType());
			if (isSigned)
				result = m_builder.create<mlir::yul::SModOp>(loc(), mapped(mod.getLhs()), mapped(mod.getRhs()));
			else
				result = m_builder.create<mlir::yul::ModOp>(loc(), mapped(mod.getLhs()), mapped(mod.getRhs()));
		}
		else if (auto exp = llvm::dyn_cast<mlir::solidity::ExpOp>(&_op))
			result = m_builder.create<mlir::yul::ExpOp>(loc(), mapped(exp.getBase()), mapped(exp.getExponent()));
		else if (auto andOp = llvm::dyn_cast<mlir::solidity::AndOp>(&_op))
			result = m_builder.create<mlir::yul::AndOp>(loc(), mapped(andOp.getLhs()), mapped(andOp.getRhs()));
		else if (auto orOp = llvm::dyn_cast<mlir::solidity::OrOp>(&_op))
			result = m_builder.create<mlir::yul::OrOp>(loc(), mapped(orOp.getLhs()), mapped(orOp.getRhs()));
		else if (auto xorOp = llvm::dyn_cast<mlir::solidity::XorOp>(&_op))
			result = m_builder.create<mlir::yul::XorOp>(loc(), mapped(xorOp.getLhs()), mapped(xorOp.getRhs()));
		else
			return false;
		m_map[_op.getResult(0)] = result;
		return true;
	}

	mlir::Value convertCmp(mlir::solidity::CmpOp _cmp)
	{
		mlir::Value lhs = mapped(_cmp.getLhs());
		mlir::Value rhs = mapped(_cmp.getRhs());
		llvm::StringRef predicate = _cmp.getPredicate();

		auto direct = [&](auto _make) -> mlir::Value { return _make(); };
		auto negated = [&](mlir::Value _v) -> mlir::Value {
			return m_builder.create<mlir::yul::IsZeroOp>(loc(), _v);
		};

		if (predicate == "eq")
			return direct([&] { return m_builder.create<mlir::yul::EqOp>(loc(), lhs, rhs); });
		if (predicate == "ne")
			return negated(m_builder.create<mlir::yul::EqOp>(loc(), lhs, rhs));
		if (predicate == "lt")
			return direct([&] { return m_builder.create<mlir::yul::LtOp>(loc(), lhs, rhs); });
		if (predicate == "gt")
			return direct([&] { return m_builder.create<mlir::yul::GtOp>(loc(), lhs, rhs); });
		if (predicate == "le")
			return negated(m_builder.create<mlir::yul::GtOp>(loc(), lhs, rhs));
		if (predicate == "ge")
			return negated(m_builder.create<mlir::yul::LtOp>(loc(), lhs, rhs));
		if (predicate == "slt")
			return direct([&] { return m_builder.create<mlir::yul::SLtOp>(loc(), lhs, rhs); });
		if (predicate == "sgt")
			return direct([&] { return m_builder.create<mlir::yul::SGtOp>(loc(), lhs, rhs); });
		if (predicate == "sle")
			return negated(m_builder.create<mlir::yul::SGtOp>(loc(), lhs, rhs));
		if (predicate == "sge")
			return negated(m_builder.create<mlir::yul::SLtOp>(loc(), lhs, rhs));
		fail("unsupported sol.cmp predicate: " + predicate.str());
	}

	/// EVM-representation conversion into the given sol type.
	mlir::Value representationConvert(mlir::Value _word, mlir::Type _solType)
	{
		if (auto uintType = llvm::dyn_cast<mlir::solidity::UIntType>(_solType))
		{
			unsigned width = uintType.getBitWidth();
			if (width >= 256)
				return _word;
			mlir::Value mask = wordConstant(llvm::APInt::getLowBitsSet(256, width));
			return m_builder.create<mlir::yul::AndOp>(loc(), _word, mask);
		}
		if (auto intType = llvm::dyn_cast<mlir::solidity::IntType>(_solType))
		{
			unsigned width = intType.getBitWidth();
			if (width >= 256)
				return _word;
			// signextend(b, x) extends from byte index b.
			mlir::Value byteIndex = wordConstant(uint64_t(width / 8 - 1));
			return m_builder.create<mlir::yul::SignExtendOp>(loc(), byteIndex, _word);
		}
		if (llvm::isa<mlir::solidity::AddressType>(_solType))
		{
			mlir::Value mask = wordConstant(llvm::APInt::getLowBitsSet(256, 160));
			return m_builder.create<mlir::yul::AndOp>(loc(), _word, mask);
		}
		if (llvm::isa<mlir::solidity::BoolType>(_solType))
		{
			mlir::Value notIt = m_builder.create<mlir::yul::IsZeroOp>(loc(), _word);
			return m_builder.create<mlir::yul::IsZeroOp>(loc(), notIt);
		}
		return _word; // bytesN & friends: representation is the word itself
	}

	//===------------------------------------------------------------------===//
	// Control flow
	//===------------------------------------------------------------------===//

	void convertIf(mlir::solidity::IfOp _if)
	{
		// Yul has no else: lower to two exclusive ifs over the pre-computed
		// condition.
		mlir::Value cond = mapped(_if.getCondition());

		{
			auto thenIf = m_builder.create<mlir::yul::IfOp>(loc(), cond);
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&thenIf.getThenRegion().emplaceBlock());
			if (!_if.getThenRegion().empty())
				convertBlockOps(_if.getThenRegion().front());
		}
		if (!_if.getElseRegion().empty())
		{
			mlir::Value notCond = m_builder.create<mlir::yul::IsZeroOp>(loc(), cond);
			auto elseIf = m_builder.create<mlir::yul::IfOp>(loc(), notCond);
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&elseIf.getThenRegion().emplaceBlock());
			convertBlockOps(_if.getElseRegion().front());
		}
	}

	/// `scf.while` is what the generator emits for every `for` and `while`, and
	/// nothing lowered it - so no contract with a loop got past this rung.
	///
	/// The two dialects disagree about where a loop's state lives. `scf.while`
	/// carries it as region arguments and yielded results, which is SSA and has
	/// no place in Yul; `yul.for` has mutable variables instead. So each
	/// carried value becomes a `yul.var`, the region arguments read it, and the
	/// terminators assign it.
	void convertWhile(mlir::scf::WhileOp _while)
	{
		llvm::SmallVector<mlir::Value, 4> slots;
		for (mlir::Value initial: _while.getInits())
			slots.push_back(m_builder.create<mlir::yul::VarOp>(loc(), mapped(initial)));

		auto readInto = [&](mlir::Block& _block) {
			for (auto [index, argument]: llvm::enumerate(_block.getArguments()))
				if (index < slots.size())
					m_map[argument] = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), slots[index]);
		};
		auto assignAll = [&](mlir::ValueRange _values) {
			for (auto [index, value]: llvm::enumerate(_values))
				if (index < slots.size())
					m_builder.create<mlir::yul::AssignOp>(loc(), slots[index], mapped(value));
		};

		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());

			mlir::Block& before = _while.getBefore().front();
			readInto(before);
			mlir::Value cond;
			for (mlir::Operation& op: before.getOperations())
			{
				if (auto condition = llvm::dyn_cast<mlir::scf::ConditionOp>(&op))
				{
					// What the condition forwards is what the body sees and
					// what the loop yields on exit, so it lands in the slots
					// either way.
					assignAll(condition.getArgs());
					cond = mapped(condition.getCondition());
					break;
				}
				convertOp(op);
			}
			m_builder.create<mlir::yul::ConditionOp>(loc(), cond ? cond : wordConstant(uint64_t(0)));
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());

			mlir::Block& after = _while.getAfter().front();
			readInto(after);
			for (mlir::Operation& op: after.getOperations())
			{
				if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(&op))
				{
					assignAll(yield.getResults());
					break;
				}
				convertOp(op);
			}
		}
		forOp.getPostRegion().emplaceBlock();

		// After the loop the carried values are whatever the condition last
		// forwarded, which is exactly what the slots hold.
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointAfter(forOp);
		for (auto [index, result]: llvm::enumerate(_while.getResults()))
			if (index < slots.size())
				m_map[result] = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), slots[index]);
	}

	void convertFor(mlir::solidity::ForOp _for)
	{
		// Init region is hoisted before the loop (ADR-004).
		if (!_for.getInit().empty())
			if (convertBlockOps(_for.getInit().front()))
				fail("terminator in for-loop init region");

		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());
			mlir::Value cond;
			if (!_for.getCondition().empty())
			{
				for (mlir::Operation& op: _for.getCondition().front().getOperations())
				{
					if (convertOp(op))
						fail("terminator in for-loop condition region");
					if (op.getNumResults() == 1)
						cond = m_map.lookup(op.getResult(0));
				}
			}
			if (!cond)
				cond = wordConstant(uint64_t(1));
			m_builder.create<mlir::yul::ConditionOp>(loc(), cond);
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());
			if (!_for.getBody().empty())
				convertBlockOps(_for.getBody().front());
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getPostRegion().emplaceBlock());
			if (!_for.getUpdate().empty())
				convertBlockOps(_for.getUpdate().front());
		}
	}
};

} // anonymous namespace

namespace solidity::mlirgen
{

mlir::OwningOpRef<mlir::ModuleOp> convertSolToYul(mlir::ModuleOp _module, std::string& _error, bool _creation)
{
	mlir::MLIRContext& ctx = *_module.getContext();
	ctx.getOrLoadDialect<mlir::yul::YulDialect>();
	referencedContracts().clear();
	return SolToYulConverter(ctx, _creation).run(_module, _error);
}

std::set<std::string> const& lastReferencedContracts()
{
	return referencedContracts();
}

} // namespace solidity::mlirgen
