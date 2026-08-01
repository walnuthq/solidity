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

class SolToYulConverter
{
public:
	explicit SolToYulConverter(mlir::MLIRContext& _ctx): m_builder(&_ctx) {}

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
							m_declaredFunctions.insert(contract.getName().str() + "." + func.getSymName().str());

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
	llvm::StringSet<> m_declaredFunctions;
	std::string m_contractName;

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

		emitDispatcher(_contract, _dst);
	}

	void convertFunction(mlir::solidity::FunctionOp _func)
	{
		mlir::FunctionType solType = _func.getFunctionType();
		llvm::SmallVector<mlir::Type, 4> paramTypes(solType.getNumInputs(), wordType());
		llvm::SmallVector<mlir::Type, 2> resultTypes(solType.getNumResults(), wordType());
		auto yulType = mlir::FunctionType::get(m_builder.getContext(), paramTypes, resultTypes);

		auto yulFunc = m_builder.create<mlir::yul::FuncOp>(
			loc(), m_contractName.empty() ? _func.getSymName().str() : qualified(_func.getSymName()), yulType);
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
		if (auto load = llvm::dyn_cast<mlir::solidity::LoadStateVarOp>(&_op))
		{
			mlir::Value slot = wordConstant(storageSlot(load.getVarName()));
			m_map[load.getResult()] = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::StoreStateVarOp>(&_op))
		{
			mlir::Value slot = wordConstant(storageSlot(store.getVarName()));
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
			if (func.getVisibility() && *func.getVisibility() != "public" && *func.getVisibility() != "external")
				continue;
			if (func.getResultTypes().size() > 1)
				continue; // multiple results need tuple encoding
			bool describable = true;
			for (mlir::Type type: func.getResultTypes())
				describable = describable && abiTypeName(type).has_value();
			if (!describable)
				continue;
			if (std::optional<std::string> const signature = abiSignature(func))
				entries.emplace_back(solidity::util::selectorFromSignatureU32(*signature), func);
		}
		if (entries.empty())
			return;

		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToEnd(_dst.getBody());

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

			mlir::Value zero = wordConstant(uint64_t(0));
			if (called->getNumResults() == 1)
			{
				m_builder.create<mlir::yul::MStoreOp>(loc(), zero, called->getResult(0));
				m_builder.create<mlir::yul::ReturnOp>(loc(), zero, wordConstant(uint64_t(32)));
			}
			else
				m_builder.create<mlir::yul::ReturnOp>(loc(), zero, zero);
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

mlir::OwningOpRef<mlir::ModuleOp> convertSolToYul(mlir::ModuleOp _module, std::string& _error)
{
	mlir::MLIRContext& ctx = *_module.getContext();
	ctx.getOrLoadDialect<mlir::yul::YulDialect>();
	return SolToYulConverter(ctx).run(_module, _error);
}

} // namespace solidity::mlirgen
