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

#include <libsolutil/Keccak256.h>

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
#include "../../Import/LibyulAST/YulASTImporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

#include <libsolutil/FunctionSelector.h>

#include <optional>
#include <cstdlib>
#include <set>
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

/// `new C(...)` needs C's creation code, which is a different object from the
/// runtime code `type(C).runtimeCode` names. Kept free of dots, because the
/// emitter reads those as a path into nested objects.
constexpr char kCreationObjectSuffix[] = "_new";

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
	SolToYulConverter(
		mlir::MLIRContext& _ctx,
		bool _creation,
		solidity::langutil::EVMVersion _evmVersion
	):
		m_builder(&_ctx), m_creation(_creation), m_evmVersion(_evmVersion)
	{}

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
							if (kindAttr && kindAttr.getValue() == "initializer")
								m_initializers.insert(contract.getName().str() + "." + bare);
						}

			for (mlir::Operation& op: _src.getBody()->getOperations())
			{
				if (auto contract = llvm::dyn_cast<mlir::solidity::ContractOp>(&op))
				{
					// A contract carried along so `new C(...)` can name it is
					// compiled as its own object, not spliced into this one.
					// Converting it here appended its functions after this
					// object's dispatcher, which ends in a revert - so the
					// module had operations past a terminator.
					if (contract->hasAttr("is_subobject"))
						continue;
					convertContract(contract, *dst);
				}
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
			if (std::getenv("SOL_MLIR_DUMP_INVALID"))
				dst->print(llvm::errs());
			_error = "converted module failed MLIR verification";
			return nullptr;
		}
		return dst;
	}

private:
	mlir::OpBuilder m_builder;
	llvm::DenseMap<mlir::Value, mlir::Value> m_map;
	llvm::StringMap<uint64_t> m_storageSlots;
	llvm::StringMap<unsigned> m_storageOffsets;
	llvm::StringMap<unsigned> m_storageBytes;
	llvm::StringSet<> m_internalFunctionStateVars;
	bool m_usesMemory = false;
	bool m_needsBytesHelpers = false;
	bool m_legacyCodegen = false;
	bool m_debugRevertStrings = false;
	bool m_hasBitwiseShifting = true;
	bool m_supportsReturndata = true;
	bool m_hasStaticCall = true;
	bool m_hasMcopy = true;
	bool m_canOverchargeCallGas = true;
	/// Objects this contract's code names, which have to be nested in it.
	std::set<std::string>& m_referencedObjects = referencedContracts();
	llvm::StringMap<mlir::Value> m_assemblyVars;
	std::vector<llvm::SmallVector<mlir::Value, 4>> m_loopSlotStack;
	unsigned m_inlineAssemblySequence = 0;

	/// `string` and `bytes` do not fit a word, so they live in memory and are
	/// named by a pointer to [length][data...].
	static bool isDynamic(mlir::Type _type)
	{
		return llvm::isa<mlir::solidity::StringType, mlir::solidity::DynamicBytesType>(_type);
	}
	llvm::StringSet<> m_declaredFunctions;
	llvm::StringSet<> m_initializers;
	llvm::StringMap<unsigned> m_constructorArity;
	llvm::StringMap<llvm::SmallVector<mlir::Type, 4>> m_constructorTypes;
	std::set<unsigned> m_constructorExternalParams;
	std::string m_contractName;
	bool m_creation = false;
	solidity::langutil::EVMVersion m_evmVersion;

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

	static bool indexedAttr(mlir::Operation* _op, llvm::StringRef _name, unsigned _index)
	{
		if (auto indices = _op->getAttrOfType<mlir::ArrayAttr>(_name))
			for (mlir::Attribute attribute: indices)
				if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attribute);
					integer && static_cast<unsigned>(integer.getInt()) == _index)
					return true;
		return false;
	}

	static uint64_t indexedIntegerAttr(mlir::Operation* _op, llvm::StringRef _name, unsigned _index)
	{
		if (auto values = _op->getAttrOfType<mlir::ArrayAttr>(_name); values && _index < values.size())
			if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(values[_index]))
				return static_cast<uint64_t>(integer.getInt());
		return 0;
	}

	mlir::Value wordConstant(llvm::APInt _value)
	{
		return m_builder.create<mlir::yul::ConstOp>(loc(), mlir::IntegerAttr::get(wordType(), std::move(_value)));
	}

	mlir::Value wordConstant(uint64_t _value) { return wordConstant(llvm::APInt(256, _value)); }

	/// Constantinople introduced SHL/SHR/SAR. The direct Solidity route also
	/// targets older revisions, so every shift produced by this lowering goes
	/// through the arithmetic identities solc used before those opcodes existed.
	mlir::Value yulShl(mlir::Location _loc, mlir::Value _shift, mlir::Value _value)
	{
		if (m_hasBitwiseShifting)
			return m_builder.create<mlir::yul::ShlOp>(_loc, _shift, _value);
		return m_builder.create<mlir::yul::MulOp>(
			_loc, _value,
			m_builder.create<mlir::yul::ExpOp>(_loc, wordConstant(uint64_t(2)), _shift));
	}

	mlir::Value yulShr(mlir::Location _loc, mlir::Value _shift, mlir::Value _value)
	{
		if (m_hasBitwiseShifting)
			return m_builder.create<mlir::yul::ShrOp>(_loc, _shift, _value);
		return m_builder.create<mlir::yul::DivOp>(
			_loc, _value,
			m_builder.create<mlir::yul::ExpOp>(_loc, wordConstant(uint64_t(2)), _shift));
	}

	mlir::Value yulSar(mlir::Location _loc, mlir::Value _shift, mlir::Value _value)
	{
		if (m_hasBitwiseShifting)
			return m_builder.create<mlir::yul::SarOp>(_loc, _shift, _value);
		mlir::Value divisor = m_builder.create<mlir::yul::ExpOp>(
			_loc, wordConstant(uint64_t(2)), _shift);
		mlir::Value positive = m_builder.create<mlir::yul::DivOp>(_loc, _value, divisor);
		mlir::Value negative = m_builder.create<mlir::yul::NotOp>(
			_loc, m_builder.create<mlir::yul::DivOp>(
				_loc, m_builder.create<mlir::yul::NotOp>(_loc, _value), divisor));
		mlir::Value signMask = m_builder.create<mlir::yul::SubOp>(
			_loc, wordConstant(uint64_t(0)),
			m_builder.create<mlir::yul::SLtOp>(_loc, _value, wordConstant(uint64_t(0))));
		return m_builder.create<mlir::yul::OrOp>(
			_loc,
			m_builder.create<mlir::yul::AndOp>(_loc, signMask, negative),
			m_builder.create<mlir::yul::AndOp>(
				_loc, m_builder.create<mlir::yul::NotOp>(_loc, signMask), positive));
	}

	/// MCOPY is a Cancun opcode. Every copy synthesized by this conversion has
	/// a separately allocated destination, so older revisions can use an exact
	/// forward copy: whole words first, then the byte tail without touching
	/// adjacent ABI data.
	void yulMCopy(
		mlir::Location _loc,
		mlir::Value _destination,
		mlir::Value _source,
		mlir::Value _length)
	{
		if (m_hasMcopy)
		{
			m_builder.create<mlir::yul::MCopyOp>(_loc, _destination, _source, _length);
			return;
		}

		mlir::Value words = m_builder.create<mlir::yul::DivOp>(
			_loc, _length, wordConstant(uint64_t(32)));
		emitIndexLoop(words, [&](mlir::Value index) {
			mlir::Value offset = m_builder.create<mlir::yul::MulOp>(
				_loc, index, wordConstant(uint64_t(32)));
			m_builder.create<mlir::yul::MStoreOp>(
				_loc,
				m_builder.create<mlir::yul::AddOp>(_loc, _destination, offset),
				m_builder.create<mlir::yul::MLoadOp>(
					_loc, m_builder.create<mlir::yul::AddOp>(_loc, _source, offset)));
		});

		mlir::Value copied = m_builder.create<mlir::yul::MulOp>(
			_loc, words, wordConstant(uint64_t(32)));
		mlir::Value tail = m_builder.create<mlir::yul::ModOp>(
			_loc, _length, wordConstant(uint64_t(32)));
		emitIndexLoop(tail, [&](mlir::Value index) {
			mlir::Value offset = m_builder.create<mlir::yul::AddOp>(_loc, copied, index);
			mlir::Value source = m_builder.create<mlir::yul::AddOp>(_loc, _source, offset);
			m_builder.create<mlir::yul::MStore8Op>(
				_loc,
				m_builder.create<mlir::yul::AddOp>(_loc, _destination, offset),
				m_builder.create<mlir::yul::ByteOp>(
					_loc, wordConstant(uint64_t(0)),
					m_builder.create<mlir::yul::MLoadOp>(_loc, source)));
		});
	}

	//===------------------------------------------------------------------===//
	// Structure
	//===------------------------------------------------------------------===//

	/// Yul has one flat namespace, but a file's contracts all land in one
	/// module and nothing stops two of them defining `value`. Qualifying the
	/// symbol by its contract is what keeps them apart; the ABI name is taken
	/// from the unqualified part, so selectors are unaffected.
	std::string qualified(llvm::StringRef _name) const { return m_contractName + "." + _name.str(); }

	std::string yulFunctionName(mlir::solidity::FunctionOp _func) const
	{
		if (auto kind = _func->getAttrOfType<mlir::StringAttr>("kind"))
			if (kind == "constructor" || kind == "receive" || kind == "fallback")
				return qualified(kind.getValue());
		return _func.getSymName().contains('.')
			? _func.getSymName().str()
			: qualified(_func.getSymName());
	}

	void convertContract(mlir::solidity::ContractOp _contract, mlir::ModuleOp _dst)
	{
		m_contractName = _contract.getName().str();
		m_legacyCodegen = _contract->hasAttr("legacy_codegen");
		m_debugRevertStrings = _contract->hasAttr("revert_strings_debug");
		m_hasBitwiseShifting = _contract->hasAttr("has_bitwise_shifting");
		m_supportsReturndata = _contract->hasAttr("supports_returndata");
		m_hasStaticCall = _contract->hasAttr("has_staticcall");
		m_hasMcopy = _contract->hasAttr("has_mcopy");
		m_canOverchargeCallGas = _contract->hasAttr("can_overcharge_call_gas");
		m_internalFunctionStateVars.clear();
		// Storage layout: sequential slots in declaration order (the real
		// pipeline uses ContractType::linearizedStateVariables(); the
		// generator emits state_var ops in that order).
		// The generator works the layout out with Solidity's own rules, which
		// account for what a struct occupies; numbering them one apiece here
		// put the next variable on top of a struct's second member.
		uint64_t nextSlot = 0;
		for (mlir::Operation& op: _contract.getBody().front().getOperations())
			if (auto stateVar = llvm::dyn_cast<mlir::solidity::StateVarOp>(&op))
			{
				if (stateVar->hasAttr("internalFunction"))
					m_internalFunctionStateVars.insert(stateVar.getName());
				if (auto declared = stateVar->getAttrOfType<mlir::IntegerAttr>("storageSlot"))
					m_storageSlots[stateVar.getName()] = static_cast<uint64_t>(declared.getInt());
				else
					m_storageSlots[stateVar.getName()] = nextSlot;
				m_storageOffsets[stateVar.getName()] = stateVar->getAttrOfType<mlir::IntegerAttr>("storageOffset")
					? static_cast<unsigned>(stateVar->getAttrOfType<mlir::IntegerAttr>("storageOffset").getInt())
					: 0;
				m_storageBytes[stateVar.getName()] = stateVar->getAttrOfType<mlir::IntegerAttr>("storageBytes")
					? static_cast<unsigned>(stateVar->getAttrOfType<mlir::IntegerAttr>("storageBytes").getInt())
					: 32;
				nextSlot = m_storageSlots[stateVar.getName()] + 1;
			}

		// A call can name a function this contract does not contain - one that
		// is inherited but never emitted here, or reached through `super`.
		// Knowing the set up front is what lets the call lowering refuse
		// instead of building a reference that nothing resolves.
		for (mlir::Operation& op: _contract.getBody().front().getOperations())
			if (auto func = llvm::dyn_cast<mlir::solidity::FunctionOp>(&op))
				m_declaredFunctions.insert(yulFunctionName(func));

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
		if (m_initializers.contains(initializerName))
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), initializerName),
				mlir::ValueRange{});

		// A constructor body was converted as a function; call it.
		//
		// Its arguments follow the creation code, which is the only place they
		// can be: there is no calldata during construction. So the code copies
		// itself to find where it ends, and reads them from there.
		std::string const constructorName = qualified("constructor");
		auto arity = m_constructorArity.find(constructorName);
		if (m_declaredFunctions.contains(constructorName))
		{
			llvm::SmallVector<mlir::Value, 4> arguments;
			unsigned const parameters = arity == m_constructorArity.end() ? 0 : arity->second;
			if (parameters > 0)
			{
				m_usesMemory = true;
				mlir::Value codeSize = m_builder.create<mlir::yul::CodeSizeOp>(loc());
				auto constructorTypes = m_constructorTypes.find(constructorName);
				bool const dynamicArguments = constructorTypes != m_constructorTypes.end()
					&& llvm::any_of(constructorTypes->second, [](mlir::Type type) {
						return isABIDynamic(type);
					});
				if (dynamicArguments)
				{
					mlir::Value suffix = allocate(wordConstant(uint64_t(32)));
					mlir::Value suffixOffset = m_builder.create<mlir::yul::SubOp>(
						loc(), codeSize, wordConstant(uint64_t(32)));
					m_builder.create<mlir::yul::CodeCopyOp>(
						loc(), suffix, suffixOffset, wordConstant(uint64_t(32)));
					mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), suffix);
					emitRevertIf(m_builder.create<mlir::yul::GtOp>(loc(), length, suffixOffset));
					mlir::Value pointer = allocate(roundedUp(length));
					m_builder.create<mlir::yul::CodeCopyOp>(
						loc(), pointer,
						m_builder.create<mlir::yul::SubOp>(loc(), suffixOffset, length),
						length);
					auto minimumHead = abiTupleHeadSize(constructorTypes->second);
					if (!minimumHead)
						fail("constructor has an unsupported ABI argument");
					emitRevertIf(m_builder.create<mlir::yul::LtOp>(
						loc(), length, wordConstant(*minimumHead)));
					uint64_t headOffset = 0;
					for (auto [i, type]: llvm::enumerate(constructorTypes->second))
					{
						mlir::Value argument = decodeABIComponent(
							type,
							m_builder.create<mlir::yul::AddOp>(
								loc(), pointer, wordConstant(headOffset)),
							pointer, length);
						if (m_constructorExternalParams.count(i))
							argument = yulShr(
								loc(), wordConstant(uint64_t(64)), argument);
						arguments.push_back(argument);
						headOffset += *abiHeadSize(type);
					}
				}
				else
				{
					auto staticHead = constructorTypes == m_constructorTypes.end()
						? std::optional<uint64_t>{}
						: abiTupleHeadSize(constructorTypes->second);
					uint64_t const staticBytes = staticHead
						? *staticHead : uint64_t(32 * parameters);
					mlir::Value length = wordConstant(staticBytes);
					mlir::Value pointer = allocate(length);
					m_builder.create<mlir::yul::CodeCopyOp>(
						loc(), pointer,
						m_builder.create<mlir::yul::SubOp>(loc(), codeSize, length),
						length);
					if (constructorTypes != m_constructorTypes.end()
						&& staticBytes != uint64_t(32 * parameters))
					{
						uint64_t headOffset = 0;
						for (auto [i, type]: llvm::enumerate(constructorTypes->second))
						{
							mlir::Value argument = decodeABIComponent(
								type,
								m_builder.create<mlir::yul::AddOp>(
									loc(), pointer, wordConstant(headOffset)),
								pointer, length);
							if (m_constructorExternalParams.count(i))
								argument = yulShr(
									loc(), wordConstant(uint64_t(64)), argument);
							arguments.push_back(argument);
							headOffset += *abiHeadSize(type);
						}
					}
					else
						for (unsigned i = 0; i < parameters; ++i)
						{
							mlir::Value argument = m_builder.create<mlir::yul::MLoadOp>(
								loc(),
								m_builder.create<mlir::yul::AddOp>(
									loc(), pointer, wordConstant(uint64_t(32 * i))));
							if (m_constructorExternalParams.count(i))
								argument = yulShr(
									loc(), wordConstant(uint64_t(64)), argument);
							arguments.push_back(argument);
						}
				}
			}

			m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), constructorName),
				arguments);
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
			m_constructorTypes[qualified("constructor")] = llvm::SmallVector<mlir::Type, 4>(
				solType0.getInputs());
			if (auto indices = _func->getAttrOfType<mlir::ArrayAttr>("external_function_params"))
				for (mlir::Attribute attribute: indices)
					if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attribute))
						m_constructorExternalParams.insert(static_cast<unsigned>(integer.getInt()));
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

		auto yulFunc = m_builder.create<mlir::yul::FuncOp>(
			loc(), yulFunctionName(_func), yulType);
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
		if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(&_op))
		{
			auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
			if (!integer)
				fail("unsupported arith.constant attribute kind");
			m_map[constant.getResult()] = wordConstant(integer.getValue().zextOrTrunc(256));
			return false;
		}
		if (auto constant = llvm::dyn_cast<mlir::solidity::ConstantOp>(&_op))
		{
			m_map[constant.getResult()] = convertConstant(constant);
			return false;
		}
		if (auto length = llvm::dyn_cast<mlir::solidity::MemoryLengthOp>(&_op))
		{
			if (_op.hasAttr("storageArray"))
			{
				mlir::Value raw = m_builder.create<mlir::yul::SLoadOp>(loc(), arrayBaseSlot(_op));
				mlir::Value isLong = m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(uint64_t(1)));
				mlir::Value shortLength = m_builder.create<mlir::yul::DivOp>(
					loc(), m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(uint64_t(0xff))),
					wordConstant(uint64_t(2)));
				mlir::Value longLength = m_builder.create<mlir::yul::DivOp>(
					loc(), raw, wordConstant(uint64_t(2)));
				m_map[length.getResult()] = m_builder.create<mlir::yul::XorOp>(
					loc(), shortLength,
					m_builder.create<mlir::yul::MulOp>(
						loc(), m_builder.create<mlir::yul::XorOp>(loc(), shortLength, longLength), isLong));
			}
			else
				m_map[length.getResult()] = m_builder.create<mlir::yul::MLoadOp>(loc(), mapped(length.getValue()));
			return false;
		}
		if (auto create = llvm::dyn_cast<mlir::solidity::MemoryArrayCreateOp>(&_op))
		{
			m_usesMemory = true;
			mlir::Value length = mapped(create.getLength());
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(create.getResult().getType());
				array && !array.isDynamicallySized())
			{
				mlir::Value pointer = allocate(m_builder.create<mlir::yul::MulOp>(
					loc(), length, wordConstant(uint64_t(32))));
				initialiseMemoryArrayElements(array.getElementType(), pointer, length);
				m_map[create.getResult()] = pointer;
				return false;
			}
			// Bytes and string pack one byte per element. Ordinary memory arrays
			// use one word per element (including reference values, which are
			// pointers). EVM memory above the monotonically-growing heap pointer is
			// already zero, which supplies Solidity's required default values.
			mlir::Value dataSize
				= mlir::isa<mlir::solidity::DynamicBytesType, mlir::solidity::StringType>(
					  create.getResult().getType())
				? roundedUp(length)
				: m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(32))).getResult();
			if (mlir::isa<mlir::solidity::DynamicBytesType, mlir::solidity::StringType>(
					create.getResult().getType()))
				emitPanicIf(m_builder.create<mlir::yul::LtOp>(loc(), dataSize, length), 0x41);
			else
			{
				mlir::Value recovered = m_builder.create<mlir::yul::DivOp>(
					loc(), dataSize, wordConstant(uint64_t(32)));
				emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
					loc(), m_builder.create<mlir::yul::EqOp>(loc(), recovered, length)), 0x41);
			}
			mlir::Value allocationSize = m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)), dataSize);
			emitPanicIf(m_builder.create<mlir::yul::LtOp>(loc(), allocationSize, dataSize), 0x41);
			mlir::Value pointer = allocate(allocationSize);
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
			// Advancing 0x40 alone does not expand EVM memory. Touch the final
			// allocated word so `new T[](huge)` pays the required quadratic
			// memory cost (and fails out-of-gas before a later low index access).
			m_builder.create<mlir::yul::MStoreOp>(
				loc(),
				m_builder.create<mlir::yul::SubOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, allocationSize),
					wordConstant(uint64_t(32))),
				wordConstant(uint64_t(0)));
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(create.getResult().getType()))
				initialiseMemoryArrayElements(
					array.getElementType(),
					m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					length);
			m_map[create.getResult()] = pointer;
			return false;
		}
		if (auto materialize = llvm::dyn_cast<mlir::solidity::StorageToMemoryOp>(&_op))
		{
			m_map[materialize.getResult()] = materializeStorageValue(
				materialize.getSource().getType(), mapped(materialize.getSource()));
			return false;
		}
		if (auto access = llvm::dyn_cast<mlir::solidity::FixedBytesAccessOp>(&_op))
		{
			auto bytes = llvm::cast<mlir::solidity::BytesType>(access.getValue().getType());
			emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
				loc(), m_builder.create<mlir::yul::LtOp>(
					loc(), mapped(access.getIndex()), wordConstant(static_cast<uint64_t>(bytes.getSize())))), 0x32);
			mlir::Value byte = m_builder.create<mlir::yul::ByteOp>(
				loc(), mapped(access.getIndex()), mapped(access.getValue()));
			m_map[access.getResult()] = yulShl(
				loc(), wordConstant(uint64_t(248)), byte);
			return false;
		}
		if (auto clear = llvm::dyn_cast<mlir::solidity::MemoryClearOp>(&_op))
		{
			clearMemoryValue(clear.getValue().getType(), mapped(clear.getValue()));
			return false;
		}
		if (auto access = llvm::dyn_cast<mlir::solidity::ArrayAccessOp>(&_op))
		{
			// Byte sequences are packed byte-by-byte after their length word.
			// A storage byte sequence is decoded first from Solidity's short/long
			// storage representation; ordinary arrays still use one word per item.
			if (isDynamic(access.getArray().getType()))
			{
				mlir::Value pointer = mapped(access.getArray());
				if (_op.hasAttr("storageArray"))
				{
					m_needsBytesHelpers = m_usesMemory = true;
					pointer = m_builder.create<mlir::yul::FuncCallOp>(
						loc(), mlir::TypeRange{wordType()},
						mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
						mlir::ValueRange{pointer}).getResult(0);
				}
				mlir::Value index = mapped(access.getIndex());
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
				emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
					loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, length)), 0x32);
				mlir::Value address = m_builder.create<mlir::yul::AddOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					index);
				mlir::Value byte = m_builder.create<mlir::yul::ByteOp>(
					loc(), wordConstant(uint64_t(0)),
					m_builder.create<mlir::yul::MLoadOp>(loc(), address));
				// bytesN values live at the high end of an EVM word.
				if (auto bytes = llvm::dyn_cast<mlir::solidity::BytesType>(access.getResult().getType()))
					byte = yulShl(
						loc(), wordConstant(uint64_t(256 - bytes.getSize() * 8)), byte);
				m_map[access.getResult()] = byte;
				return false;
			}
			// In memory an array is a pointer to [length][data...], the same
			// shape a `string` or `bytes` has, so the element is a word into
			// that. In storage it is a slot.
			if (_op.hasAttr("storageArray"))
			{
				mlir::Value length;
				if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(access.getArray().getType()))
					length = array.isDynamicallySized()
						? m_builder.create<mlir::yul::SLoadOp>(loc(), arrayBaseSlot(_op)).getResult()
						: wordConstant(static_cast<uint64_t>(array.getSize()));
				if (length)
				{
					mlir::Value invalid = m_builder.create<mlir::yul::IsZeroOp>(
						loc(), m_builder.create<mlir::yul::LtOp>(
							loc(), mapped(access.getIndex()), length));
					if (_op.hasAttr("getterBounds"))
						emitRevertIf(invalid);
					else
						emitPanicIf(invalid, 0x32);
				}
					mlir::Value slot = arrayElementSlot(_op, access.getIndex());
					m_map[access.getResult()] = _op.hasAttr("asReference")
						? slot
						: loadStorageArrayElement(
							access.getResult().getType(), slot,
							mapped(access.getIndex()), elementBytes(_op));
			}
			else
			{
				if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(access.getArray().getType()))
				{
					mlir::Value length = array.isDynamicallySized()
						? m_builder.create<mlir::yul::MLoadOp>(loc(), mapped(access.getArray())).getResult()
						: wordConstant(static_cast<uint64_t>(array.getSize()));
					emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
						loc(), m_builder.create<mlir::yul::LtOp>(loc(), mapped(access.getIndex()), length)), 0x32);
				}
				m_map[access.getResult()] = m_builder.create<mlir::yul::MLoadOp>(
					loc(), memoryElement(_op, mapped(access.getArray()), mapped(access.getIndex())));
			}
			return false;
		}
		if (auto slice = llvm::dyn_cast<mlir::solidity::ArraySliceOp>(&_op))
		{
			bool const byteSequence
				= isDynamic(slice.getArray().getType()) && isDynamic(slice.getResult().getType());
			auto sourceArray = llvm::dyn_cast<mlir::solidity::ArrayType>(slice.getArray().getType());
			auto resultArray = llvm::dyn_cast<mlir::solidity::ArrayType>(slice.getResult().getType());
			bool const valueArray = sourceArray && resultArray
				&& sourceArray.isDynamicallySized() && resultArray.isDynamicallySized();
			if (!byteSequence && !valueArray)
				fail("array slice has an unsupported representation");
			m_usesMemory = true;
			mlir::Value source = mapped(slice.getArray());
			mlir::Value sourceLength = m_builder.create<mlir::yul::MLoadOp>(loc(), source);
			mlir::Value start = slice.getHasStart()
				? mapped(slice.getStart()) : wordConstant(uint64_t(0));
			mlir::Value end = slice.getHasEnd()
				? mapped(slice.getEnd()) : sourceLength;
			mlir::Value invalid = m_builder.create<mlir::yul::OrOp>(
				loc(),
				m_builder.create<mlir::yul::GtOp>(loc(), start, end),
				m_builder.create<mlir::yul::GtOp>(loc(), end, sourceLength));
			// Solidity's calldata slice checks revert without Panic(uint256)
			// data. Indexing the materialised result below still uses panic 0x32.
			emitRevertIf(invalid);
			mlir::Value length = m_builder.create<mlir::yul::SubOp>(loc(), end, start);
			mlir::Value copySize = byteSequence
				? length
				: m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(32))).getResult();
			mlir::Value allocatedSize = byteSequence ? roundedUp(copySize) : copySize;
			mlir::Value pointer = allocate(m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)), allocatedSize));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
			mlir::Value sourceOffset = byteSequence
				? start
				: m_builder.create<mlir::yul::MulOp>(loc(), start, wordConstant(uint64_t(32))).getResult();
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::AddOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), source, wordConstant(uint64_t(32))),
					sourceOffset),
				copySize);
			m_map[slice.getResult()] = pointer;
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::ArrayStoreOp>(&_op))
		{
			if (isDynamic(store.getArray().getType()))
			{
				mlir::Value pointer = mapped(store.getArray());
				if (_op.hasAttr("storageArray"))
				{
					m_needsBytesHelpers = m_usesMemory = true;
					pointer = m_builder.create<mlir::yul::FuncCallOp>(
						loc(), mlir::TypeRange{wordType()},
						mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
						mlir::ValueRange{pointer}).getResult(0);
				}
				mlir::Value index = mapped(store.getIndex());
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
				emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
					loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, length)), 0x32);
				mlir::Value address = m_builder.create<mlir::yul::AddOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					index);
				m_builder.create<mlir::yul::MStore8Op>(
					loc(), address,
					m_builder.create<mlir::yul::ByteOp>(
						loc(), wordConstant(uint64_t(0)), mapped(store.getValue())));
				if (_op.hasAttr("storageArray"))
					m_builder.create<mlir::yul::FuncCallOp>(
						loc(), mlir::TypeRange{},
						mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
						mlir::ValueRange{mapped(store.getArray()), pointer});
			}
			else if (_op.hasAttr("storageArray"))
			{
				mlir::Value length;
				if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(store.getArray().getType()))
					length = array.isDynamicallySized()
						? m_builder.create<mlir::yul::SLoadOp>(loc(), arrayBaseSlot(_op)).getResult()
						: wordConstant(static_cast<uint64_t>(array.getSize()));
				if (length)
					emitPanicIf(m_builder.create<mlir::yul::IsZeroOp>(
						loc(), m_builder.create<mlir::yul::LtOp>(loc(), mapped(store.getIndex()), length)), 0x32);
				mlir::Type targetElement = llvm::cast<mlir::solidity::ArrayType>(
					store.getArray().getType()).getElementType();
				mlir::Value targetSlot = arrayElementSlot(_op, store.getIndex());
				if (isDynamic(targetElement) || isABIArray(targetElement)
					|| mlir::isa<mlir::solidity::StructType>(targetElement))
				{
					mlir::Value source = store.getValue();
					mlir::Type sourceType = source.getType();
					if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
						if (isDynamic(conversion.getInput().getType())
							|| isABIArray(conversion.getInput().getType())
							|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
						{
							source = conversion.getInput();
							sourceType = source.getType();
						}
					if (_op.hasAttr("storageValue"))
						copyStorageValueToStorage(
							sourceType, mapped(source), targetElement, targetSlot);
					else
						copyMemoryValueToStorage(
							sourceType, mapped(source), targetElement, targetSlot);
				}
				else
					storeStorageArrayElement(
						targetElement, targetSlot,
						mapped(store.getIndex()), mapped(store.getValue()), elementBytes(_op));
			}
			else
			{
				mlir::Value value = mapped(store.getValue());
				if (_op.hasAttr("storageValue"))
					value = materializeStorageValue(store.getValue().getType(), value);
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					memoryElement(_op, mapped(store.getArray()), mapped(store.getIndex())),
					value);
			}
			return false;
		}
		if (auto length = llvm::dyn_cast<mlir::solidity::ArrayLengthOp>(&_op))
		{
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(length.getArray().getType());
				array && !array.isDynamicallySized())
			{
				m_map[length.getResult()] = wordConstant(static_cast<uint64_t>(array.getSize()));
				return false;
			}
			// A dynamic array keeps its length in its own slot; in memory it is
			// the word the pointer addresses.
			if (_op.hasAttr("storageArray"))
				m_map[length.getResult()]
					= m_builder.create<mlir::yul::SLoadOp>(loc(), arrayBaseSlot(_op));
			else
				m_map[length.getResult()]
					= m_builder.create<mlir::yul::MLoadOp>(loc(), mapped(length.getArray()));
			return false;
		}
		if (auto push = llvm::dyn_cast<mlir::solidity::ArrayPushOp>(&_op))
		{
			if (!_op.hasAttr("storageArray") || !_op.hasAttr("dynamic"))
				fail("push onto an array that is not dynamic storage");
			mlir::Value slot = arrayBaseSlot(_op);
			mlir::Value length = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			// The new element goes where the old length pointed, and the length
			// moves on.
			m_builder.create<mlir::yul::SStoreOp>(
				loc(), slot, m_builder.create<mlir::yul::AddOp>(loc(), length, wordConstant(uint64_t(1))));
			mlir::Type elementType = llvm::cast<mlir::solidity::ArrayType>(push.getArray().getType()).getElementType();
			unsigned const bytes = elementBytes(_op);
			mlir::Value elementSlot = storageArrayElementSlot(
				dynamicArrayData(slot), length, elementType, bytes);
			if (isDynamic(elementType) || isABIArray(elementType)
				|| mlir::isa<mlir::solidity::StructType>(elementType))
			{
				mlir::Value source = push.getValue();
				mlir::Type sourceType = source.getType();
				if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
					if (isDynamic(conversion.getInput().getType())
						|| isABIArray(conversion.getInput().getType())
						|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
					{
						source = conversion.getInput();
						sourceType = source.getType();
					}
				if (_op.hasAttr("storageValue"))
					copyStorageValueToStorage(sourceType, mapped(source), elementType, elementSlot);
				else
					copyMemoryValueToStorage(sourceType, mapped(source), elementType, elementSlot);
			}
			else
				storeStorageArrayElement(
					elementType, elementSlot, length, mapped(push.getValue()), bytes);
			return false;
		}
		if (auto push = llvm::dyn_cast<mlir::solidity::ArrayPushEmptyOp>(&_op))
		{
			if (!_op.hasAttr("storageArray") || !_op.hasAttr("dynamic"))
				fail("empty push onto an array that is not dynamic storage");
			mlir::Value slot = arrayBaseSlot(_op);
			mlir::Value length = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			mlir::Type elementType = llvm::cast<mlir::solidity::ArrayType>(push.getArray().getType()).getElementType();
			unsigned const bytes = elementBytes(_op);
			mlir::Value element = storageArrayElementSlot(
				dynamicArrayData(slot), length, elementType, bytes);
			if (packedType(elementType))
				storeStorageArrayElement(
					elementType, element, length, wordConstant(uint64_t(0)), bytes);
			else
				clearStorageValue(elementType, element);
			m_builder.create<mlir::yul::SStoreOp>(
				loc(), slot, m_builder.create<mlir::yul::AddOp>(loc(), length, wordConstant(uint64_t(1))));
			m_map[push.getResult()] = packedType(elementType)
				? wordConstant(uint64_t(0)) : element;
			return false;
		}
		if (auto push = llvm::dyn_cast<mlir::solidity::StorageBytesPushEmptyOp>(&_op))
		{
			m_needsBytesHelpers = m_usesMemory = true;
			mlir::Value slot = mapped(push.getBytes());
			mlir::Value oldValue = m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{wordType()},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
				mlir::ValueRange{slot})->getResult(0);
			mlir::Value oldLength = m_builder.create<mlir::yul::MLoadOp>(loc(), oldValue);
			mlir::Value newLength = m_builder.create<mlir::yul::AddOp>(
				loc(), oldLength, wordConstant(uint64_t(1)));
			emitPanicIf(isZero(newLength), 0x41);
			mlir::Value pointer = allocate(m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)), roundedUp(newLength)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, newLength);
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::AddOp>(loc(), oldValue, wordConstant(uint64_t(32))),
				oldLength);
			m_builder.create<mlir::yul::MStore8Op>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(uint64_t(32))), oldLength),
				wordConstant(uint64_t(0)));
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{slot, pointer});
			m_map[push.getResult()] = wordConstant(uint64_t(0));
			return false;
		}
		if (auto push = llvm::dyn_cast<mlir::solidity::StorageBytesPushOp>(&_op))
		{
			m_needsBytesHelpers = m_usesMemory = true;
			mlir::Value slot = mapped(push.getBytes());
			mlir::Value oldValue = m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{wordType()},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
				mlir::ValueRange{slot})->getResult(0);
			mlir::Value oldLength = m_builder.create<mlir::yul::MLoadOp>(loc(), oldValue);
			mlir::Value newLength = m_builder.create<mlir::yul::AddOp>(
				loc(), oldLength, wordConstant(uint64_t(1)));
			emitPanicIf(isZero(newLength), 0x41);
			mlir::Value pointer = allocate(m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)), roundedUp(newLength)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, newLength);
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::AddOp>(loc(), oldValue, wordConstant(uint64_t(32))),
				oldLength);
			m_builder.create<mlir::yul::MStore8Op>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(uint64_t(32))), oldLength),
				m_builder.create<mlir::yul::ByteOp>(
					loc(), wordConstant(uint64_t(0)), mapped(push.getValue())));
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{slot, pointer});
			return false;
		}
		if (auto pop = llvm::dyn_cast<mlir::solidity::StorageBytesPopOp>(&_op))
		{
			m_needsBytesHelpers = m_usesMemory = true;
			mlir::Value slot = mapped(pop.getBytes());
			mlir::Value value = m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{wordType()},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
				mlir::ValueRange{slot})->getResult(0);
			mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), value);
			emitPanicIf(isZero(length), 0x31);
			m_builder.create<mlir::yul::MStoreOp>(
				loc(), value,
				m_builder.create<mlir::yul::SubOp>(
					loc(), length, wordConstant(uint64_t(1))));
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{slot, value});
			return false;
		}
		if (auto pop = llvm::dyn_cast<mlir::solidity::ArrayPopOp>(&_op))
		{
			if (!_op.hasAttr("storageArray") || !_op.hasAttr("dynamic"))
				fail("pop from an array that is not dynamic storage");
			mlir::Value slot = arrayBaseSlot(_op);
			mlir::Value length = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
			emitPanicIf(isZero(length), 0x31);
			mlir::Value next = m_builder.create<mlir::yul::SubOp>(loc(), length, wordConstant(uint64_t(1)));
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, next);
			mlir::Type elementType = llvm::cast<mlir::solidity::ArrayType>(pop.getArray().getType()).getElementType();
			unsigned const bytes = elementBytes(_op);
			mlir::Value element = storageArrayElementSlot(
				dynamicArrayData(slot), next, elementType, bytes);
			if (packedType(elementType))
				storeStorageArrayElement(
					elementType, element, next, wordConstant(uint64_t(0)), bytes);
			else
				clearStorageValue(elementType, element);
			return false;
		}
		if (auto create = llvm::dyn_cast<mlir::solidity::StructCreateOp>(&_op))
		{
			// A struct in memory is its fields, one word each, in order.
			m_usesMemory = true;
			unsigned const fields = create.getFields().size();
			if (fields == 0 && mlir::isa<mlir::solidity::StructType>(create.getResult().getType()))
			{
				m_map[create.getResult()] = allocateDefaultMemoryValue(create.getResult().getType());
				return false;
			}
			mlir::Value pointer = allocate(wordConstant(uint64_t(32 * fields)));
			for (auto [position, field]: llvm::enumerate(create.getFields()))
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32 * position))),
					mapped(field));
			m_map[create.getResult()] = pointer;
			return false;
		}
		if (auto member = llvm::dyn_cast<mlir::solidity::MemberAccessOp>(&_op))
		{
			// The storage form went through storage_member_load; this is the
			// memory one, where the struct is a pointer and the member is a
			// word into it.
			auto index = _op.getAttrOfType<mlir::IntegerAttr>("fieldIndex");
			if (!index)
				fail("member access without a field index");
			m_map[member.getResult()] = m_builder.create<mlir::yul::MLoadOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), mapped(member.getObject()), wordConstant(uint64_t(32 * index.getInt()))));
			return false;
		}
		if (auto emit = llvm::dyn_cast<mlir::solidity::EmitOp>(&_op))
		{
			// An event is a log: its signature hash is topic0 unless it is
			// anonymous, each indexed argument is a further topic, and the rest
			// are ABI-encoded into the data.
			llvm::SmallVector<mlir::Value, 4> topics;
			if (!emit.getAnonymous())
			{
				auto signature = emit.getEventSignature();
				if (!signature)
					fail("emit without a signature");
				topics.push_back(
					wordConstant(llvm::APInt(256, solidity::u256(solidity::util::keccak256(signature->str())).str(), 10)));
			}

			llvm::SmallVector<mlir::Type, 4> dataTypes;
			llvm::SmallVector<mlir::Value, 4> dataValues;
			auto indexed = emit.getIndexed();
			for (auto [position, argument]: llvm::enumerate(emit.getArgs()))
			{
				mlir::Type type = argument.getType();
				if (!isABIEncodable(type))
					fail("emit with a non-elementary argument");
				bool const isTopic = indexed && position < indexed->size()
					&& llvm::cast<mlir::BoolAttr>((*indexed)[position]).getValue();
				mlir::Value value = abiOperand(
					emit.getOperation(), position, type, mapped(argument));
				if (!isTopic)
				{
					dataTypes.push_back(type);
					dataValues.push_back(value);
					continue;
				}

				if (packedType(type))
				{
					topics.push_back(value);
					continue;
				}

				// Indexed reference values are represented by keccak256 of their
				// special in-place encoding. Bytes/string hash their raw contents;
				// a value array hashes its concatenated 32-byte elements.
				m_usesMemory = true;
				if (isDynamic(type))
					topics.push_back(m_builder.create<mlir::yul::Keccak256Op>(
						loc(),
						m_builder.create<mlir::yul::AddOp>(loc(), value, wordConstant(uint64_t(32))),
						m_builder.create<mlir::yul::MLoadOp>(loc(), value)));
				else
				{
					auto array = llvm::cast<mlir::solidity::ArrayType>(type);
					mlir::Value length = array.isDynamicallySized()
						? m_builder.create<mlir::yul::MLoadOp>(loc(), value).getResult()
						: wordConstant(static_cast<uint64_t>(array.getSize()));
					mlir::Value data = array.isDynamicallySized()
						? m_builder.create<mlir::yul::AddOp>(loc(), value, wordConstant(uint64_t(32))).getResult()
						: value;
					topics.push_back(m_builder.create<mlir::yul::Keccak256Op>(
						loc(), data, m_builder.create<mlir::yul::MulOp>(loc(), length, wordConstant(uint64_t(32)))));
				}
			}
			if (topics.size() > 4)
				fail("emit with more topics than a log can carry");

			mlir::Value pointer = wordConstant(uint64_t(0));
			mlir::Value length = wordConstant(uint64_t(0));
			if (!dataValues.empty())
			{
				mlir::Value encoded = encodeABIValues(dataTypes, dataValues);
				length = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
				pointer = m_builder.create<mlir::yul::AddOp>(loc(), encoded, wordConstant(uint64_t(32)));
			}

			switch (topics.size())
			{
			case 0: m_builder.create<mlir::yul::Log0Op>(loc(), pointer, length); break;
			case 1: m_builder.create<mlir::yul::Log1Op>(loc(), pointer, length, topics[0]); break;
			case 2: m_builder.create<mlir::yul::Log2Op>(loc(), pointer, length, topics[0], topics[1]); break;
			case 3:
				m_builder.create<mlir::yul::Log3Op>(loc(), pointer, length, topics[0], topics[1], topics[2]);
				break;
			default:
				m_builder.create<mlir::yul::Log4Op>(
					loc(), pointer, length, topics[0], topics[1], topics[2], topics[3]);
				break;
			}
			return false;
		}
		if (auto packed = llvm::dyn_cast<mlir::solidity::AbiEncodePackedOp>(&_op))
		{
			m_map[packed.getResult()] = encodePacked(packed.getArgs(), packed.getOperation());
			return false;
		}
		if (auto concatenated = llvm::dyn_cast<mlir::solidity::ConcatOp>(&_op))
		{
			// Both Solidity concat builtins have the same memory representation
			// and byte-exact copy semantics as packed encoding for their permitted
			// dynamic and bytesN operands.
			m_map[concatenated.getResult()] = encodePacked(
				concatenated.getArgs(), concatenated.getOperation());
			return false;
		}
		if (auto encoded = llvm::dyn_cast<mlir::solidity::AbiEncodeOp>(&_op))
		{
			m_map[encoded.getResult()] = encodeABI(encoded.getArgs(), encoded.getOperation());
			return false;
		}
		if (auto encoded = llvm::dyn_cast<mlir::solidity::AbiEncodeWithSelectorOp>(&_op))
		{
			if (encoded.getArgs().empty())
				fail("abi.encodeWithSelector without a selector");
			m_map[encoded.getResult()] = encodeABIWithSelector(
				mapped(encoded.getArgs().front()), encoded.getArgs().drop_front(), encoded.getOperation(), 1);
			return false;
		}
		if (auto encoded = llvm::dyn_cast<mlir::solidity::AbiEncodeWithSignatureOp>(&_op))
		{
			if (encoded.getArgs().empty() || !isDynamic(encoded.getArgs().front().getType()))
				fail("abi.encodeWithSignature without a string signature");
			mlir::Value signature = mapped(encoded.getArgs().front());
			mlir::Value selector = m_builder.create<mlir::yul::Keccak256Op>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), signature, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::MLoadOp>(loc(), signature));
			m_map[encoded.getResult()] = encodeABIWithSelector(
				selector, encoded.getArgs().drop_front(), encoded.getOperation(), 1);
			return false;
		}
		if (auto hash = llvm::dyn_cast<mlir::solidity::Keccak256Op>(&_op))
		{
			if (!isDynamic(hash.getData().getType()))
				fail("keccak256 input is not bytes or string in memory");
			mlir::Value pointer = mapped(hash.getData());
			mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
			m_map[hash.getResult()] = m_builder.create<mlir::yul::Keccak256Op>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				length);
			return false;
		}
		if (auto hash = llvm::dyn_cast<mlir::solidity::Sha256Op>(&_op))
		{
			if (!isDynamic(hash.getData().getType()))
				fail("sha256 input is not bytes or string in memory");
			mlir::Value pointer = mapped(hash.getData());
			m_map[hash.getResult()] = callPrecompile(
				2,
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::MLoadOp>(loc(), pointer));
			return false;
		}
		if (auto hash = llvm::dyn_cast<mlir::solidity::Ripemd160Op>(&_op))
		{
			if (!isDynamic(hash.getData().getType()))
				fail("ripemd160 input is not bytes or string in memory");
			mlir::Value pointer = mapped(hash.getData());
			// The precompile returns a right-aligned 160-bit word, whereas bytes20
			// is left-aligned in Solidity's stack representation.
			m_map[hash.getResult()] = yulShl(
				loc(),
				wordConstant(uint64_t(96)),
				callPrecompile(
					3,
					m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					m_builder.create<mlir::yul::MLoadOp>(loc(), pointer)));
			return false;
		}
		if (auto recovery = llvm::dyn_cast<mlir::solidity::EcrecoverOp>(&_op))
		{
			m_usesMemory = true;
			mlir::Value input = allocate(wordConstant(uint64_t(128)));
			for (auto [index, value]: llvm::enumerate(recovery->getOperands()))
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), input, wordConstant(uint64_t(32 * index))),
					mapped(value));
			m_map[recovery.getResult()] = callPrecompile(1, input, wordConstant(uint64_t(128)));
			return false;
		}
		if (llvm::isa<mlir::solidity::AddModOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::AddModOp>(
				loc(), mapped(_op.getOperand(0)), mapped(_op.getOperand(1)), mapped(_op.getOperand(2)));
			return false;
		}
		if (llvm::isa<mlir::solidity::MulModOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::MulModOp>(
				loc(), mapped(_op.getOperand(0)), mapped(_op.getOperand(1)), mapped(_op.getOperand(2)));
			return false;
		}
		if (llvm::isa<mlir::solidity::GasLeftOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::GasOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::BlockhashOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::BlockHashOp>(loc(), mapped(_op.getOperand(0)));
			return false;
		}
		if (llvm::isa<mlir::solidity::BlobHashOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::BlobHashOp>(loc(), mapped(_op.getOperand(0)));
			return false;
		}
		if (llvm::isa<mlir::solidity::SelfAddressOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::AddressOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::MsgSenderOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::CallerOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::MsgValueOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::CallValueOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::MsgSigOp>(&_op))
		{
			// Fixed bytes are left-aligned in an EVM word. Keep the first four
			// calldata bytes in their original position and clear the rest.
			mlir::Value word = m_builder.create<mlir::yul::CallDataLoadOp>(loc(), wordConstant(uint64_t(0)));
			m_map[_op.getResult(0)] = yulShl(
				loc(),
				wordConstant(uint64_t(224)),
				yulShr(loc(), wordConstant(uint64_t(224)), word));
			return false;
		}
		if (auto data = llvm::dyn_cast<mlir::solidity::MsgDataOp>(&_op))
		{
			m_usesMemory = true;
			mlir::Value length = m_builder.create<mlir::yul::CallDataSizeOp>(loc());
			mlir::Value pointer = allocate(
				m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(length)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
			m_builder.create<mlir::yul::CallDataCopyOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				wordConstant(uint64_t(0)),
				length);
			m_map[data.getResult()] = pointer;
			return false;
		}
		if (llvm::isa<mlir::solidity::BlockTimestampOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::TimestampOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::BlockNumberOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::NumberOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::BlockChainIdOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::ChainIdOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::TxOriginOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::OriginOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::TxGasPriceOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::GasPriceOp>(loc());
			return false;
		}
		if (llvm::isa<mlir::solidity::AddressBalanceOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::BalanceOp>(loc(), mapped(_op.getOperand(0)));
			return false;
		}
		if (llvm::isa<mlir::solidity::AddressCodehashOp>(&_op))
		{
			m_map[_op.getResult(0)] = m_builder.create<mlir::yul::ExtCodeHashOp>(loc(), mapped(_op.getOperand(0)));
			return false;
		}
		if (auto code = llvm::dyn_cast<mlir::solidity::AddressCodeOp>(&_op))
		{
			m_usesMemory = true;
			mlir::Value address = mapped(code.getAddress());
			mlir::Value length = m_builder.create<mlir::yul::ExtCodeSizeOp>(loc(), address);
			mlir::Value pointer = allocate(
				m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(length)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
			m_builder.create<mlir::yul::ExtCodeCopyOp>(
				loc(),
				address,
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				wordConstant(uint64_t(0)),
				length);
			m_map[code.getCode()] = pointer;
			return false;
		}
		if (auto logicalNot = llvm::dyn_cast<mlir::solidity::LogicalNotOp>(&_op))
		{
			m_map[logicalNot.getResult()] = m_builder.create<mlir::yul::IsZeroOp>(
				loc(), mapped(logicalNot.getOperand()));
			return false;
		}
		if (auto transfer = llvm::dyn_cast<mlir::solidity::TransferOp>(&_op))
		{
			mlir::Value amount = mapped(transfer.getAmount());
			// CALL itself supplies the 2300 value-transfer stipend. Solidity asks
			// for another 2300 only when the amount is zero, keeping transfer's
			// effective callee budget consistent in both cases.
			mlir::Value gas = m_builder.create<mlir::yul::MulOp>(
				loc(), wordConstant(uint64_t(2300)), isZero(amount));
			mlir::Value zero = wordConstant(uint64_t(0));
			mlir::Value success = m_builder.create<mlir::yul::CallOp>(
				loc(), wordType(), gas, mapped(transfer.getTo()), amount,
				zero, zero, zero, zero);
			auto failed = m_builder.create<mlir::yul::IfOp>(loc(), isZero(success));
			{
				mlir::OpBuilder::InsertionGuard guard(m_builder);
				m_builder.setInsertionPointToStart(&failed.getThenRegion().emplaceBlock());
				mlir::Value size = m_builder.create<mlir::yul::ReturnDataSizeOp>(loc());
				mlir::Value pointer = allocate(roundedUp(size));
				m_builder.create<mlir::yul::ReturnDataCopyOp>(
					loc(), pointer, wordConstant(uint64_t(0)), size);
				m_builder.create<mlir::yul::RevertOp>(loc(), pointer, size);
			}
			return false;
		}
		if (llvm::isa<mlir::solidity::SelfdestructOp>(&_op))
		{
			m_builder.create<mlir::yul::SelfDestructOp>(loc(), mapped(_op.getOperand(0)));
			return true;
		}
		if (auto decode = llvm::dyn_cast<mlir::solidity::AbiDecodeOp>(&_op))
		{
			// The first operand is encoded bytes in memory: [length][data...].
			// Decode the same ABI subset accepted by encodeABIValues, including
			// byte sequences and value arrays.
			for (mlir::Type type: decode->getResultTypes())
				if (!isABIEncodable(type))
					fail("abi.decode of a non-elementary component");

			mlir::Value encoded = mapped(decode.getOperands().front());
			mlir::Value encodedLength = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
			mlir::Value data = m_builder.create<mlir::yul::AddOp>(
				loc(), encoded, wordConstant(uint64_t(32)));
			auto minimumHead = abiTupleHeadSize(decode->getResultTypes());
			if (!minimumHead)
				fail("abi.decode of a non-elementary component");
			mlir::Value tooShort = m_builder.create<mlir::yul::LtOp>(
				loc(), encodedLength, wordConstant(*minimumHead));
			if (m_debugRevertStrings)
				emitErrorStringRevertIf(tooShort, "Calldata too short");
			else
				emitRevertIf(tooShort);

			uint64_t headOffset = 0;
			for (auto result: decode->getResults())
			{
				mlir::Type type = result.getType();
				mlir::Value head = m_builder.create<mlir::yul::AddOp>(
					loc(), data, wordConstant(headOffset));
				m_map[result] = decodeABIComponent(type, head, data, encodedLength);
				headOffset += *abiHeadSize(type);
			}
			return false;
		}
		if (auto reference = llvm::dyn_cast<mlir::solidity::ExternalFunctionRefOp>(&_op))
		{
			mlir::Value address = m_builder.create<mlir::yul::AndOp>(
				loc(), mapped(reference.getTarget()),
				wordConstant(llvm::APInt::getLowBitsSet(256, 160)));
			mlir::Value shifted = yulShl(
				loc(), wordConstant(uint64_t(32)), address);
			m_map[reference.getResult()] = m_builder.create<mlir::yul::OrOp>(
				loc(), shifted,
				wordConstant(uint64_t(static_cast<uint32_t>(reference.getSelector()))));
			return false;
		}
		if (auto call = llvm::dyn_cast<mlir::solidity::ExternalFunctionCallOp>(&_op))
		{
			for (mlir::Value argument: call.getArgs())
				if (!isABIEncodable(argument.getType()))
					fail("external call with a non-elementary argument");
			for (mlir::Type result: call->getResultTypes())
				if (!isABIEncodable(result))
					fail("external call with a non-elementary result");

			m_usesMemory = true;
			llvm::SmallVector<mlir::Type, 4> argumentTypes;
			llvm::SmallVector<mlir::Value, 4> argumentValues;
			for (auto [index, argument]: llvm::enumerate(call.getArgs()))
			{
				argumentTypes.push_back(argument.getType());
				mlir::Value value = abiOperand(
					call.getOperation(), index, argument.getType(), mapped(argument));
				if (uint64_t const enumLimit = indexedIntegerAttr(
						call.getOperation(), "enum_arg_limits", index))
					emitPanicIf(
						isZero(m_builder.create<mlir::yul::LtOp>(loc(), value, wordConstant(enumLimit))),
						0x21);
				if (indexedAttr(call.getOperation(), "external_function_args", index))
					value = yulShl(
						loc(), wordConstant(uint64_t(64)), value);
				else if (indexedAttr(call.getOperation(), "external_function_array_args", index))
					value = denormalizeExternalFunctionArray(argument.getType(), value);
				argumentValues.push_back(value);
			}
			mlir::Value encoded = encodeABIValues(argumentTypes, argumentValues);
			mlir::Value encodedLength = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
			mlir::Value inputLength = m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(4)), encodedLength);
			mlir::Value input = allocate(roundedUp(inputLength));
			mlir::Value target = mapped(call.getTarget());
			mlir::Value selectorWord;
			if (call->hasAttr("functionPointer"))
			{
				mlir::Value pointer = target;
				target = m_builder.create<mlir::yul::AndOp>(
					loc(),
					yulShr(loc(), wordConstant(uint64_t(32)), pointer),
					wordConstant(llvm::APInt::getLowBitsSet(256, 160)));
				selectorWord = yulShl(
					loc(),
					wordConstant(uint64_t(224)),
					m_builder.create<mlir::yul::AndOp>(
						loc(), pointer, wordConstant(uint64_t(0xffffffffu))));
			}
			else
			{
				llvm::APInt selector(256, static_cast<uint64_t>(call.getSelector()));
				selector <<= 224;
				selectorWord = wordConstant(selector);
			}
			m_builder.create<mlir::yul::MStoreOp>(loc(), input, selectorWord);
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), input, wordConstant(uint64_t(4))),
				m_builder.create<mlir::yul::AddOp>(loc(), encoded, wordConstant(uint64_t(32))),
				encodedLength);

			// Dynamic returndata has no size known before the call. Ask CALL to
			// copy none, then copy and decode the complete return buffer after it
			// succeeds. The same path also handles all-static tuples.
			mlir::Value noOutput = wordConstant(uint64_t(0));
			mlir::Value success;
			if (call.getIsStatic())
				success = m_builder.create<mlir::yul::StaticCallOp>(
					loc(),
					wordType(),
					m_builder.create<mlir::yul::GasOp>(loc()),
					target,
					input,
					inputLength,
					noOutput,
					noOutput);
			else
				success = m_builder.create<mlir::yul::CallOp>(
					loc(),
					wordType(),
					m_builder.create<mlir::yul::GasOp>(loc()),
					target,
					wordConstant(uint64_t(0)),
					input,
					inputLength,
					noOutput,
					noOutput);

			// A high-level external call bubbles the exact revert data from the
			// callee. Copying into the input buffer is safe because this path
			// terminates immediately.
			auto failed = m_builder.create<mlir::yul::IfOp>(loc(), isZero(success));
			{
				mlir::OpBuilder::InsertionGuard guard(m_builder);
				m_builder.setInsertionPointToStart(&failed.getThenRegion().emplaceBlock());
				mlir::Value size = m_builder.create<mlir::yul::ReturnDataSizeOp>(loc());
				m_builder.create<mlir::yul::ReturnDataCopyOp>(
					loc(), input, wordConstant(uint64_t(0)), size);
				m_builder.create<mlir::yul::RevertOp>(loc(), input, size);
			}

			mlir::Value returnSize = m_builder.create<mlir::yul::ReturnDataSizeOp>(loc());
			auto outputHeadSize = abiTupleHeadSize(call->getResultTypes());
			if (!outputHeadSize)
				fail("external call with a non-elementary result");
			uint64_t const minimumOutputSize = *outputHeadSize;
			if (minimumOutputSize > 0)
			{
				mlir::Value tooShort = m_builder.create<mlir::yul::LtOp>(
					loc(), returnSize, wordConstant(minimumOutputSize));
				if (m_debugRevertStrings)
					emitErrorStringRevertIf(tooShort, "ABI memory decoding: invalid data length");
				else
					emitRevertIf(tooShort);
			}
			mlir::Value output = call->getNumResults() == 0
				? noOutput
				: allocate(roundedUp(returnSize));
			if (call->getNumResults() > 0)
				m_builder.create<mlir::yul::ReturnDataCopyOp>(
					loc(), output, wordConstant(uint64_t(0)), returnSize);

			uint64_t resultHeadOffset = 0;
			for (auto [resultIndex, result]: llvm::enumerate(call->getResults()))
			{
				mlir::Type type = result.getType();
				mlir::Value head = m_builder.create<mlir::yul::AddOp>(
					loc(), output, wordConstant(resultHeadOffset));
				mlir::Value value = decodeABIComponent(
					type, head, output, returnSize, ABIDecodeContext::Memory);
				if (indexedAttr(call.getOperation(), "external_function_results", resultIndex))
					value = yulShr(
						loc(), wordConstant(uint64_t(64)), value);
				else if (indexedAttr(call.getOperation(), "external_function_array_results", resultIndex))
					value = normalizeExternalFunctionArray(type, value);
				m_map[result] = value;
				resultHeadOffset += *abiHeadSize(type);
			}
			return false;
		}
		if (auto call = llvm::dyn_cast<mlir::solidity::LowLevelCallOp>(&_op))
		{
			if (!isDynamic(call.getData().getType()))
				fail("low-level call input is not bytes in memory");
			m_usesMemory = true;
			mlir::Value data = mapped(call.getData());
			mlir::Value input = m_builder.create<mlir::yul::AddOp>(
				loc(), data, wordConstant(uint64_t(32)));
			mlir::Value inputSize = m_builder.create<mlir::yul::MLoadOp>(loc(), data);
			mlir::Value zero = wordConstant(uint64_t(0));
			mlir::Value success;
			if (call.getKind() == "staticcall")
				success = m_builder.create<mlir::yul::StaticCallOp>(
					loc(), wordType(), m_builder.create<mlir::yul::GasOp>(loc()),
					mapped(call.getTo()), input, inputSize, zero, zero);
			else if (call.getKind() == "delegatecall")
				success = m_builder.create<mlir::yul::DelegateCallOp>(
					loc(), wordType(), m_builder.create<mlir::yul::GasOp>(loc()),
					mapped(call.getTo()), input, inputSize, zero, zero);
			else if (call.getKind() == "call")
				success = m_builder.create<mlir::yul::CallOp>(
					loc(), wordType(), m_builder.create<mlir::yul::GasOp>(loc()),
					mapped(call.getTo()), mapped(call.getValue()), input, inputSize, zero, zero);
			else
				fail("unknown low-level call kind: " + call.getKind().str());

			mlir::Value size = m_builder.create<mlir::yul::ReturnDataSizeOp>(loc());
			mlir::Value result = allocate(
				m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(size)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), result, size);
			m_builder.create<mlir::yul::ReturnDataCopyOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), result, wordConstant(uint64_t(32))),
				zero,
				size);
			m_map[call.getSuccess()] = success;
			m_map[call.getReturndata()] = result;
			return false;
		}
		if (auto create = llvm::dyn_cast<mlir::solidity::CreateContractOp>(&_op))
		{
			// `new C(...)` deploys C's *creation* code, which is a different
			// object from the runtime code `type(C).runtimeCode` names - so it
			// is nested under its own name. Constructor arguments follow the
			// code, which is where the constructor reads them from.
			std::string const object = create.getContractName().str() + kCreationObjectSuffix;
			m_referencedObjects.insert(object);
			m_usesMemory = true;

			mlir::Value code = m_builder.create<mlir::yul::DataSizeOp>(loc(), object);
			bool hasDynamicArgument = false;
			for (mlir::Value argument: create.getConstructorArgs())
				hasDynamicArgument = hasDynamicArgument || isABIDynamic(argument.getType());
			llvm::SmallVector<mlir::Type, 4> argumentTypes;
			llvm::SmallVector<mlir::Value, 4> argumentValues;
			for (auto [position, argument]: llvm::enumerate(create.getConstructorArgs()))
			{
				argumentTypes.push_back(argument.getType());
				mlir::Value value = mapped(argument);
				if (indexedAttr(create.getOperation(), "external_function_args", position))
					value = yulShl(
						loc(), wordConstant(uint64_t(64)), value);
				argumentValues.push_back(value);
			}
			mlir::Value encoded;
			mlir::Value encodedLength;
			if (hasDynamicArgument)
			{
				encoded = encodeABIValues(argumentTypes, argumentValues);
				encodedLength = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
			}
			mlir::Value argumentsLength = hasDynamicArgument
				? m_builder.create<mlir::yul::AddOp>(
					loc(), encodedLength, wordConstant(uint64_t(32))).getResult()
				: wordConstant(uint64_t(32 * create.getConstructorArgs().size()));
			mlir::Value total = m_builder.create<mlir::yul::AddOp>(loc(), code, argumentsLength);
			mlir::Value pointer = allocate(total);
			m_builder.create<mlir::yul::DataCopyOp>(
				loc(), pointer, m_builder.create<mlir::yul::DataOffsetOp>(loc(), object), code);

			mlir::Value at = m_builder.create<mlir::yul::AddOp>(loc(), pointer, code);
			if (hasDynamicArgument)
			{
				yulMCopy(
					loc(), at,
					m_builder.create<mlir::yul::AddOp>(loc(), encoded, wordConstant(uint64_t(32))),
					encodedLength);
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), at, encodedLength),
					encodedLength);
			}
			else
				for (auto [position, value]: llvm::enumerate(argumentValues))
					m_builder.create<mlir::yul::MStoreOp>(
						loc(),
						m_builder.create<mlir::yul::AddOp>(
							loc(), at, wordConstant(uint64_t(32 * position))),
						value);

			m_map[create.getResult()]
				= m_builder.create<mlir::yul::CreateOp>(loc(), wordType(), mapped(create.getValue()), pointer, total);
			return false;
		}
		if (auto code = llvm::dyn_cast<mlir::solidity::ContractCodeOp>(&_op))
		{
			// The named contract is nested as a sub-object, so its code is data
			// here: its size is known when the object is assembled, and copying
			// it into memory makes it an ordinary `bytes` value.
			// Runtime and creation objects are separately compiled children. The
			// latter includes its own nested runtime object, exactly as CREATE
			// expects.
			std::string const object = code.getContractName().str() +
				(code.getCreation() ? kCreationObjectSuffix : "");
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
			// A storage mapping, array, or struct used as an argument is a
			// reference to its base slot. Loading the contents here would turn
			// that reference into the current length/first word before the
			// callee has a chance to index it.
			if (load->hasAttr("asReference"))
			{
				m_map[load.getResult()] = slot;
				return false;
			}
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
			mlir::Value value = loadPackedState(load.getVarName(), load.getResult().getType());
			if (m_legacyCodegen && m_internalFunctionStateVars.contains(load.getVarName()))
			{
				// Legacy codegen exposes a nonzero invalid-entry tag for an
				// uninitialised internal function pointer. A call through it still
				// reaches the dispatcher's reverting fallback.
				llvm::APInt invalidTag = llvm::APInt::getAllOnes(256);
				value = m_builder.create<mlir::yul::OrOp>(
					loc(), value,
					m_builder.create<mlir::yul::MulOp>(
						loc(), m_builder.create<mlir::yul::IsZeroOp>(loc(), value),
						wordConstant(invalidTag)));
			}
			m_map[load.getResult()] = value;
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::StoreStateVarOp>(&_op))
		{
			mlir::Value slot = wordConstant(storageSlot(store.getVarName()));
			if (_op.hasAttr("deleteValue"))
			{
				clearStorageValue(store.getValue().getType(), slot);
				return false;
			}
			if (isDynamic(store.getValue().getType()))
			{
				m_needsBytesHelpers = m_usesMemory = true;
				mlir::Value value = mapped(store.getValue());
				if (_op.hasAttr("storageValue"))
					value = materializeStorageValue(store.getValue().getType(), value);
				m_builder.create<mlir::yul::FuncCallOp>(
					loc(),
					mlir::TypeRange{},
					mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
					mlir::ValueRange{slot, value});
				return false;
			}
			if (isABIArray(store.getValue().getType())
				|| mlir::isa<mlir::solidity::StructType>(store.getValue().getType()))
			{
				if (_op.hasAttr("storageValue"))
				{
					mlir::Value source = store.getValue();
					mlir::Type sourceType = source.getType();
					// Aggregate implicit conversion is representation-free, but its
					// source type is essential here: bytes8[9] and bytes17[10]
					// address and pack the same source slot very differently.
					if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
						if (isABIArray(conversion.getInput().getType())
							|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
						{
							source = conversion.getInput();
							sourceType = source.getType();
						}
					copyStorageValueToStorage(
						sourceType, mapped(source), store.getValue().getType(), slot);
					return false;
				}
				mlir::Value source = store.getValue();
				mlir::Type sourceType = source.getType();
				if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
					if (isABIArray(conversion.getInput().getType())
						|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
					{
						source = conversion.getInput();
						sourceType = source.getType();
					}
				copyMemoryValueToStorage(
					sourceType, mapped(source), store.getValue().getType(), slot);
				return false;
			}
			storePackedState(store.getVarName(), mapped(store.getValue()), store.getValue().getType());
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
			if (isDynamic(convert.getInput().getType())
				&& llvm::isa<mlir::solidity::BytesType>(convert.getOutput().getType()))
			{
				m_usesMemory = true;
				mlir::Value pointer = mapped(convert.getInput());
				if (_op.hasAttr("storageValue"))
					pointer = materializeStorageValue(convert.getInput().getType(), pointer);
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), pointer);
				mlir::Value data = m_builder.create<mlir::yul::MLoadOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(uint64_t(32))));
				unsigned const targetBytes
					= llvm::cast<mlir::solidity::BytesType>(convert.getOutput().getType()).getSize();
				auto retained = m_builder.create<mlir::yul::VarOp>(
					loc(), wordConstant(static_cast<uint64_t>(targetBytes)));
				auto shorter = m_builder.create<mlir::yul::IfOp>(
					loc(), m_builder.create<mlir::yul::LtOp>(
						loc(), length, wordConstant(static_cast<uint64_t>(targetBytes))));
				{
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&shorter.getThenRegion().emplaceBlock());
					m_builder.create<mlir::yul::AssignOp>(loc(), retained, length);
				}
				mlir::Value retainedValue = m_builder.create<mlir::yul::VarLoadOp>(
					loc(), wordType(), retained);
				mlir::Value shift = m_builder.create<mlir::yul::MulOp>(
					loc(),
					m_builder.create<mlir::yul::SubOp>(
						loc(), wordConstant(uint64_t(32)), retainedValue),
					wordConstant(uint64_t(8)));
				mlir::Value lowMask = m_builder.create<mlir::yul::SubOp>(
					loc(),
					yulShl(
						loc(), shift, wordConstant(uint64_t(1))),
					wordConstant(uint64_t(1)));
				mlir::Value highMask = m_builder.create<mlir::yul::NotOp>(loc(), lowMask);
				m_map[convert.getOutput()] = m_builder.create<mlir::yul::AndOp>(loc(), data, highMask);
				return false;
			}
			if (m_legacyCodegen)
				if (auto sourceBytes = llvm::dyn_cast<mlir::solidity::BytesType>(convert.getInput().getType()))
					if (auto targetBytes = llvm::dyn_cast<mlir::solidity::BytesType>(convert.getOutput().getType());
						targetBytes && targetBytes.getSize() < sourceBytes.getSize())
					{
						// The old code generator kept the discarded low bytes dirty until
						// a typed use cleaned them. Inline assembly can observe that word.
						m_map[convert.getOutput()] = mapped(convert.getInput());
						return false;
					}
			m_map[convert.getOutput()] = representationConvert(
				mapped(convert.getInput()), convert.getInput().getType(), convert.getOutput().getType());
			return false;
		}
		if (auto select = llvm::dyn_cast<mlir::solidity::SelectOp>(&_op))
		{
			// Branch-free, because both arms are already evaluated by the time
			// this op exists - `b xor ((a xor b) * c)` with c in {0,1}, which is
			// the same shape the EVM backend uses for arith.select.
			mlir::Value trueValue = mapped(select.getTrueValue());
			mlir::Value falseValue = mapped(select.getFalseValue());
			mlir::Value condition = mapped(select.getCondition());
			mlir::Value difference = m_builder.create<mlir::yul::XorOp>(loc(), trueValue, falseValue);
			mlir::Value chosen = m_builder.create<mlir::yul::MulOp>(loc(), difference, condition);
			m_map[select.getResult()] = m_builder.create<mlir::yul::XorOp>(loc(), falseValue, chosen);
			return false;
		}
		if (auto toI1 = llvm::dyn_cast<mlir::solidity::ToI1Op>(&_op))
		{
			// Everything is a word here, so narrowing a bool to i1 is only a
			// change of type - the value is already 0 or 1.
			m_map[toI1.getResult()] = mapped(toI1.getOperand());
			return false;
		}
		if (auto assembly = llvm::dyn_cast<mlir::solidity::InlineAssemblyOp>(&_op))
		{
			// Inline assembly may use Solidity's documented free-memory pointer
			// at 0x40 without containing a high-level allocation. Conservatively
			// initialise it before any such block; otherwise codecopy to
			// mload(0x40) overwrites the pointer and can turn the next memory
			// operation into an out-of-gas access.
			m_usesMemory = true;
			return convertInlineAssembly(assembly);
		}
		if (auto bind = llvm::dyn_cast<mlir::solidity::AssemblyBindOp>(&_op))
		{
			// What the block left in that variable.
			auto known = m_assemblyVars.find(bind.getVarName().str());
			if (known == m_assemblyVars.end())
				fail("inline assembly does not define '" + bind.getVarName().str() + "'");
			m_map[bind.getResult()] = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), known->second);
			return false;
		}
		if (auto scfIf = llvm::dyn_cast<mlir::scf::IfOp>(&_op))
		{
			convertIfWithResults(scfIf);
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
			mlir::ValueRange keys = access.getKeys();
			mlir::Value base = access->hasAttr("baseSlotOperand")
				? mapped(keys.front())
				: wordConstant(storageSlot(access.getVarName()));
			if (access->hasAttr("baseSlotOperand"))
				keys = keys.drop_front();
			validateMappingEnumKeys(access.getOperation(), keys);
			mlir::Value slot = mappingSlot(base, keys);
			// A struct does not fit a word, so what the expression names is the
			// place rather than what is in it.
			m_map[access.getResult()]
				= access.getAsReference() ? slot : m_builder.create<mlir::yul::SLoadOp>(loc(), slot).getResult();
			return false;
		}
		if (auto load = llvm::dyn_cast<mlir::solidity::StorageMemberLoadOp>(&_op))
		{
			mlir::Value slot = m_builder.create<mlir::yul::AddOp>(
				loc(), mapped(load.getSlot()), wordConstant(static_cast<uint64_t>(load.getOffset())));
			if (load->hasAttr("asReference"))
				m_map[load.getResult()] = slot;
			else
			{
				unsigned const byteOffset = load->getAttrOfType<mlir::IntegerAttr>("storageByteOffset")
					? static_cast<unsigned>(load->getAttrOfType<mlir::IntegerAttr>("storageByteOffset").getInt()) : 0;
				unsigned const bytes = load->getAttrOfType<mlir::IntegerAttr>("storageBytes")
					? static_cast<unsigned>(load->getAttrOfType<mlir::IntegerAttr>("storageBytes").getInt())
					: storageValueBytes(load.getResult().getType());
				mlir::Value value = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
				if (byteOffset)
					value = yulShr(
						loc(), wordConstant(static_cast<uint64_t>(byteOffset * 8)), value);
				if (bytes < 32)
					value = m_builder.create<mlir::yul::AndOp>(
						loc(), value, wordConstant(llvm::APInt::getLowBitsSet(256, bytes * 8)));
				m_map[load.getResult()] = representationConvert(
					value, mlir::solidity::UIntType::get(m_builder.getContext(), 256), load.getResult().getType());
			}
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::StorageMemberStoreOp>(&_op))
		{
			mlir::Value slot = m_builder.create<mlir::yul::AddOp>(
				loc(), mapped(store.getSlot()), wordConstant(static_cast<uint64_t>(store.getOffset())));
			mlir::Value source = store.getValue();
			mlir::Type targetType = source.getType();
			mlir::Type sourceType = targetType;
			if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
				if (isDynamic(conversion.getInput().getType())
					|| isABIArray(conversion.getInput().getType())
					|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
				{
					source = conversion.getInput();
					sourceType = source.getType();
				}
			if (isDynamic(targetType) || isABIArray(targetType)
				|| mlir::isa<mlir::solidity::StructType>(targetType))
			{
				if (_op.hasAttr("storageValue"))
					copyStorageValueToStorage(sourceType, mapped(source), targetType, slot);
				else
					copyMemoryValueToStorage(sourceType, mapped(source), targetType, slot);
				return false;
			}
			unsigned const byteOffset = store->getAttrOfType<mlir::IntegerAttr>("storageByteOffset")
				? static_cast<unsigned>(store->getAttrOfType<mlir::IntegerAttr>("storageByteOffset").getInt()) : 0;
			unsigned const bytes = store->getAttrOfType<mlir::IntegerAttr>("storageBytes")
				? static_cast<unsigned>(store->getAttrOfType<mlir::IntegerAttr>("storageBytes").getInt())
				: storageValueBytes(targetType);
			if (bytes >= 32 && byteOffset == 0)
				m_builder.create<mlir::yul::SStoreOp>(loc(), slot, mapped(source));
			else
			{
				mlir::Value raw = representationConvert(
					mapped(source), sourceType,
					mlir::solidity::UIntType::get(m_builder.getContext(), 256));
				llvm::APInt const mask = llvm::APInt::getLowBitsSet(256, bytes * 8);
				raw = m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(mask));
				unsigned const shiftBits = byteOffset * 8;
				if (shiftBits)
					raw = yulShl(loc(), wordConstant(shiftBits), raw);
				mlir::Value shiftedMask = shiftBits
					? yulShl(loc(), wordConstant(shiftBits), wordConstant(mask))
					: wordConstant(mask);
				mlir::Value retained = m_builder.create<mlir::yul::AndOp>(
					loc(), m_builder.create<mlir::yul::SLoadOp>(loc(), slot),
					m_builder.create<mlir::yul::NotOp>(loc(), shiftedMask));
				m_builder.create<mlir::yul::SStoreOp>(
					loc(), slot, m_builder.create<mlir::yul::OrOp>(loc(), retained, raw));
			}
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::MemoryMemberStoreOp>(&_op))
		{
			m_builder.create<mlir::yul::MStoreOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), mapped(store.getObject()),
					wordConstant(static_cast<uint64_t>(32 * store.getFieldIndex()))),
				mapped(store.getValue()));
			return false;
		}
		if (auto store = llvm::dyn_cast<mlir::solidity::MappingStoreOp>(&_op))
		{
			// The operands are the keys followed by the value; only the count
			// of keys says where the split is.
			unsigned const keyCount = static_cast<unsigned>(store.getNumKeys());
			bool const hasBase = store->hasAttr("baseSlotOperand");
			if (store.getOperands().size() != keyCount + 1 + (hasBase ? 1 : 0))
				fail("mapping_store operand count disagrees with its numKeys attribute");
			unsigned const keyStart = hasBase ? 1 : 0;
			mlir::Value base = hasBase
				? mapped(store.getOperands().front())
				: wordConstant(storageSlot(store.getVarName()));
			validateMappingEnumKeys(
				store.getOperation(), store.getOperands().slice(keyStart, keyCount));
			mlir::Value slot = mappingSlot(
				base, store.getOperands().slice(keyStart, keyCount));
			mlir::Value source = store.getOperands().back();
			mlir::Type targetType = source.getType();
			mlir::Type sourceType = targetType;
			// Aggregate conversions do not change their one-word reference at
			// this rung, but the original shape controls how storage is addressed.
			if (auto conversion = source.getDefiningOp<mlir::solidity::ConvertOp>())
				if (isDynamic(conversion.getInput().getType())
					|| isABIArray(conversion.getInput().getType())
					|| mlir::isa<mlir::solidity::StructType>(conversion.getInput().getType()))
				{
					source = conversion.getInput();
					sourceType = source.getType();
				}
			if (isDynamic(targetType))
			{
				m_needsBytesHelpers = m_usesMemory = true;
				mlir::Value value = mapped(source);
				if (_op.hasAttr("storageValue"))
					value = materializeStorageValue(sourceType, value);
				m_builder.create<mlir::yul::FuncCallOp>(
					loc(), mlir::TypeRange{},
					mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
					mlir::ValueRange{slot, value});
				return false;
			}
			if (isABIArray(targetType) || mlir::isa<mlir::solidity::StructType>(targetType))
			{
				if (_op.hasAttr("storageValue"))
					copyStorageValueToStorage(sourceType, mapped(source), targetType, slot);
				else
					copyMemoryValueToStorage(sourceType, mapped(source), targetType, slot);
				return false;
			}
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, mapped(source));
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
			for (auto [index, operand]: llvm::enumerate(ret.getOperands()))
				results.push_back(indexedAttr(ret.getOperation(), "storage_results", index)
					? materializeStorageValue(operand.getType(), mapped(operand))
					: mapped(operand));
			m_builder.create<mlir::yul::LeaveOp>(loc(), results);
			return true;
		}
		if (auto transfer = llvm::dyn_cast<mlir::solidity::ControlTransferOp>(&_op))
		{
			llvm::StringRef kind = transfer.getKind();
			if (kind == "return")
			{
				llvm::SmallVector<mlir::Value, 2> results;
				for (auto [index, operand]: llvm::enumerate(transfer.getOperands()))
					results.push_back(indexedAttr(transfer.getOperation(), "storage_results", index)
						? materializeStorageValue(operand.getType(), mapped(operand))
						: mapped(operand));
				m_builder.create<mlir::yul::LeaveOp>(loc(), results);
			}
			else if (kind == "break" || kind == "continue")
			{
				// A structured branch can transfer out before its scf.yield.
				// Persist the source values at that exact point into the
				// enclosing loop's mutable Yul slots first.
				if (!m_loopSlotStack.empty())
					for (auto [index, value]: llvm::enumerate(transfer.getOperands()))
						if (index < m_loopSlotStack.back().size())
							m_builder.create<mlir::yul::AssignOp>(
								loc(), m_loopSlotStack.back()[index], mapped(value));
				if (kind == "break")
					m_builder.create<mlir::yul::BreakOp>(loc());
				else
					m_builder.create<mlir::yul::ContinueOp>(loc());
			}
			else if (kind == "revert")
			{
				mlir::Value zero = wordConstant(uint64_t(0));
				m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
			}
			else
				fail("unknown nested control transfer: " + kind.str());
			return true;
		}
		if (llvm::isa<mlir::solidity::BreakOp>(&_op))
		{
			m_builder.create<mlir::yul::BreakOp>(loc());
			return true;
		}
		if (llvm::isa<mlir::solidity::ContinueOp>(&_op))
		{
			m_builder.create<mlir::yul::ContinueOp>(loc());
			return true;
		}
		if (auto require = llvm::dyn_cast<mlir::solidity::RequireOp>(&_op))
		{
			mlir::Value failed = m_builder.create<mlir::yul::IsZeroOp>(loc(), mapped(require.getCondition()));
			if (auto message = require.getMessageAttr())
				emitErrorStringRevertIf(failed, message.getValue().str());
			else if (require.getMessageValue())
				emitErrorStringRevertIf(failed, mapped(require.getMessageValue()));
			else
				emitRevertIf(failed);
			return false;
		}
		if (auto assertOp = llvm::dyn_cast<mlir::solidity::AssertOp>(&_op))
		{
			mlir::Value failed = m_builder.create<mlir::yul::IsZeroOp>(loc(), mapped(assertOp.getCondition()));
			emitPanicIf(failed, 0x01);
			return false;
		}
		if (auto revert = llvm::dyn_cast<mlir::solidity::RevertOp>(&_op))
		{
			if (auto panicCode = revert->getAttrOfType<mlir::IntegerAttr>("panicCode"))
				emitPanic(panicCode.getValue().getZExtValue());
			else if (auto reason = revert.getReasonAttr())
				emitErrorStringRevertIf(wordConstant(uint64_t(1)), reason.getValue().str());
			else
			{
				mlir::Value zero = wordConstant(uint64_t(0));
				m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
			}
			return true;
		}
		if (auto revert = llvm::dyn_cast<mlir::solidity::RevertValueOp>(&_op))
		{
			emitErrorStringRevertIf(wordConstant(uint64_t(1)), mapped(revert.getReason()));
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

	static bool needsDefaultMemoryAllocation(mlir::Type _type)
	{
		return llvm::isa<
			mlir::solidity::ArrayType,
			mlir::solidity::StructType,
			mlir::solidity::DynamicBytesType,
			mlir::solidity::StringType>(_type);
	}

	/// Reference-typed memory values own their pointees even when they are
	/// default-initialised. In particular, every element of `new S[](n)` must
	/// point at a distinct zeroed S, and a fixed array field inside S must point
	/// at its own words rather than the scratch area at address zero.
	mlir::Value allocateDefaultMemoryValue(mlir::Type _type)
	{
		if (llvm::isa<mlir::solidity::DynamicBytesType, mlir::solidity::StringType>(_type))
		{
			mlir::Value pointer = allocate(wordConstant(uint64_t(32)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, wordConstant(uint64_t(0)));
			return pointer;
		}
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
		{
			if (array.isDynamicallySized())
			{
				mlir::Value pointer = allocate(wordConstant(uint64_t(32)));
				m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, wordConstant(uint64_t(0)));
				return pointer;
			}
			mlir::Value count = wordConstant(static_cast<uint64_t>(array.getSize()));
			mlir::Value pointer = allocate(m_builder.create<mlir::yul::MulOp>(
				loc(), count, wordConstant(uint64_t(32))));
			initialiseMemoryArrayElements(array.getElementType(), pointer, count);
			return pointer;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			mlir::Value pointer = allocate(
				wordConstant(static_cast<uint64_t>(32 * structure.getFieldTypes().size())));
			for (auto [index, field]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value value = needsDefaultMemoryAllocation(field)
					? allocateDefaultMemoryValue(field) : wordConstant(uint64_t(0));
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(static_cast<uint64_t>(32 * index))),
					value);
			}
			return pointer;
		}
		return wordConstant(uint64_t(0));
	}

	void initialiseMemoryArrayElements(
		mlir::Type _elementType, mlir::Value _data, mlir::Value _count)
	{
		if (!needsDefaultMemoryAllocation(_elementType))
			return;
		emitIndexLoop(_count, [&](mlir::Value index) {
			m_builder.create<mlir::yul::MStoreOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), _data,
					m_builder.create<mlir::yul::MulOp>(
						loc(), index, wordConstant(uint64_t(32)))),
				allocateDefaultMemoryValue(_elementType));
		});
	}

	/// Invoke a fixed-output EVM precompile and return its 32-byte result. The
	/// output word is cleared first because ecrecover legitimately succeeds
	/// with empty output for malformed signatures, which Solidity exposes as
	/// address(0). A true call failure bubbles the exact returndata.
	mlir::Value callPrecompile(uint64_t _address, mlir::Value _input, mlir::Value _length)
	{
		m_usesMemory = true;
		mlir::Value output = allocate(wordConstant(uint64_t(32)));
		m_builder.create<mlir::yul::MStoreOp>(loc(), output, wordConstant(uint64_t(0)));
		mlir::Value success = m_builder.create<mlir::yul::StaticCallOp>(
			loc(),
			wordType(),
			m_builder.create<mlir::yul::GasOp>(loc()),
			wordConstant(_address),
			_input,
			_length,
			output,
			wordConstant(uint64_t(32)));

		auto failed = m_builder.create<mlir::yul::IfOp>(loc(), isZero(success));
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&failed.getThenRegion().emplaceBlock());
			mlir::Value size = m_builder.create<mlir::yul::ReturnDataSizeOp>(loc());
			m_builder.create<mlir::yul::ReturnDataCopyOp>(
				loc(), output, wordConstant(uint64_t(0)), size);
			m_builder.create<mlir::yul::RevertOp>(loc(), output, size);
		}
		return m_builder.create<mlir::yul::MLoadOp>(loc(), output);
	}

	struct PackedType
	{
		unsigned bytes;
		bool leftAligned;
	};

	/// Width and in-word alignment used by abi.encodePacked for an elementary
	/// type. Integer-like values occupy their low bytes; bytesN occupies its
	/// high bytes. Dynamic bytes and string are handled separately.
	static std::optional<PackedType> packedType(mlir::Type _type)
	{
		if (auto uintType = llvm::dyn_cast<mlir::solidity::UIntType>(_type))
			return PackedType{uintType.getBitWidth() / 8, false};
		if (auto intType = llvm::dyn_cast<mlir::solidity::IntType>(_type))
			return PackedType{intType.getBitWidth() / 8, false};
		if (llvm::isa<mlir::solidity::AddressType>(_type))
			return PackedType{20, false};
		if (llvm::isa<mlir::solidity::BoolType>(_type))
			return PackedType{1, false};
		if (auto bytesType = llvm::dyn_cast<mlir::solidity::BytesType>(_type))
			return PackedType{bytesType.getSize(), true};
		return std::nullopt;
	}

	/// Arrays use one memory word per element; an aggregate element is itself a
	/// pointer. That representation is recursive, so the ABI implementation can
	/// be recursive as well.
	static bool isABIArray(mlir::Type _type)
	{
		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		return array && isABIEncodable(array.getElementType());
	}

	static bool isABIDynamic(mlir::Type _type)
	{
		if (isDynamic(_type))
			return true;
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
			return isABIArray(_type)
				&& (array.isDynamicallySized() || isABIDynamic(array.getElementType()));
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
			return llvm::any_of(structure.getFieldTypes(), [](mlir::Type field) {
				return isABIDynamic(field);
			});
		return false;
	}

	static bool isABIEncodable(mlir::Type _type)
	{
		if (packedType(_type).has_value() || isDynamic(_type) || isABIArray(_type))
			return true;
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
			return llvm::all_of(structure.getFieldTypes(), [](mlir::Type field) {
				return isABIEncodable(field);
			});
		return false;
	}

	/// Bytes occupied by a value inline in an ABI tuple. Dynamic values occupy
	/// one offset word; a fixed value array occupies one word per element.
	static std::optional<uint64_t> abiHeadSize(mlir::Type _type)
	{
		if (!isABIEncodable(_type))
			return std::nullopt;
		if (isABIDynamic(_type) || packedType(_type))
			return 32;
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
		{
			auto elementSize = abiHeadSize(array.getElementType());
			if (!elementSize)
				return std::nullopt;
			return *elementSize * static_cast<uint64_t>(array.getSize());
		}
		auto structure = llvm::cast<mlir::solidity::StructType>(_type);
		uint64_t total = 0;
		for (mlir::Type field: structure.getFieldTypes())
		{
			auto fieldSize = abiHeadSize(field);
			if (!fieldSize)
				return std::nullopt;
			total += *fieldSize;
		}
		return total;
	}

	static std::optional<uint64_t> abiTupleHeadSize(mlir::TypeRange _types)
	{
		uint64_t total = 0;
		for (mlir::Type type: _types)
		{
			auto size = abiHeadSize(type);
			if (!size)
				return std::nullopt;
			total += *size;
		}
		return total;
	}

	static unsigned storageValueBytes(mlir::Type _type)
	{
		if (auto integer = llvm::dyn_cast<mlir::solidity::UIntType>(_type))
			return std::max(1u, integer.getBitWidth() / 8);
		if (auto integer = llvm::dyn_cast<mlir::solidity::IntType>(_type))
			return std::max(1u, integer.getBitWidth() / 8);
		if (auto bytes = llvm::dyn_cast<mlir::solidity::BytesType>(_type))
			return bytes.getSize();
		if (llvm::isa<mlir::solidity::AddressType>(_type))
			return 20;
		if (llvm::isa<mlir::solidity::BoolType>(_type))
			return 1;
		return 32;
	}

	static uint64_t storageSlotSpan(mlir::Type _type)
	{
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
		{
			if (array.isDynamicallySized())
				return 1;
			unsigned const bytes = storageValueBytes(array.getElementType());
			if (bytes < 32 && !llvm::isa<mlir::solidity::ArrayType, mlir::solidity::StructType>(array.getElementType()))
			{
				uint64_t const perSlot = 32 / bytes;
				return (static_cast<uint64_t>(array.getSize()) + perSlot - 1) / perSlot;
			}
			return static_cast<uint64_t>(array.getSize()) * storageSlotSpan(array.getElementType());
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			if (structure.getFieldSlots().size() == structure.getFieldTypes().size())
			{
				uint64_t slots = 0;
				for (size_t index = 0; index < structure.getFieldTypes().size(); ++index)
					slots = std::max(
						slots,
						static_cast<uint64_t>(structure.getFieldSlots()[index])
							+ storageSlotSpan(structure.getFieldTypes()[index]));
				return slots;
			}
			uint64_t slots = 0;
			for (mlir::Type field: structure.getFieldTypes())
				slots += storageSlotSpan(field);
			return slots;
		}
		return 1;
	}

	static uint64_t structFieldSlot(mlir::solidity::StructType _type, size_t _index)
	{
		if (_type.getFieldSlots().size() == _type.getFieldTypes().size())
			return static_cast<uint64_t>(_type.getFieldSlots()[_index]);
		uint64_t slot = 0;
		for (size_t index = 0; index < _index; ++index)
			slot += storageSlotSpan(_type.getFieldTypes()[index]);
		return slot;
	}

	static unsigned structFieldByteOffset(mlir::solidity::StructType _type, size_t _index)
	{
		return _type.getFieldByteOffsets().size() == _type.getFieldTypes().size()
			? static_cast<unsigned>(_type.getFieldByteOffsets()[_index]) : 0;
	}

	mlir::Value loadStorageField(mlir::Type _type, mlir::Value _slot, unsigned _byteOffset)
	{
		mlir::Value value = m_builder.create<mlir::yul::SLoadOp>(loc(), _slot);
		if (auto packed = packedType(_type))
		{
			if (_byteOffset)
				value = yulShr(
					loc(), wordConstant(static_cast<uint64_t>(_byteOffset * 8)), value);
			if (packed->bytes < 32)
				value = m_builder.create<mlir::yul::AndOp>(
					loc(), value,
					wordConstant(llvm::APInt::getLowBitsSet(256, packed->bytes * 8)));
		}
		return representationConvert(
			value, mlir::solidity::UIntType::get(m_builder.getContext(), 256), _type);
	}

	void storeStorageField(
		mlir::Type _type, mlir::Value _slot, unsigned _byteOffset, mlir::Value _value)
	{
		auto packed = packedType(_type);
		if (!packed || (packed->bytes == 32 && _byteOffset == 0))
		{
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, _value);
			return;
		}
		mlir::Value raw = representationConvert(
			_value, _type, mlir::solidity::UIntType::get(m_builder.getContext(), 256));
		llvm::APInt const mask = llvm::APInt::getLowBitsSet(256, packed->bytes * 8);
		raw = m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(mask));
		unsigned const shiftBits = _byteOffset * 8;
		if (shiftBits)
			raw = yulShl(loc(), wordConstant(shiftBits), raw);
		mlir::Value shiftedMask = shiftBits
			? yulShl(loc(), wordConstant(shiftBits), wordConstant(mask))
			: wordConstant(mask);
		mlir::Value previous = m_builder.create<mlir::yul::SLoadOp>(loc(), _slot);
		mlir::Value retained = m_builder.create<mlir::yul::AndOp>(
			loc(), previous, m_builder.create<mlir::yul::NotOp>(loc(), shiftedMask));
		m_builder.create<mlir::yul::SStoreOp>(
			loc(), _slot, m_builder.create<mlir::yul::OrOp>(loc(), retained, raw));
	}

	/// Reset the memory value rooted at `_pointer` without changing the pointer
	/// stored by its parent. Memory arrays and structs store reference-typed
	/// children as pointers, so deletion recursively clears the pointee while
	/// scalar words are cleared in place.
	void clearMemoryValue(mlir::Type _type, mlir::Value _pointer)
	{
		if (isDynamic(_type))
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), _pointer, wordConstant(uint64_t(0)));
			return;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			for (auto [index, field]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value address = m_builder.create<mlir::yul::AddOp>(
					loc(), _pointer, wordConstant(static_cast<uint64_t>(32 * index)));
				if (needsDefaultMemoryAllocation(field))
					clearMemoryValue(
						field, m_builder.create<mlir::yul::MLoadOp>(loc(), address));
				else
					m_builder.create<mlir::yul::MStoreOp>(loc(), address, wordConstant(uint64_t(0)));
			}
			return;
		}
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
		{
			mlir::Value count = array.isDynamicallySized()
				? m_builder.create<mlir::yul::MLoadOp>(loc(), _pointer).getResult()
				: wordConstant(static_cast<uint64_t>(array.getSize()));
			mlir::Value data = array.isDynamicallySized()
				? m_builder.create<mlir::yul::AddOp>(loc(), _pointer, wordConstant(uint64_t(32))).getResult()
				: _pointer;
			mlir::Type elementType = array.getElementType();
			emitIndexLoop(count, [&](mlir::Value index) {
				mlir::Value address = m_builder.create<mlir::yul::AddOp>(
					loc(), data,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32))));
				if (needsDefaultMemoryAllocation(elementType))
					clearMemoryValue(
						elementType, m_builder.create<mlir::yul::MLoadOp>(loc(), address));
				else
					m_builder.create<mlir::yul::MStoreOp>(loc(), address, wordConstant(uint64_t(0)));
			});
			if (array.isDynamicallySized())
				m_builder.create<mlir::yul::MStoreOp>(loc(), _pointer, wordConstant(uint64_t(0)));
			return;
		}
		m_builder.create<mlir::yul::MStoreOp>(loc(), _pointer, wordConstant(uint64_t(0)));
	}

	/// Clear the storage value rooted at `_slot`. Dynamic children are walked
	/// before their length slot is reset so growing an array after a shrink sees
	/// Solidity's required zero-initialised elements rather than stale data.
	void clearStorageValue(mlir::Type _type, mlir::Value _slot)
	{
		if (isDynamic(_type))
		{
			// Loading first both validates the short/long tag and gives the store
			// helper enough information to clear an old long value's data area.
			m_needsBytesHelpers = m_usesMemory = true;
			mlir::Value pointer = m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{wordType()},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
				mlir::ValueRange{_slot})->getResult(0);
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, wordConstant(uint64_t(0)));
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{_slot, pointer});
			return;
		}
		if (packedType(_type))
		{
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, wordConstant(uint64_t(0)));
			return;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			for (auto [index, field]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value slot = m_builder.create<mlir::yul::AddOp>(
					loc(), _slot, wordConstant(structFieldSlot(structure, index)));
				if (packedType(field))
					storeStorageField(
						field, slot, structFieldByteOffset(structure, index), wordConstant(uint64_t(0)));
				else
					clearStorageValue(field, slot);
			}
			return;
		}
		auto array = llvm::cast<mlir::solidity::ArrayType>(_type);
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::SLoadOp>(loc(), _slot).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value base = array.isDynamicallySized() ? dynamicArrayData(_slot) : _slot;
		uint64_t const stride = storageSlotSpan(array.getElementType());
		emitIndexLoop(count, [&](mlir::Value index) {
			clearStorageValue(
				array.getElementType(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), base,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(stride))));
		});
		if (array.isDynamicallySized())
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, wordConstant(uint64_t(0)));
	}

	/// Deep-copy one storage value to another while retaining both layouts.
	/// Solidity permits fixed arrays whose lengths and elementary packing widths
	/// differ at an assignment boundary.  Materialising the source using the
	/// destination type reads the wrong slots; copying recursively keeps source
	/// and target addressing independent and also clears target-only tails.
	void copyStorageValueToStorage(
		mlir::Type _sourceType,
		mlir::Value _sourceSlot,
		mlir::Type _targetType,
		mlir::Value _targetSlot)
	{
		if (isDynamic(_sourceType) && isDynamic(_targetType))
		{
			mlir::Value value = materializeStorageValue(_sourceType, _sourceSlot);
			storeMemoryValueToStorage(_targetType, _targetSlot, value);
			return;
		}

		auto sourceStruct = llvm::dyn_cast<mlir::solidity::StructType>(_sourceType);
		auto targetStruct = llvm::dyn_cast<mlir::solidity::StructType>(_targetType);
		if (sourceStruct && targetStruct)
		{
			size_t const common = std::min(
				sourceStruct.getFieldTypes().size(), targetStruct.getFieldTypes().size());
			for (size_t index = 0; index < common; ++index)
			{
				mlir::Type sourceField = sourceStruct.getFieldTypes()[index];
				mlir::Type targetField = targetStruct.getFieldTypes()[index];
				mlir::Value sourceSlot = m_builder.create<mlir::yul::AddOp>(
					loc(), _sourceSlot, wordConstant(structFieldSlot(sourceStruct, index)));
				mlir::Value targetSlot = m_builder.create<mlir::yul::AddOp>(
					loc(), _targetSlot, wordConstant(structFieldSlot(targetStruct, index)));
				if (packedType(sourceField) && packedType(targetField))
				{
					mlir::Value value = loadStorageField(
						sourceField, sourceSlot, structFieldByteOffset(sourceStruct, index));
					value = representationConvert(value, sourceField, targetField);
					storeStorageField(
						targetField, targetSlot, structFieldByteOffset(targetStruct, index), value);
				}
				else
					copyStorageValueToStorage(sourceField, sourceSlot, targetField, targetSlot);
			}
			for (size_t index = common; index < targetStruct.getFieldTypes().size(); ++index)
			{
				mlir::Type field = targetStruct.getFieldTypes()[index];
				mlir::Value slot = m_builder.create<mlir::yul::AddOp>(
					loc(), _targetSlot, wordConstant(structFieldSlot(targetStruct, index)));
				if (packedType(field))
					storeStorageField(
						field, slot, structFieldByteOffset(targetStruct, index), wordConstant(uint64_t(0)));
				else
					clearStorageValue(field, slot);
			}
			return;
		}

		auto sourceArray = llvm::dyn_cast<mlir::solidity::ArrayType>(_sourceType);
		auto targetArray = llvm::dyn_cast<mlir::solidity::ArrayType>(_targetType);
		if (sourceArray && targetArray)
		{
			mlir::Value sourceCount = sourceArray.isDynamicallySized()
				? m_builder.create<mlir::yul::SLoadOp>(loc(), _sourceSlot).getResult()
				: wordConstant(static_cast<uint64_t>(sourceArray.getSize()));
			mlir::Value targetCount = targetArray.isDynamicallySized()
				? sourceCount
				: wordConstant(static_cast<uint64_t>(targetArray.getSize()));
			mlir::Value oldTargetCount = targetArray.isDynamicallySized()
				? m_builder.create<mlir::yul::SLoadOp>(loc(), _targetSlot).getResult()
				: targetCount;
			mlir::Value sourceBase = sourceArray.isDynamicallySized()
				? dynamicArrayData(_sourceSlot) : _sourceSlot;
			mlir::Value targetBase = targetArray.isDynamicallySized()
				? dynamicArrayData(_targetSlot) : _targetSlot;
			if (targetArray.isDynamicallySized())
				m_builder.create<mlir::yul::SStoreOp>(loc(), _targetSlot, sourceCount);

			mlir::Type sourceElement = sourceArray.getElementType();
			mlir::Type targetElement = targetArray.getElementType();
			unsigned const sourceBytes = storageValueBytes(sourceElement);
			unsigned const targetBytes = storageValueBytes(targetElement);

			auto copyElement = [&](mlir::Value index) {
				mlir::Value sourceElementSlot = storageArrayElementSlot(
					sourceBase, index, sourceElement, sourceBytes);
				mlir::Value targetElementSlot = storageArrayElementSlot(
					targetBase, index, targetElement, targetBytes);
				if (packedType(sourceElement) && packedType(targetElement))
				{
					mlir::Value value = loadStorageArrayElement(
						sourceElement, sourceElementSlot, index, sourceBytes);
					value = representationConvert(value, sourceElement, targetElement);
					storeStorageArrayElement(
						targetElement, targetElementSlot, index, value, targetBytes);
				}
				else
					copyStorageValueToStorage(
						sourceElement, sourceElementSlot, targetElement, targetElementSlot);
			};
			auto clearElement = [&](mlir::Value index) {
				mlir::Value targetElementSlot = storageArrayElementSlot(
					targetBase, index, targetElement, targetBytes);
				if (packedType(targetElement))
					storeStorageArrayElement(
						targetElement, targetElementSlot, index,
						wordConstant(uint64_t(0)), targetBytes);
				else
					clearStorageValue(targetElement, targetElementSlot);
			};

			emitIndexLoop(targetCount, [&](mlir::Value index) {
				mlir::Value inSource = m_builder.create<mlir::yul::LtOp>(
					loc(), index, sourceCount);
				auto copy = m_builder.create<mlir::yul::IfOp>(loc(), inSource);
				{
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&copy.getThenRegion().emplaceBlock());
					copyElement(index);
				}
				auto clear = m_builder.create<mlir::yul::IfOp>(
					loc(), m_builder.create<mlir::yul::IsZeroOp>(loc(), inSource));
				{
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&clear.getThenRegion().emplaceBlock());
					clearElement(index);
				}
			});

			if (targetArray.isDynamicallySized())
				emitIndexLoop(oldTargetCount, [&](mlir::Value index) {
					auto trailing = m_builder.create<mlir::yul::IfOp>(
						loc(), m_builder.create<mlir::yul::IsZeroOp>(
							loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, sourceCount)));
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&trailing.getThenRegion().emplaceBlock());
					clearElement(index);
				});
			return;
		}

		// Elementary storage values at this level occupy a full slot. Packed
		// array elements are handled by the parent array branch above.
		mlir::Value value = m_builder.create<mlir::yul::SLoadOp>(loc(), _sourceSlot);
		value = representationConvert(value, _sourceType, _targetType);
		m_builder.create<mlir::yul::SStoreOp>(loc(), _targetSlot, value);
	}

	/// Copy the recursive in-memory representation used for calldata and memory
	/// values into storage while preserving the source and destination shapes.
	/// Fixed/dynamic conversions only change the container header and element
	/// count; each aggregate memory element remains a pointer in one word.
	void copyMemoryValueToStorage(
		mlir::Type _sourceType,
		mlir::Value _sourceValue,
		mlir::Type _targetType,
		mlir::Value _targetSlot)
	{
		if (isDynamic(_sourceType) && isDynamic(_targetType))
		{
			m_needsBytesHelpers = m_usesMemory = true;
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{_targetSlot, _sourceValue});
			return;
		}

		auto sourceStruct = llvm::dyn_cast<mlir::solidity::StructType>(_sourceType);
		auto targetStruct = llvm::dyn_cast<mlir::solidity::StructType>(_targetType);
		if (sourceStruct && targetStruct)
		{
			size_t const common = std::min(
				sourceStruct.getFieldTypes().size(), targetStruct.getFieldTypes().size());
			for (size_t index = 0; index < common; ++index)
			{
				mlir::Type targetField = targetStruct.getFieldTypes()[index];
				mlir::Value sourceField = m_builder.create<mlir::yul::MLoadOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), _sourceValue, wordConstant(static_cast<uint64_t>(index * 32))));
				mlir::Value targetSlot = m_builder.create<mlir::yul::AddOp>(
					loc(), _targetSlot, wordConstant(structFieldSlot(targetStruct, index)));
				if (packedType(targetField))
				{
					sourceField = representationConvert(
						sourceField, sourceStruct.getFieldTypes()[index], targetField);
					storeStorageField(
						targetField, targetSlot,
						structFieldByteOffset(targetStruct, index), sourceField);
				}
				else
					copyMemoryValueToStorage(
						sourceStruct.getFieldTypes()[index], sourceField, targetField, targetSlot);
			}
			for (size_t index = common; index < targetStruct.getFieldTypes().size(); ++index)
			{
				mlir::Type field = targetStruct.getFieldTypes()[index];
				mlir::Value slot = m_builder.create<mlir::yul::AddOp>(
					loc(), _targetSlot, wordConstant(structFieldSlot(targetStruct, index)));
				if (packedType(field))
					storeStorageField(
						field, slot, structFieldByteOffset(targetStruct, index), wordConstant(uint64_t(0)));
				else
					clearStorageValue(field, slot);
			}
			return;
		}

		auto sourceArray = llvm::dyn_cast<mlir::solidity::ArrayType>(_sourceType);
		auto targetArray = llvm::dyn_cast<mlir::solidity::ArrayType>(_targetType);
		if (sourceArray && targetArray)
		{
			mlir::Value sourceCount = sourceArray.isDynamicallySized()
				? m_builder.create<mlir::yul::MLoadOp>(loc(), _sourceValue).getResult()
				: wordConstant(static_cast<uint64_t>(sourceArray.getSize()));
			mlir::Value sourceBase = sourceArray.isDynamicallySized()
				? m_builder.create<mlir::yul::AddOp>(
					loc(), _sourceValue, wordConstant(uint64_t(32))).getResult()
				: _sourceValue;
			mlir::Value targetCount = targetArray.isDynamicallySized()
				? sourceCount
				: wordConstant(static_cast<uint64_t>(targetArray.getSize()));
			mlir::Value oldTargetCount = targetArray.isDynamicallySized()
				? m_builder.create<mlir::yul::SLoadOp>(loc(), _targetSlot).getResult()
				: targetCount;
			mlir::Value targetBase = targetArray.isDynamicallySized()
				? dynamicArrayData(_targetSlot) : _targetSlot;
			if (targetArray.isDynamicallySized())
				m_builder.create<mlir::yul::SStoreOp>(loc(), _targetSlot, sourceCount);

			mlir::Type sourceElement = sourceArray.getElementType();
			mlir::Type targetElement = targetArray.getElementType();
			unsigned const targetBytes = storageValueBytes(targetElement);

			auto copyElement = [&](mlir::Value index) {
				mlir::Value sourceElementValue = m_builder.create<mlir::yul::MLoadOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), sourceBase,
						m_builder.create<mlir::yul::MulOp>(
							loc(), index, wordConstant(uint64_t(32)))));
				mlir::Value targetElementSlot = storageArrayElementSlot(
					targetBase, index, targetElement, targetBytes);
				if (packedType(targetElement))
				{
					sourceElementValue = representationConvert(
						sourceElementValue, sourceElement, targetElement);
					storeStorageArrayElement(
						targetElement, targetElementSlot, index,
						sourceElementValue, targetBytes);
				}
				else
					copyMemoryValueToStorage(
						sourceElement, sourceElementValue, targetElement, targetElementSlot);
			};
			auto clearElement = [&](mlir::Value index) {
				mlir::Value targetElementSlot = storageArrayElementSlot(
					targetBase, index, targetElement, targetBytes);
				if (packedType(targetElement))
					storeStorageArrayElement(
						targetElement, targetElementSlot, index,
						wordConstant(uint64_t(0)), targetBytes);
				else
					clearStorageValue(targetElement, targetElementSlot);
			};

			emitIndexLoop(targetCount, [&](mlir::Value index) {
				mlir::Value inSource = m_builder.create<mlir::yul::LtOp>(
					loc(), index, sourceCount);
				auto copy = m_builder.create<mlir::yul::IfOp>(loc(), inSource);
				{
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&copy.getThenRegion().emplaceBlock());
					copyElement(index);
				}
				auto clear = m_builder.create<mlir::yul::IfOp>(
					loc(), m_builder.create<mlir::yul::IsZeroOp>(loc(), inSource));
				{
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&clear.getThenRegion().emplaceBlock());
					clearElement(index);
				}
			});
			if (targetArray.isDynamicallySized())
				emitIndexLoop(oldTargetCount, [&](mlir::Value index) {
					auto trailing = m_builder.create<mlir::yul::IfOp>(
						loc(), m_builder.create<mlir::yul::IsZeroOp>(
							loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, sourceCount)));
					mlir::OpBuilder::InsertionGuard guard(m_builder);
					m_builder.setInsertionPointToStart(&trailing.getThenRegion().emplaceBlock());
					clearElement(index);
				});
			return;
		}

		mlir::Value value = representationConvert(_sourceValue, _sourceType, _targetType);
		m_builder.create<mlir::yul::SStoreOp>(loc(), _targetSlot, value);
	}

	/// Deep-copy the internal memory representation of a value into Solidity
	/// storage. Dynamic arrays own a length slot and keccak-derived data area;
	/// aggregate memory elements are pointers held one per word.
	void storeMemoryValueToStorage(mlir::Type _type, mlir::Value _slot, mlir::Value _value)
	{
		if (isDynamic(_type))
		{
			m_needsBytesHelpers = m_usesMemory = true;
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kStoreBytesHelper),
				mlir::ValueRange{_slot, _value});
			return;
		}
		if (packedType(_type))
		{
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, _value);
			return;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			uint64_t fieldSlot = 0;
			for (auto [index, field]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value fieldValue = m_builder.create<mlir::yul::MLoadOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(
						loc(), _value, wordConstant(uint64_t(index * 32))));
				storeMemoryValueToStorage(
					field,
					m_builder.create<mlir::yul::AddOp>(loc(), _slot, wordConstant(fieldSlot)),
					fieldValue);
				fieldSlot += storageSlotSpan(field);
			}
			return;
		}

		auto array = llvm::cast<mlir::solidity::ArrayType>(_type);
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value source = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		mlir::Value oldCount = array.isDynamicallySized()
			? m_builder.create<mlir::yul::SLoadOp>(loc(), _slot).getResult()
			: count;
		mlir::Value base = array.isDynamicallySized() ? dynamicArrayData(_slot) : _slot;
		if (array.isDynamicallySized())
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, count);
		mlir::Type elementType = array.getElementType();
		unsigned const elementByteWidth = storageValueBytes(elementType);
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value element = m_builder.create<mlir::yul::MLoadOp>(
				loc(), m_builder.create<mlir::yul::AddOp>(
					loc(), source,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32)))));
			mlir::Value elementSlot = storageArrayElementSlot(
				base, index, elementType, elementByteWidth);
			if (packedType(elementType))
				storeStorageArrayElement(
					elementType, elementSlot, index, element, elementByteWidth);
			else
				storeMemoryValueToStorage(elementType, elementSlot, element);
		});
		if (array.isDynamicallySized())
			emitIndexLoop(oldCount, [&](mlir::Value index) {
				auto trailing = m_builder.create<mlir::yul::IfOp>(
					loc(), m_builder.create<mlir::yul::IsZeroOp>(
						loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, count)));
				mlir::OpBuilder::InsertionGuard guard(m_builder);
				m_builder.setInsertionPointToStart(&trailing.getThenRegion().emplaceBlock());
				mlir::Value elementSlot = storageArrayElementSlot(
					base, index, elementType, elementByteWidth);
				if (packedType(elementType))
					storeStorageArrayElement(
						elementType, elementSlot, index, wordConstant(uint64_t(0)), elementByteWidth);
				else
					clearStorageValue(elementType, elementSlot);
			});
	}

	/// Turn a storage reference/base slot into the recursive memory shape used
	/// by the ABI encoder. Aggregate operations carry an explicit storage_args
	/// marker from the AST generator, so a memory pointer is never guessed to be
	/// a slot merely because it is word-typed.
	mlir::Value materializeStorageValue(mlir::Type _type, mlir::Value _slot)
	{
		m_usesMemory = true;
		if (isDynamic(_type))
		{
			m_needsBytesHelpers = true;
			return m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{wordType()},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kLoadBytesHelper),
				mlir::ValueRange{_slot}).getResult(0);
		}
		if (packedType(_type))
			return m_builder.create<mlir::yul::SLoadOp>(loc(), _slot);
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			mlir::Value pointer = allocate(
				wordConstant(uint64_t(32 * structure.getFieldTypes().size())));
			for (auto [index, fieldType]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value fieldSlot = m_builder.create<mlir::yul::AddOp>(
					loc(), _slot, wordConstant(structFieldSlot(structure, index)));
				mlir::Value field = packedType(fieldType)
					? loadStorageField(
						fieldType, fieldSlot, structFieldByteOffset(structure, index))
					: materializeStorageValue(fieldType, fieldSlot);
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(uint64_t(32 * index))),
					field);
			}
			return pointer;
		}
		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array)
			fail("storage ABI value has no materialisable shape");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::SLoadOp>(loc(), _slot).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value bytes = m_builder.create<mlir::yul::MulOp>(
			loc(), count, wordConstant(uint64_t(32)));
		mlir::Value pointer = allocate(array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), bytes).getResult()
			: bytes);
		mlir::Value data = pointer;
		if (array.isDynamicallySized())
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, count);
			data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		}
		mlir::Value storageData = array.isDynamicallySized() ? dynamicArrayData(_slot) : _slot;
		mlir::Type elementType = array.getElementType();
		unsigned const elementByteWidth = storageValueBytes(elementType);
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value elementSlot = storageArrayElementSlot(
				storageData, index, elementType, elementByteWidth);
			mlir::Value element = packedType(elementType)
				? loadStorageArrayElement(elementType, elementSlot, index, elementByteWidth)
				: materializeStorageValue(elementType, elementSlot);
			m_builder.create<mlir::yul::MStoreOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), data,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32)))),
				element);
		});
		return pointer;
	}

	mlir::Value abiOperand(
		mlir::Operation* _operation,
		unsigned _index,
		mlir::Type _type,
		mlir::Value _value)
	{
		return _operation && indexedAttr(_operation, "storage_args", _index)
			? materializeStorageValue(_type, _value) : _value;
	}

	void emitErrorStringRevertIf(mlir::Value _condition, std::string const& _message)
	{
		auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), _condition);
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());

		llvm::APInt selector(256, 0x08c379a0u);
		selector <<= 224;
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), wordConstant(uint64_t(0)), wordConstant(selector));
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), wordConstant(uint64_t(4)), wordConstant(uint64_t(32)));
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), wordConstant(uint64_t(36)), wordConstant(static_cast<uint64_t>(_message.size())));
		for (size_t offset = 0; offset < _message.size(); offset += 32)
		{
			llvm::APInt word(256, 0);
			for (size_t index = 0; index < 32 && offset + index < _message.size(); ++index)
			{
				llvm::APInt byte(256, static_cast<unsigned char>(_message[offset + index]));
				word |= byte << static_cast<unsigned>((31 - index) * 8);
			}
			m_builder.create<mlir::yul::MStoreOp>(
				loc(), wordConstant(static_cast<uint64_t>(68 + offset)), wordConstant(word));
		}
		uint64_t const paddedLength = (_message.size() + 31) / 32 * 32;
		m_builder.create<mlir::yul::RevertOp>(
			loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(68 + paddedLength)));
	}

	void emitABIBoundsCheck(
		mlir::Value _offset,
		mlir::Value _size,
		mlir::Value _total,
		std::string const& _message = {})
	{
		mlir::Value end = m_builder.create<mlir::yul::AddOp>(loc(), _offset, _size);
		mlir::Value overflow = m_builder.create<mlir::yul::LtOp>(loc(), end, _offset);
		mlir::Value pastEnd = m_builder.create<mlir::yul::GtOp>(loc(), end, _total);
		mlir::Value invalid = m_builder.create<mlir::yul::OrOp>(loc(), overflow, pastEnd);
		if (_message.empty())
			emitRevertIf(invalid);
		else
			emitErrorStringRevertIf(invalid, _message);
	}

	void emitABIMultiplicationCheck(
		mlir::Value _count,
		uint64_t _stride,
		mlir::Value _product,
		std::string const& _message = {})
	{
		if (_stride == 0)
			return;
		mlir::Value recovered = m_builder.create<mlir::yul::DivOp>(
			loc(), _product, wordConstant(_stride));
		mlir::Value invalid = m_builder.create<mlir::yul::IsZeroOp>(
			loc(), m_builder.create<mlir::yul::EqOp>(loc(), recovered, _count));
		if (_message.empty())
			emitRevertIf(invalid);
		else
			emitErrorStringRevertIf(invalid, _message);
	}

	enum class ABIDecodeContext
	{
		Generic,
		Calldata,
		Memory
	};

	std::string debugMessage(llvm::StringRef _message) const
	{
		return m_debugRevertStrings ? _message.str() : std::string{};
	}

	std::string abiHeadMessage(ABIDecodeContext _context) const
	{
		if (_context == ABIDecodeContext::Calldata)
			return debugMessage("ABI calldata decoding: invalid head pointer");
		if (_context == ABIDecodeContext::Memory)
			return debugMessage("ABI memory decoding: invalid data start");
		return debugMessage("Calldata too short");
	}

	std::string abiDataMessage(ABIDecodeContext _context) const
	{
		if (_context == ABIDecodeContext::Calldata)
			return debugMessage("ABI calldata decoding: invalid data pointer");
		if (_context == ABIDecodeContext::Memory)
			return debugMessage("ABI memory decoding: invalid data length");
		return debugMessage("Calldata too short");
	}

	/// Decode the standalone ABI representation beginning at `_encoded`. The
	/// returned aggregate uses Solidity's internal memory shape: dynamic arrays
	/// and byte strings have a length word, fixed arrays do not, and aggregate
	/// elements are pointers stored one per word.
	mlir::Value decodeABIValue(
		mlir::Type _type,
		mlir::Value _encoded,
		mlir::Value _available,
		ABIDecodeContext _context = ABIDecodeContext::Generic)
	{
		if (packedType(_type))
		{
			emitABIBoundsCheck(
				wordConstant(uint64_t(0)), wordConstant(uint64_t(32)), _available,
				abiDataMessage(_context));
			mlir::Value value = m_builder.create<mlir::yul::MLoadOp>(loc(), _encoded);
			// ABI words may carry dirty high bits for integer-like values or dirty
			// low bits for bytesN. High-level parameters always enter in their
			// declared representation, including with ABI coder v1.
			if (llvm::isa<mlir::solidity::BytesType>(_type))
				return representationConvert(value, _type, _type);
			return representationConvert(
				value, mlir::solidity::UIntType::get(m_builder.getContext(), 256), _type);
		}
		if (isDynamic(_type))
		{
			emitABIBoundsCheck(
				wordConstant(uint64_t(0)), wordConstant(uint64_t(32)), _available,
				abiDataMessage(_context));
			mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), _encoded);
			emitABIBoundsCheck(
				wordConstant(uint64_t(32)), length, _available, abiDataMessage(_context));
			mlir::Value pointer = allocate(m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)), roundedUp(length)));
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::AddOp>(loc(), _encoded, wordConstant(uint64_t(32))),
				length);
			return pointer;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			auto headSize = abiTupleHeadSize(structure.getFieldTypes());
			if (!headSize)
				fail("abi.decode of a non-elementary component");
			emitABIBoundsCheck(
				wordConstant(uint64_t(0)), wordConstant(*headSize), _available,
				abiDataMessage(_context));
			mlir::Value pointer = allocate(
				wordConstant(uint64_t(32 * structure.getFieldTypes().size())));
			uint64_t headOffset = 0;
			for (auto [index, fieldType]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value field = decodeABIComponent(
					fieldType,
					m_builder.create<mlir::yul::AddOp>(loc(), _encoded, wordConstant(headOffset)),
					_encoded, _available, _context);
				m_builder.create<mlir::yul::MStoreOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(
						loc(), pointer, wordConstant(uint64_t(32 * index))),
					field);
				headOffset += *abiHeadSize(fieldType);
			}
			return pointer;
		}

		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array || !isABIArray(_type))
			fail("abi.decode of a non-elementary component");
		mlir::Type elementType = array.getElementType();
		auto elementHeadSize = abiHeadSize(elementType);
		if (!elementHeadSize)
			fail("abi.decode of a non-elementary component");
		mlir::Value prefix = array.isDynamicallySized()
			? wordConstant(uint64_t(32)) : wordConstant(uint64_t(0));
		if (array.isDynamicallySized())
			emitABIBoundsCheck(
				wordConstant(uint64_t(0)), wordConstant(uint64_t(32)), _available,
				abiDataMessage(_context));
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _encoded).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value encodedHeads = m_builder.create<mlir::yul::MulOp>(
			loc(), count, wordConstant(*elementHeadSize));
		emitABIMultiplicationCheck(
			count, *elementHeadSize, encodedHeads,
			abiDataMessage(_context));
		emitABIBoundsCheck(
			prefix, encodedHeads, _available,
			abiDataMessage(_context));

		mlir::Value internalBytes = m_builder.create<mlir::yul::MulOp>(
			loc(), count, wordConstant(uint64_t(32)));
		emitABIMultiplicationCheck(count, 32, internalBytes);
		mlir::Value allocationBytes = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), internalBytes).getResult()
			: internalBytes;
		mlir::Value pointer = allocate(allocationBytes);
		mlir::Value internalData = pointer;
		if (array.isDynamicallySized())
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, count);
			internalData = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		}
		mlir::Value tuple = m_builder.create<mlir::yul::AddOp>(loc(), _encoded, prefix);
		mlir::Value tupleAvailable = m_builder.create<mlir::yul::SubOp>(loc(), _available, prefix);
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value headOffset = m_builder.create<mlir::yul::MulOp>(
				loc(), index, wordConstant(*elementHeadSize));
			mlir::Value head = m_builder.create<mlir::yul::AddOp>(loc(), tuple, headOffset);
			mlir::Value element;
			if (isABIDynamic(elementType))
			{
				mlir::Value offset = m_builder.create<mlir::yul::MLoadOp>(loc(), head);
				mlir::Value regularOffset = m_builder.create<mlir::yul::IsZeroOp>(
					loc(), m_builder.create<mlir::yul::GtOp>(loc(), offset, tupleAvailable));
				mlir::Value validOffset = regularOffset;
				// The legacy external decoder accepts exactly -32 for an element
				// of a dynamic outer array, which addresses the enclosing length
				// word. Do not generalise that compatibility quirk to arbitrary
				// wrapped pointers: abi.decode overflow probes rely on rejecting
				// those before they can escape the supplied byte buffer.
				if (array.isDynamicallySized())
				{
					llvm::APInt minusWord(256, 0);
					minusWord -= 32;
					validOffset = m_builder.create<mlir::yul::OrOp>(
						loc(), validOffset,
						m_builder.create<mlir::yul::EqOp>(loc(), offset, wordConstant(minusWord)));
				}
				mlir::Value invalidOffset = m_builder.create<mlir::yul::IsZeroOp>(loc(), validOffset);
				if (std::string message = abiDataMessage(_context); !message.empty())
					emitErrorStringRevertIf(invalidOffset, message);
				else
					emitRevertIf(invalidOffset);
				element = decodeABIValue(
					elementType,
					m_builder.create<mlir::yul::AddOp>(loc(), tuple, offset),
					m_builder.create<mlir::yul::SubOp>(loc(), tupleAvailable, offset),
					_context);
			}
			else
				element = decodeABIValue(
					elementType, head,
					m_builder.create<mlir::yul::SubOp>(loc(), tupleAvailable, headOffset),
					_context);
			m_builder.create<mlir::yul::MStoreOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), internalData,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32)))),
				element);
		});
		return pointer;
	}

	mlir::Value decodeABIComponent(
		mlir::Type _type,
		mlir::Value _head,
		mlir::Value _tuple,
		mlir::Value _tupleLength,
		ABIDecodeContext _context = ABIDecodeContext::Generic)
	{
		if (isABIDynamic(_type))
		{
			mlir::Value offset = m_builder.create<mlir::yul::MLoadOp>(loc(), _head);
			emitABIBoundsCheck(
				offset, wordConstant(uint64_t(32)), _tupleLength, abiHeadMessage(_context));
			return decodeABIValue(
				_type,
				m_builder.create<mlir::yul::AddOp>(loc(), _tuple, offset),
				m_builder.create<mlir::yul::SubOp>(loc(), _tupleLength, offset),
				_context);
		}
		mlir::Value headOffset = m_builder.create<mlir::yul::SubOp>(loc(), _head, _tuple);
		emitABIBoundsCheck(
			headOffset, wordConstant(*abiHeadSize(_type)), _tupleLength, abiHeadMessage(_context));
		return decodeABIValue(
			_type, _head,
			m_builder.create<mlir::yul::SubOp>(loc(), _tupleLength, headOffset),
			_context);
	}

	/// External function values occupy bytes24 in the ABI (left-aligned in the
	/// word) but use address<<32|selector internally. The ordinary ABI marker
	/// handles a scalar parameter; arrays need the same normalization applied
	/// recursively to their decoded memory elements.
	mlir::Value normalizeExternalFunctionArray(mlir::Type _type, mlir::Value _value)
	{
		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array)
			fail("external-function array marker applied to a non-array parameter");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value data = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value address = m_builder.create<mlir::yul::AddOp>(
				loc(), data,
				m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32))));
			mlir::Value element = m_builder.create<mlir::yul::MLoadOp>(loc(), address);
			if (llvm::isa<mlir::solidity::ArrayType>(array.getElementType()))
				element = normalizeExternalFunctionArray(array.getElementType(), element);
			else
				element = yulShr(
					loc(), wordConstant(uint64_t(64)), element);
			m_builder.create<mlir::yul::MStoreOp>(loc(), address, element);
		});
		return _value;
	}

	void validateEnumArray(mlir::Type _type, mlir::Value _value, uint64_t _limit)
	{
		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array)
			fail("enum-array metadata applied to a non-array parameter");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value data = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value address = m_builder.create<mlir::yul::AddOp>(
				loc(), data,
				m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32))));
			mlir::Value invalid = isZero(m_builder.create<mlir::yul::LtOp>(
				loc(), m_builder.create<mlir::yul::MLoadOp>(loc(), address), wordConstant(_limit)));
			if (m_debugRevertStrings)
				emitErrorStringRevertIf(invalid, "Enum out of range");
			else
				emitRevertIf(invalid);
		});
	}

	/// Materialise a private ABI-facing copy of an external-function array.
	/// Internal values are address<<32|selector, whereas ABI bytes24 values are
	/// left-aligned by another 64 bits.  Do not rewrite the caller's memory: a
	/// Solidity expression may legally observe the source array after the call.
	mlir::Value denormalizeExternalFunctionArray(mlir::Type _type, mlir::Value _value)
	{
		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array)
			fail("external-function array marker applied to a non-array argument");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value bytes = m_builder.create<mlir::yul::MulOp>(
			loc(), count, wordConstant(uint64_t(32)));
		if (array.isDynamicallySized())
			bytes = m_builder.create<mlir::yul::AddOp>(loc(), bytes, wordConstant(uint64_t(32)));
		mlir::Value result = allocate(bytes);
		if (array.isDynamicallySized())
			m_builder.create<mlir::yul::MStoreOp>(loc(), result, count);
		mlir::Value source = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		mlir::Value destination = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), result, wordConstant(uint64_t(32))).getResult()
			: result;
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value byteOffset = m_builder.create<mlir::yul::MulOp>(
				loc(), index, wordConstant(uint64_t(32)));
			mlir::Value element = m_builder.create<mlir::yul::MLoadOp>(
				loc(), m_builder.create<mlir::yul::AddOp>(loc(), source, byteOffset));
			if (llvm::isa<mlir::solidity::ArrayType>(array.getElementType()))
				element = denormalizeExternalFunctionArray(array.getElementType(), element);
			else
				element = yulShl(
					loc(), wordConstant(uint64_t(64)), element);
			m_builder.create<mlir::yul::MStoreOp>(
				loc(), m_builder.create<mlir::yul::AddOp>(loc(), destination, byteOffset), element);
		});
		return result;
	}

	mlir::Value abiEncodedSize(mlir::Type _type, mlir::Value _value)
	{
		if (packedType(_type))
			return wordConstant(uint64_t(32));
		if (isDynamic(_type))
			return m_builder.create<mlir::yul::AddOp>(
				loc(), wordConstant(uint64_t(32)),
				roundedUp(m_builder.create<mlir::yul::MLoadOp>(loc(), _value)));
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			auto headSize = abiTupleHeadSize(structure.getFieldTypes());
			if (!headSize)
				fail("abi.encode of a non-elementary argument");
			mlir::Value total = wordConstant(*headSize);
			for (auto [index, fieldType]: llvm::enumerate(structure.getFieldTypes()))
				if (isABIDynamic(fieldType))
				{
					mlir::Value field = m_builder.create<mlir::yul::MLoadOp>(
						loc(),
						m_builder.create<mlir::yul::AddOp>(
							loc(), _value, wordConstant(uint64_t(32 * index))));
					total = m_builder.create<mlir::yul::AddOp>(
						loc(), total, abiEncodedSize(fieldType, field));
				}
			return total;
		}

		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array || !isABIArray(_type))
			fail("abi.encode of a non-elementary argument");
		mlir::Type elementType = array.getElementType();
		auto elementHeadSize = abiHeadSize(elementType);
		if (!elementHeadSize)
			fail("abi.encode of a non-elementary argument");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value total = m_builder.create<mlir::yul::AddOp>(
			loc(),
			array.isDynamicallySized() ? wordConstant(uint64_t(32)) : wordConstant(uint64_t(0)),
			m_builder.create<mlir::yul::MulOp>(loc(), count, wordConstant(*elementHeadSize)));
		if (!isABIDynamic(elementType))
			return total;

		auto totalVar = m_builder.create<mlir::yul::VarOp>(loc(), total);
		mlir::Value source = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value element = m_builder.create<mlir::yul::MLoadOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), source,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32)))));
			mlir::Value oldTotal = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), totalVar);
			m_builder.create<mlir::yul::AssignOp>(
				loc(), totalVar,
				m_builder.create<mlir::yul::AddOp>(loc(), oldTotal, abiEncodedSize(elementType, element)));
		});
		return m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), totalVar);
	}

	/// Encode one value at the beginning of its standalone ABI representation.
	/// For an array this is either `[length][element tuple]` (dynamic array) or
	/// just the element tuple (fixed array). Offsets inside either tuple are
	/// relative to the first element head, as required by the ABI.
	void encodeABIValue(mlir::Type _type, mlir::Value _value, mlir::Value _destination)
	{
		if (packedType(_type))
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), _destination, _value);
			return;
		}
		if (isDynamic(_type))
		{
			mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), _value);
			m_builder.create<mlir::yul::MStoreOp>(loc(), _destination, length);
			yulMCopy(
				loc(),
				m_builder.create<mlir::yul::AddOp>(loc(), _destination, wordConstant(uint64_t(32))),
				m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))),
				length);
			return;
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			auto headSize = abiTupleHeadSize(structure.getFieldTypes());
			if (!headSize)
				fail("abi.encode of a non-elementary argument");
			mlir::Value tail = wordConstant(*headSize);
			uint64_t headOffset = 0;
			for (auto [index, fieldType]: llvm::enumerate(structure.getFieldTypes()))
			{
				mlir::Value field = m_builder.create<mlir::yul::MLoadOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(
						loc(), _value, wordConstant(uint64_t(32 * index))));
				mlir::Value head = m_builder.create<mlir::yul::AddOp>(
					loc(), _destination, wordConstant(headOffset));
				if (!isABIDynamic(fieldType))
					encodeABIValue(fieldType, field, head);
				else
				{
					m_builder.create<mlir::yul::MStoreOp>(loc(), head, tail);
					encodeABIValue(
						fieldType, field,
						m_builder.create<mlir::yul::AddOp>(loc(), _destination, tail));
					tail = m_builder.create<mlir::yul::AddOp>(
						loc(), tail, abiEncodedSize(fieldType, field));
				}
				headOffset += *abiHeadSize(fieldType);
			}
			return;
		}

		auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type);
		if (!array || !isABIArray(_type))
			fail("abi.encode of a non-elementary argument");
		mlir::Type elementType = array.getElementType();
		auto elementHeadSize = abiHeadSize(elementType);
		if (!elementHeadSize)
			fail("abi.encode of a non-elementary argument");
		mlir::Value count = array.isDynamicallySized()
			? m_builder.create<mlir::yul::MLoadOp>(loc(), _value).getResult()
			: wordConstant(static_cast<uint64_t>(array.getSize()));
		mlir::Value source = array.isDynamicallySized()
			? m_builder.create<mlir::yul::AddOp>(loc(), _value, wordConstant(uint64_t(32))).getResult()
			: _value;
		mlir::Value tuple = _destination;
		if (array.isDynamicallySized())
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), _destination, count);
			tuple = m_builder.create<mlir::yul::AddOp>(loc(), _destination, wordConstant(uint64_t(32)));
		}
		mlir::Value initialTail = m_builder.create<mlir::yul::MulOp>(
			loc(), count, wordConstant(*elementHeadSize));
		auto tailVar = m_builder.create<mlir::yul::VarOp>(loc(), initialTail);

		emitIndexLoop(count, [&](mlir::Value index) {
			mlir::Value element = m_builder.create<mlir::yul::MLoadOp>(
				loc(),
				m_builder.create<mlir::yul::AddOp>(
					loc(), source,
					m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(uint64_t(32)))));
			mlir::Value head = m_builder.create<mlir::yul::AddOp>(
				loc(), tuple,
				m_builder.create<mlir::yul::MulOp>(loc(), index, wordConstant(*elementHeadSize)));
			if (!isABIDynamic(elementType))
			{
				encodeABIValue(elementType, element, head);
				return;
			}
			mlir::Value tail = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), tailVar);
			m_builder.create<mlir::yul::MStoreOp>(loc(), head, tail);
			encodeABIValue(
				elementType, element,
				m_builder.create<mlir::yul::AddOp>(loc(), tuple, tail));
			m_builder.create<mlir::yul::AssignOp>(
				loc(), tailVar,
				m_builder.create<mlir::yul::AddOp>(loc(), tail, abiEncodedSize(elementType, element)));
		});
	}

	/// Standard ABI encoding as a Solidity memory byte array. The array's data
	/// is the ABI head followed by dynamic tails; the memory-array length word
	/// itself is not part of the ABI payload. `_values` are already lowered Yul
	/// words: scalars are values and memory aggregates are pointers.
	mlir::Value encodeABIValues(mlir::TypeRange _types, mlir::ValueRange _values)
	{
		if (_types.size() != _values.size())
			fail("ABI encoder type/value arity mismatch");
		auto headSize = abiTupleHeadSize(_types);
		if (!headSize)
			fail("abi.encode of a non-elementary argument");

		m_usesMemory = true;
		mlir::Value total = wordConstant(*headSize);
		for (auto [type, value]: llvm::zip(_types, _values))
			if (isABIDynamic(type))
				total = m_builder.create<mlir::yul::AddOp>(
					loc(), total, abiEncodedSize(type, value));

		mlir::Value pointer = allocate(
			m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), total));
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, total);
		mlir::Value data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		mlir::Value tail = wordConstant(*headSize);
		uint64_t headOffset = 0;

		for (auto [type, value]: llvm::zip(_types, _values))
		{
			mlir::Value head = m_builder.create<mlir::yul::AddOp>(
				loc(), data, wordConstant(headOffset));
			if (!isABIDynamic(type))
			{
				encodeABIValue(type, value, head);
				headOffset += *abiHeadSize(type);
				continue;
			}

			m_builder.create<mlir::yul::MStoreOp>(loc(), head, tail);
			mlir::Value destination = m_builder.create<mlir::yul::AddOp>(loc(), data, tail);
			encodeABIValue(type, value, destination);
			tail = m_builder.create<mlir::yul::AddOp>(
				loc(), tail, abiEncodedSize(type, value));
			headOffset += 32;
		}
		return pointer;
	}

	mlir::Value encodeABI(
		mlir::ValueRange _arguments,
		mlir::Operation* _operation = nullptr,
		unsigned _indexBase = 0)
	{
		llvm::SmallVector<mlir::Type, 4> types;
		llvm::SmallVector<mlir::Value, 4> values;
		for (auto [index, argument]: llvm::enumerate(_arguments))
		{
			types.push_back(argument.getType());
			values.push_back(abiOperand(
				_operation, static_cast<unsigned>(index) + _indexBase,
				argument.getType(), mapped(argument)));
		}
		return encodeABIValues(types, values);
	}

	/// ABI encoding prefixed with the high four bytes of `_selector`. A bytes4
	/// value and a keccak256 hash are both already left-aligned, so one mstore
	/// writes the selector in exactly the position the ABI requires.
	mlir::Value encodeABIWithSelector(
		mlir::Value _selector,
		mlir::ValueRange _arguments,
		mlir::Operation* _operation = nullptr,
		unsigned _indexBase = 0)
	{
		m_usesMemory = true;
		mlir::Value encoded = encodeABI(_arguments, _operation, _indexBase);
		mlir::Value encodedLength = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
		mlir::Value length = m_builder.create<mlir::yul::AddOp>(
			loc(), wordConstant(uint64_t(4)), encodedLength);
		mlir::Value pointer = allocate(
			m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(length)));
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, length);
		mlir::Value data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		m_builder.create<mlir::yul::MStoreOp>(loc(), data, _selector);
		yulMCopy(
			loc(),
			m_builder.create<mlir::yul::AddOp>(loc(), data, wordConstant(uint64_t(4))),
			m_builder.create<mlir::yul::AddOp>(loc(), encoded, wordConstant(uint64_t(32))),
			encodedLength);
		return pointer;
	}

	/// ABI packed encoding as a Solidity memory byte array: [length][data...].
	/// Elementary values are written byte-exactly and dynamic bytes/string are
	/// copied without their length word or padding.
	mlir::Value encodePacked(mlir::ValueRange _arguments, mlir::Operation* _operation = nullptr)
	{
		m_usesMemory = true;
		llvm::SmallVector<mlir::Value, 4> values;
		for (auto [index, argument]: llvm::enumerate(_arguments))
			values.push_back(abiOperand(
				_operation, index, argument.getType(), mapped(argument)));
		mlir::Value total = wordConstant(uint64_t(0));
		for (auto [argument, value]: llvm::zip(_arguments, values))
			if (isDynamic(argument.getType()))
				total = m_builder.create<mlir::yul::AddOp>(
					loc(), total, m_builder.create<mlir::yul::MLoadOp>(loc(), value));
			else if (std::optional<PackedType> type = packedType(argument.getType()))
				total = m_builder.create<mlir::yul::AddOp>(loc(), total, wordConstant(uint64_t(type->bytes)));
			else if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(argument.getType());
				array && packedType(array.getElementType()))
			{
				mlir::Value count = array.isDynamicallySized()
					? m_builder.create<mlir::yul::MLoadOp>(loc(), value).getResult()
					: wordConstant(static_cast<uint64_t>(array.getSize()));
				total = m_builder.create<mlir::yul::AddOp>(
					loc(), total,
					m_builder.create<mlir::yul::MulOp>(loc(), count, wordConstant(uint64_t(32))));
			}
			else
				fail("abi.encodePacked of a non-elementary argument");

		mlir::Value pointer = allocate(
			m_builder.create<mlir::yul::AddOp>(loc(), wordConstant(uint64_t(32)), roundedUp(total)));
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, total);
		mlir::Value data = m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32)));
		mlir::Value offset = wordConstant(uint64_t(0));

		for (auto [argument, value]: llvm::zip(_arguments, values))
		{
			if (isDynamic(argument.getType()))
			{
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), value);
				mlir::Value source = m_builder.create<mlir::yul::AddOp>(loc(), value, wordConstant(uint64_t(32)));
				emitWordLoop(length, [&](mlir::Value _index) {
					m_builder.create<mlir::yul::MStoreOp>(
						loc(),
						m_builder.create<mlir::yul::AddOp>(
							loc(), data, m_builder.create<mlir::yul::AddOp>(loc(), offset, _index)),
						m_builder.create<mlir::yul::MLoadOp>(
							loc(), m_builder.create<mlir::yul::AddOp>(loc(), source, _index)));
				});
				offset = m_builder.create<mlir::yul::AddOp>(loc(), offset, length);
				continue;
			}
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(argument.getType()))
			{
				mlir::Value count = array.isDynamicallySized()
					? m_builder.create<mlir::yul::MLoadOp>(loc(), value).getResult()
					: wordConstant(static_cast<uint64_t>(array.getSize()));
				mlir::Value source = array.isDynamicallySized()
					? m_builder.create<mlir::yul::AddOp>(loc(), value, wordConstant(uint64_t(32))).getResult()
					: value;
				emitIndexLoop(count, [&](mlir::Value index) {
					mlir::Value byteOffset = m_builder.create<mlir::yul::MulOp>(
						loc(), index, wordConstant(uint64_t(32)));
					m_builder.create<mlir::yul::MStoreOp>(
						loc(),
						m_builder.create<mlir::yul::AddOp>(
							loc(), data,
							m_builder.create<mlir::yul::AddOp>(loc(), offset, byteOffset)),
						m_builder.create<mlir::yul::MLoadOp>(
							loc(), m_builder.create<mlir::yul::AddOp>(loc(), source, byteOffset)));
				});
				offset = m_builder.create<mlir::yul::AddOp>(
					loc(), offset,
					m_builder.create<mlir::yul::MulOp>(loc(), count, wordConstant(uint64_t(32))));
				continue;
			}

			PackedType const type = *packedType(argument.getType());
			unsigned const firstByte = type.leftAligned ? 0 : 32 - type.bytes;
			for (unsigned byte = 0; byte < type.bytes; ++byte)
				m_builder.create<mlir::yul::MStore8Op>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(
						loc(), data,
						m_builder.create<mlir::yul::AddOp>(loc(), offset, wordConstant(uint64_t(byte)))),
					m_builder.create<mlir::yul::ByteOp>(loc(), wordConstant(uint64_t(firstByte + byte)), value));
			offset = m_builder.create<mlir::yul::AddOp>(loc(), offset, wordConstant(uint64_t(type.bytes)));
		}
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

	/// Reject impossible encodings before any read or mutation. Short values
	/// can contain at most 31 bytes (tag <= 62); long values start at 32 bytes.
	/// Solidity specifies Panic(0x22) for either malformed representation.
	void validateStorageBytesEncoding(mlir::Value _packed)
	{
		mlir::Value isLong = m_builder.create<mlir::yul::AndOp>(
			loc(), _packed, wordConstant(uint64_t(1)));
		mlir::Value notLong = m_builder.create<mlir::yul::IsZeroOp>(loc(), isLong);
		mlir::Value shortTag = m_builder.create<mlir::yul::AndOp>(
			loc(), _packed, wordConstant(uint64_t(0xff)));
		mlir::Value longLength = yulShr(
			loc(), wordConstant(uint64_t(1)), _packed);
		mlir::Value invalidShort = m_builder.create<mlir::yul::AndOp>(
			loc(), notLong,
			m_builder.create<mlir::yul::LtOp>(
				loc(), wordConstant(uint64_t(62)), shortTag));
		mlir::Value invalidLong = m_builder.create<mlir::yul::AndOp>(
			loc(), isLong,
			m_builder.create<mlir::yul::LtOp>(
				loc(), longLength, wordConstant(uint64_t(32))));
		emitPanicIf(
			m_builder.create<mlir::yul::OrOp>(loc(), invalidShort, invalidLong), 0x22);
	}

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
		validateStorageBytesEncoding(packed);
		mlir::Value isLong = m_builder.create<mlir::yul::AndOp>(loc(), packed, wordConstant(uint64_t(1)));

		// length = isLong ? (packed - 1) / 2 : (packed & 0xff) / 2, branch-free.
		mlir::Value shortLength = m_builder.create<mlir::yul::AndOp>(loc(), packed, wordConstant(uint64_t(0xff)));
		mlir::Value chosen = m_builder.create<mlir::yul::MulOp>(loc(), isLong, packed);
		mlir::Value notLong = m_builder.create<mlir::yul::IsZeroOp>(loc(), isLong);
		mlir::Value shortPart = m_builder.create<mlir::yul::MulOp>(loc(), notLong, shortLength);
		mlir::Value combined = m_builder.create<mlir::yul::AddOp>(loc(), chosen, shortPart);
		mlir::Value length = yulShr(loc(), wordConstant(uint64_t(1)), combined);

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
			mlir::Value masked = yulShl(
				loc(), keep, yulShr(loc(), keep, packed));
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
					loc(), base, yulShr(loc(), wordConstant(uint64_t(5)), _index));
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
		mlir::Value oldPacked = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
		validateStorageBytesEncoding(oldPacked);
		mlir::Value oldIsLong = m_builder.create<mlir::yul::AndOp>(
			loc(), oldPacked, wordConstant(uint64_t(1)));
		mlir::Value oldLength = yulShr(
			loc(), wordConstant(uint64_t(1)), oldPacked);
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
			mlir::Value masked = yulShl(
				loc(), keep, yulShr(loc(), keep, word));
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
					loc(), base, yulShr(loc(), wordConstant(uint64_t(5)), _index));
				m_builder.create<mlir::yul::SStoreOp>(loc(), to, word);
			});
		}

		// Shrinking a long value must erase every no-longer-owned data slot.
		// A short replacement owns none of the keccak-derived area, while a
		// shorter long replacement retains only its rounded-up word count.
		auto clearOld = m_builder.create<mlir::yul::IfOp>(loc(), oldIsLong);
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&clearOld.getThenRegion().emplaceBlock());
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), slot);
			mlir::Value oldBase = m_builder.create<mlir::yul::Keccak256Op>(
				loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(32)));
			mlir::Value retainedBytes = m_builder.create<mlir::yul::MulOp>(
				loc(), isLong, roundedUp(length));
			emitWordLoop(oldLength, [&](mlir::Value index) {
				auto trailing = m_builder.create<mlir::yul::IfOp>(
					loc(), m_builder.create<mlir::yul::IsZeroOp>(
						loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, retainedBytes)));
				mlir::OpBuilder::InsertionGuard clearGuard(m_builder);
				m_builder.setInsertionPointToStart(&trailing.getThenRegion().emplaceBlock());
				mlir::Value oldSlot = m_builder.create<mlir::yul::AddOp>(
					loc(), oldBase,
					yulShr(
						loc(), wordConstant(uint64_t(5)), index));
				m_builder.create<mlir::yul::SStoreOp>(loc(), oldSlot, wordConstant(uint64_t(0)));
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

	/// `for (i = 0; i < count; ++i) _body(i)`. Aggregate ABI values use one
	/// internal memory word per element, but their encoded elements can have a
	/// different (and sometimes dynamic) stride, so a true element index is
	/// clearer than reusing the byte-oriented word-copy loop above.
	template<typename Body>
	void emitIndexLoop(mlir::Value _count, Body _body)
	{
		auto var = m_builder.create<mlir::yul::VarOp>(loc(), wordConstant(uint64_t(0)));
		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());
			mlir::Value index = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), var);
			m_builder.create<mlir::yul::ConditionOp>(
				loc(), m_builder.create<mlir::yul::LtOp>(loc(), index, _count));
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
				loc(), var, m_builder.create<mlir::yul::AddOp>(loc(), index, wordConstant(uint64_t(1))));
		}
	}

	/// Solidity >= 0.8 reports a failed arithmetic check as `Panic(uint256)`:
	/// selector 0x4e487b71, then the code - 0x11 for overflow, 0x12 for a
	/// division by zero. Reverting bare would be the right control flow with
	/// the wrong answer to `why`.
	void emitPanicIf(mlir::Value _condition, uint64_t _code)
	{
		auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), _condition);
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
		emitPanic(_code);
	}

	void emitPanic(uint64_t _code)
	{
		llvm::APInt selector(256, 0x4e487b71u);
		selector <<= 224;
		m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), wordConstant(selector));
		m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(4)), wordConstant(_code));
		m_builder.create<mlir::yul::RevertOp>(loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(0x24)));
	}

	/// Revert with Error(string) where the message is already a canonical
	/// memory value [length][data...]. The ABI payload begins four bytes after
	/// the selector and includes the required padded string tail.
	void emitErrorStringRevertIf(mlir::Value _condition, mlir::Value _message)
	{
		auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), _condition);
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
		m_usesMemory = true;
		mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), _message);
		mlir::Value padded = roundedUp(length);
		mlir::Value size = m_builder.create<mlir::yul::AddOp>(
			loc(), wordConstant(uint64_t(68)), padded);
		mlir::Value pointer = allocate(size);
		llvm::APInt selector(256, 0x08c379a0u);
		selector <<= 224;
		m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, wordConstant(selector));
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(4))),
			wordConstant(uint64_t(32)));
		m_builder.create<mlir::yul::MStoreOp>(
			loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(36))), length);
		yulMCopy(
			loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(68))),
			m_builder.create<mlir::yul::AddOp>(loc(), _message, wordConstant(uint64_t(32))), length);
		m_builder.create<mlir::yul::RevertOp>(loc(), pointer, size);
	}

	/// Everything outside `unchecked` is checked; the generator marks the
	/// operations inside it, because the block itself is inlined away.
	static bool checked(mlir::Operation& _op) { return !_op.hasAttr("unchecked"); }

	static constexpr uint64_t kPanicOverflow = 0x11;
	static constexpr uint64_t kPanicDivisionByZero = 0x12;

	mlir::Value notValue(mlir::Value _value) { return m_builder.create<mlir::yul::NotOp>(loc(), _value); }
	mlir::Value isZero(mlir::Value _value) { return m_builder.create<mlir::yul::IsZeroOp>(loc(), _value); }

	/// The smallest signed 256-bit value, which is its own negation - the one
	/// input that makes signed negation, division and multiplication overflow.
	mlir::Value signedMinimum()
	{
		llvm::APInt minimum(256, 0);
		minimum.setBit(255);
		return wordConstant(minimum);
	}

	void emitRevertIf(mlir::Value _condition)
	{
		auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), _condition);
		mlir::OpBuilder::InsertionGuard guard(m_builder);
		m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
		mlir::Value zero = wordConstant(uint64_t(0));
		m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
	}

	void validateMappingEnumKeys(mlir::Operation* _operation, mlir::ValueRange _keys)
	{
		auto limits = _operation->getAttrOfType<mlir::ArrayAttr>("enum_key_limits");
		if (!limits)
			return;
		if (limits.size() != _keys.size())
			fail("mapping enum-key metadata disagrees with key count");
		for (auto [key, attribute]: llvm::zip(_keys, limits))
		{
			uint64_t const limit = static_cast<uint64_t>(
				llvm::cast<mlir::IntegerAttr>(attribute).getInt());
			if (limit)
				emitPanicIf(
					isZero(m_builder.create<mlir::yul::LtOp>(loc(), mapped(key), wordConstant(limit))),
					0x21);
		}
	}

	/// Slot of `mapping[key]`, which Solidity defines as
	/// keccak256(key . slot) over the two words written to scratch memory at
	/// 0x00 and 0x20 - the region the language reserves for exactly this. A
	/// nested mapping applies the rule once per key, the result of one round
	/// becoming the base slot of the next.
	mlir::Value mappingSlot(mlir::Value _baseSlot, mlir::ValueRange _keys)
	{
		mlir::Value slot = _baseSlot;
		for (mlir::Value key: _keys)
		{
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), mapped(key));
			m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(32)), slot);
			slot = m_builder.create<mlir::yul::Keccak256Op>(
				loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(64)));
		}
		return slot;
	}

	mlir::Value mappingSlot(uint64_t _baseSlot, mlir::ValueRange _keys)
	{
		return mappingSlot(wordConstant(_baseSlot), _keys);
	}

	/// The ABI name of a type, or nothing when it is one this dispatcher cannot
	/// describe. Arrays retain their recursive element type in the sol dialect;
	/// only the value-element shapes the ABI lowering implements are admitted.
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
		if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_type))
		{
			if (!isABIArray(_type))
				return std::nullopt;
			auto element = abiTypeName(array.getElementType());
			if (!element)
				return std::nullopt;
			return *element + (array.isDynamicallySized() ? "[]" : "[" + std::to_string(array.getSize()) + "]");
		}
		if (auto structure = llvm::dyn_cast<mlir::solidity::StructType>(_type))
		{
			std::string tuple = "(";
			bool first = true;
			for (mlir::Type field: structure.getFieldTypes())
			{
				auto name = abiTypeName(field);
				if (!name)
					return std::nullopt;
				tuple += (first ? "" : ",") + *name;
				first = false;
			}
			return tuple + ")";
		}
		return std::nullopt;
	}

	/// The canonical ABI signature, when every type in it can be named.
	static std::optional<std::string> abiSignature(mlir::solidity::FunctionOp _func)
	{
		if (auto signature = _func->getAttrOfType<mlir::StringAttr>("abi_signature"))
			return signature.str();
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
	/// Value arrays and byte sequences are materialised in memory before the
	/// internal function call, then results go through the same ABI encoder used
	/// by high-level external calls.
	void emitDispatcher(mlir::solidity::ContractOp _contract, mlir::ModuleOp _dst)
	{
		std::vector<std::pair<uint32_t, mlir::solidity::FunctionOp>> entries;
		mlir::solidity::FunctionOp receiveFunction;
		mlir::solidity::FunctionOp fallbackFunction;
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
			if (auto kind = func->getAttrOfType<mlir::StringAttr>("kind"))
			{
				if (kind == "receive")
					receiveFunction = func;
				else if (kind == "fallback")
					fallbackFunction = func;
				continue;
			}
			if (func.getVisibility() && *func.getVisibility() != "public" && *func.getVisibility() != "external")
				continue;
			bool describable = true;
			for (mlir::Type type: func.getArgumentTypes())
				describable = describable && isABIEncodable(type) && abiTypeName(type).has_value();
			for (mlir::Type type: func.getResultTypes())
				describable = describable && isABIEncodable(type) && abiTypeName(type).has_value();
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
		if (entries.empty() && !receiveFunction && !fallbackFunction)
		{
			mlir::Value zero = wordConstant(uint64_t(0));
			m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
			return;
		}

		mlir::Value calldataSize = m_builder.create<mlir::yul::CallDataSizeOp>(loc());
		mlir::Value zero = wordConstant(uint64_t(0));
		if (receiveFunction)
		{
			auto empty = m_builder.create<mlir::yul::IfOp>(
				loc(), m_builder.create<mlir::yul::IsZeroOp>(loc(), calldataSize));
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&empty.getThenRegion().emplaceBlock());
			m_builder.create<mlir::yul::FuncCallOp>(
				loc(), mlir::TypeRange{},
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), yulFunctionName(receiveFunction)),
				mlir::ValueRange{});
			m_builder.create<mlir::yul::ReturnOp>(loc(), zero, zero);
		}

		mlir::Value word = m_builder.create<mlir::yul::CallDataLoadOp>(loc(), wordConstant(uint64_t(0)));
		mlir::Value selector = yulShr(loc(), wordConstant(uint64_t(224)), word);
		mlir::Value hasSelector = m_builder.create<mlir::yul::IsZeroOp>(
			loc(), m_builder.create<mlir::yul::LtOp>(loc(), calldataSize, wordConstant(uint64_t(4))));

		for (auto& [value, func]: entries)
		{
			mlir::Value matches = m_builder.create<mlir::yul::AndOp>(
				loc(), hasSelector,
				m_builder.create<mlir::yul::EqOp>(loc(), selector, wordConstant(value)));
			auto ifOp = m_builder.create<mlir::yul::IfOp>(loc(), matches);
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());

			// Calldata short of the declared arguments has to revert. Reading
			// past the end yields zeros rather than faulting, so without this
			// the call quietly succeeds on arguments nobody supplied - which is
			// what the differential against solc caught.
			size_t const argumentCount = func.getArgumentTypes().size();
			auto declaredHeadSize = abiTupleHeadSize(func.getArgumentTypes());
			if (!declaredHeadSize)
				fail("dispatcher function has an unsupported ABI argument");
			if (argumentCount > 0)
			{
				mlir::Value size = m_builder.create<mlir::yul::CallDataSizeOp>(loc());
				mlir::Value tooShort = m_builder.create<mlir::yul::LtOp>(
					loc(), size, wordConstant(uint64_t(4 + *declaredHeadSize)));
				if (m_debugRevertStrings)
					emitErrorStringRevertIf(tooShort, "Calldata too short");
				else
					emitRevertIf(tooShort);
			}

			llvm::SmallVector<mlir::Value, 4> arguments;
			uint64_t argumentHeadOffset = 0;
			mlir::Value argumentData = zero;
			mlir::Value argumentDataLength = zero;
			if (argumentCount > 0)
			{
				m_usesMemory = true;
				argumentDataLength = m_builder.create<mlir::yul::SubOp>(
					loc(), calldataSize, wordConstant(uint64_t(4)));
				argumentData = allocate(roundedUp(argumentDataLength));
				m_builder.create<mlir::yul::CallDataCopyOp>(
					loc(), argumentData, wordConstant(uint64_t(4)), argumentDataLength);
			}
			for (unsigned i = 0; i < argumentCount; ++i)
			{
				mlir::Type type = func.getArgumentTypes()[i];
				mlir::Value head = m_builder.create<mlir::yul::AddOp>(
					loc(), argumentData, wordConstant(argumentHeadOffset));
				mlir::Value argument = decodeABIComponent(
					type, head, argumentData, argumentDataLength, ABIDecodeContext::Calldata);
				if (uint64_t const enumArrayLimit = indexedIntegerAttr(
						func.getOperation(), "enum_array_limits", i))
					validateEnumArray(type, argument, enumArrayLimit);
				if (indexedAttr(func.getOperation(), "external_function_params", i))
					argument = yulShr(
						loc(), wordConstant(uint64_t(64)), argument);
				if (indexedAttr(func.getOperation(), "external_function_array_params", i))
					argument = normalizeExternalFunctionArray(type, argument);
				arguments.push_back(argument);
				argumentHeadOffset += *abiHeadSize(type);
			}

			llvm::SmallVector<mlir::Type, 1> resultTypes(func.getResultTypes().size(), wordType());
			auto called = m_builder.create<mlir::yul::FuncCallOp>(
				loc(),
				resultTypes,
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), yulFunctionName(func)),
				arguments);

			unsigned const resultCount = called->getNumResults();
			if (resultCount == 0)
				m_builder.create<mlir::yul::ReturnOp>(loc(), zero, zero);
			else
			{
				llvm::SmallVector<mlir::Value, 4> results(called->getResults());
				for (unsigned i = 0; i < results.size(); ++i)
					if (packedType(func.getResultTypes()[i]))
						results[i] = representationConvert(
							results[i], func.getResultTypes()[i], func.getResultTypes()[i]);
				for (unsigned i = 0; i < results.size(); ++i)
					if (indexedAttr(func.getOperation(), "external_function_results", i))
						results[i] = yulShl(
							loc(), wordConstant(uint64_t(64)), results[i]);
				mlir::Value encoded = encodeABIValues(func.getResultTypes(), results);
				mlir::Value length = m_builder.create<mlir::yul::MLoadOp>(loc(), encoded);
				m_builder.create<mlir::yul::ReturnOp>(
					loc(),
					m_builder.create<mlir::yul::AddOp>(loc(), encoded, wordConstant(uint64_t(32))),
					length);
			}
		}

		if (fallbackFunction)
		{
			llvm::SmallVector<mlir::Value, 1> arguments;
			if (fallbackFunction.getArgumentTypes().size() == 1)
			{
				m_usesMemory = true;
				mlir::Value pointer = allocate(m_builder.create<mlir::yul::AddOp>(
					loc(), wordConstant(uint64_t(32)), roundedUp(calldataSize)));
				m_builder.create<mlir::yul::MStoreOp>(loc(), pointer, calldataSize);
				m_builder.create<mlir::yul::CallDataCopyOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					zero, calldataSize);
				arguments.push_back(pointer);
			}
			llvm::SmallVector<mlir::Type, 1> resultTypes(
				fallbackFunction.getResultTypes().size(), wordType());
			auto called = m_builder.create<mlir::yul::FuncCallOp>(
				loc(), resultTypes,
				mlir::FlatSymbolRefAttr::get(m_builder.getContext(), yulFunctionName(fallbackFunction)),
				arguments);
			if (called.getNumResults() == 1)
			{
				mlir::Value pointer = called.getResult(0);
				m_builder.create<mlir::yul::ReturnOp>(
					loc(), m_builder.create<mlir::yul::AddOp>(loc(), pointer, wordConstant(uint64_t(32))),
					m_builder.create<mlir::yul::MLoadOp>(loc(), pointer));
			}
			else
				m_builder.create<mlir::yul::ReturnOp>(loc(), zero, zero);
			return;
		}

		m_builder.create<mlir::yul::RevertOp>(loc(), zero, zero);
	}

	uint64_t storageSlot(llvm::StringRef _name)
	{
		auto it = m_storageSlots.find(_name);
		if (it == m_storageSlots.end())
			fail("reference to unknown state variable '" + _name.str() + "'");
		return it->second;
	}

	mlir::Value loadPackedState(llvm::StringRef _name, mlir::Type _type)
	{
		mlir::Value value = m_builder.create<mlir::yul::SLoadOp>(
			loc(), wordConstant(storageSlot(_name)));
		unsigned const offset = m_storageOffsets.lookup(_name);
		unsigned const bytes = m_storageBytes.lookup(_name) ? m_storageBytes.lookup(_name) : 32;
		if (offset)
			value = yulShr(
				loc(), wordConstant(uint64_t(offset * 8)), value);
		if (bytes < 32)
			value = m_builder.create<mlir::yul::AndOp>(
				loc(), value, wordConstant(llvm::APInt::getLowBitsSet(256, bytes * 8)));
		return representationConvert(
			value, mlir::solidity::UIntType::get(m_builder.getContext(), 256), _type);
	}

	void storePackedState(llvm::StringRef _name, mlir::Value _value, mlir::Type _type)
	{
		unsigned const offset = m_storageOffsets.lookup(_name);
		unsigned const bytes = m_storageBytes.lookup(_name) ? m_storageBytes.lookup(_name) : 32;
		mlir::Value slot = wordConstant(storageSlot(_name));
		if (offset == 0 && bytes >= 32)
		{
			m_builder.create<mlir::yul::SStoreOp>(loc(), slot, _value);
			return;
		}

		mlir::Value raw = representationConvert(
			_value, _type, mlir::solidity::UIntType::get(m_builder.getContext(), 256));
		llvm::APInt fieldMask = llvm::APInt::getLowBitsSet(256, bytes * 8);
		raw = m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(fieldMask));
		if (offset)
			raw = yulShl(
				loc(), wordConstant(uint64_t(offset * 8)), raw);

		fieldMask <<= offset * 8;
		mlir::Value previous = m_builder.create<mlir::yul::SLoadOp>(loc(), slot);
		mlir::Value retained = m_builder.create<mlir::yul::AndOp>(
			loc(), previous, wordConstant(~fieldMask));
		m_builder.create<mlir::yul::SStoreOp>(
			loc(), slot, m_builder.create<mlir::yul::OrOp>(loc(), retained, raw));
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
		auto cleanOperand = [&](mlir::Value _operand) {
			return representationConvert(mapped(_operand), _operand.getType(), _operand.getType());
		};
		// sol shift ops carry ($value, $shift); yul builtins take
		// (shift, value) - swap during conversion. A shift is also a typed
		// cleanup boundary: legacy inline assembly may leave dirty bytes below a
		// bytesN value, but those bytes do not participate in the operation.
		mlir::Value result;
		if (auto shl = llvm::dyn_cast<mlir::solidity::ShlOp>(&_op))
		{
			mlir::Value shift = mapped(shl.getShift());
			mlir::Value value = representationConvert(
				mapped(shl.getValue()), shl.getValue().getType(), shl.getValue().getType());
			result = yulShl(loc(), shift, value);
		}
		else if (auto shr = llvm::dyn_cast<mlir::solidity::ShrOp>(&_op))
		{
			mlir::Value shift = mapped(shr.getShift());
			mlir::Value value = representationConvert(
				mapped(shr.getValue()), shr.getValue().getType(), shr.getValue().getType());
			result = yulShr(loc(), shift, value);
		}
		else if (auto sar = llvm::dyn_cast<mlir::solidity::SarOp>(&_op))
		{
			mlir::Value shift = mapped(sar.getShift());
			mlir::Value value = representationConvert(
				mapped(sar.getValue()), sar.getValue().getType(), sar.getValue().getType());
			result = yulSar(loc(), shift, value);
		}
		else if (auto add = llvm::dyn_cast<mlir::solidity::AddOp>(&_op))
		{
			mlir::Value a = cleanOperand(add.getLhs()), b = cleanOperand(add.getRhs());
			result = m_builder.create<mlir::yul::AddOp>(loc(), a, b);
			if (checked(_op))
			{
				if (llvm::isa<mlir::solidity::IntType>(add.getLhs().getType()))
				{
					// Signed: the sum moved the wrong way for the sign of b.
					mlir::Value zero = wordConstant(uint64_t(0));
					mlir::Value positive = m_builder.create<mlir::yul::SGtOp>(loc(), b, zero);
					mlir::Value negative = m_builder.create<mlir::yul::SLtOp>(loc(), b, zero);
					mlir::Value fell = m_builder.create<mlir::yul::SLtOp>(loc(), result, a);
					mlir::Value rose = m_builder.create<mlir::yul::SGtOp>(loc(), result, a);
					emitPanicIf(
						m_builder.create<mlir::yul::OrOp>(
							loc(),
							m_builder.create<mlir::yul::AndOp>(loc(), positive, fell),
							m_builder.create<mlir::yul::AndOp>(loc(), negative, rose)),
						kPanicOverflow);
				}
				else
					// Unsigned: a sum below either operand wrapped.
					emitPanicIf(m_builder.create<mlir::yul::LtOp>(loc(), result, a), kPanicOverflow);
			}
		}
		else if (auto sub = llvm::dyn_cast<mlir::solidity::SubOp>(&_op))
		{
			mlir::Value a = cleanOperand(sub.getLhs()), b = cleanOperand(sub.getRhs());
			result = m_builder.create<mlir::yul::SubOp>(loc(), a, b);
			if (checked(_op))
			{
				if (llvm::isa<mlir::solidity::IntType>(sub.getLhs().getType()))
				{
					mlir::Value zero = wordConstant(uint64_t(0));
					mlir::Value positive = m_builder.create<mlir::yul::SGtOp>(loc(), b, zero);
					mlir::Value negative = m_builder.create<mlir::yul::SLtOp>(loc(), b, zero);
					mlir::Value rose = m_builder.create<mlir::yul::SGtOp>(loc(), result, a);
					mlir::Value fell = m_builder.create<mlir::yul::SLtOp>(loc(), result, a);
					emitPanicIf(
						m_builder.create<mlir::yul::OrOp>(
							loc(),
							m_builder.create<mlir::yul::AndOp>(loc(), positive, rose),
							m_builder.create<mlir::yul::AndOp>(loc(), negative, fell)),
						kPanicOverflow);
				}
				else
					// Unsigned: subtracting more than there was.
					emitPanicIf(m_builder.create<mlir::yul::LtOp>(loc(), a, b), kPanicOverflow);
			}
		}
		else if (auto mul = llvm::dyn_cast<mlir::solidity::MulOp>(&_op))
		{
			mlir::Value a = cleanOperand(mul.getLhs()), b = cleanOperand(mul.getRhs());
			result = m_builder.create<mlir::yul::MulOp>(loc(), a, b);
			if (checked(_op))
			{
				// Dividing the product back out has to give the other operand.
				// Zero is excluded first because the division would be by zero.
				bool const isSigned = llvm::isa<mlir::solidity::IntType>(mul.getLhs().getType());
				mlir::Value back = isSigned
					? m_builder.create<mlir::yul::SDivOp>(loc(), result, a).getResult()
					: m_builder.create<mlir::yul::DivOp>(loc(), result, a).getResult();
				mlir::Value mismatched = isZero(m_builder.create<mlir::yul::EqOp>(loc(), back, b));
				mlir::Value wrong = m_builder.create<mlir::yul::AndOp>(loc(), isZero(isZero(a)), mismatched);
				if (isSigned)
				{
					// The one case the division cannot see: the smallest value
					// times -1 is itself.
					mlir::Value isMinusOne
						= m_builder.create<mlir::yul::EqOp>(loc(), a, wordConstant(~llvm::APInt(256, 0)));
					mlir::Value isMinimum = m_builder.create<mlir::yul::EqOp>(loc(), b, signedMinimum());
					wrong = m_builder.create<mlir::yul::OrOp>(
						loc(), wrong, m_builder.create<mlir::yul::AndOp>(loc(), isMinusOne, isMinimum));
				}
				emitPanicIf(wrong, kPanicOverflow);
			}
		}
		else if (auto div = llvm::dyn_cast<mlir::solidity::DivOp>(&_op))
		{
			bool isSigned = llvm::isa<mlir::solidity::IntType>(div.getLhs().getType());
			mlir::Value lhs = cleanOperand(div.getLhs());
			mlir::Value rhs = cleanOperand(div.getRhs());
			// EVM defines x/0 as 0; Solidity does not allow it at all.
			if (checked(_op))
			{
				emitPanicIf(isZero(rhs), kPanicDivisionByZero);
				if (isSigned)
					emitPanicIf(
						m_builder.create<mlir::yul::AndOp>(
							loc(),
							m_builder.create<mlir::yul::EqOp>(loc(), lhs, signedMinimum()),
							m_builder.create<mlir::yul::EqOp>(
								loc(), rhs, wordConstant(~llvm::APInt(256, 0)))),
						kPanicOverflow);
			}
			if (isSigned)
				result = m_builder.create<mlir::yul::SDivOp>(loc(), lhs, rhs);
			else
				result = m_builder.create<mlir::yul::DivOp>(loc(), lhs, rhs);
		}
		else if (auto mod = llvm::dyn_cast<mlir::solidity::ModOp>(&_op))
		{
			bool isSigned = llvm::isa<mlir::solidity::IntType>(mod.getLhs().getType());
			mlir::Value lhs = cleanOperand(mod.getLhs());
			mlir::Value rhs = cleanOperand(mod.getRhs());
			if (checked(_op))
				emitPanicIf(isZero(rhs), kPanicDivisionByZero);
			if (isSigned)
				result = m_builder.create<mlir::yul::SModOp>(loc(), lhs, rhs);
			else
				result = m_builder.create<mlir::yul::ModOp>(loc(), lhs, rhs);
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

		// The EVM arithmetic operators are 256-bit. Solidity arithmetic wraps
		// or overflows at the declared integer width, so narrow results need a
		// typed cleanup boundary of their own after the full-word operation.
		if (llvm::isa<mlir::solidity::AddOp, mlir::solidity::SubOp, mlir::solidity::MulOp>(&_op))
		{
			mlir::Type resultType = _op.getResult(0).getType();
			unsigned width = 256;
			if (auto type = llvm::dyn_cast<mlir::solidity::UIntType>(resultType))
				width = type.getBitWidth();
			else if (auto type = llvm::dyn_cast<mlir::solidity::IntType>(resultType))
				width = type.getBitWidth();
			if (width < 256)
			{
				mlir::Value narrow = representationConvert(result, resultType, resultType);
				if (checked(_op))
					emitPanicIf(
						isZero(m_builder.create<mlir::yul::EqOp>(loc(), narrow, result)),
						kPanicOverflow);
				result = narrow;
			}
		}
		m_map[_op.getResult(0)] = result;
		return true;
	}

	mlir::Value convertCmp(mlir::solidity::CmpOp _cmp)
	{
		// Comparison is a typed cleanup boundary. This remains observable for
		// legacy dirty values such as a bytes4 narrowed to bytes2.
		mlir::Value lhs = representationConvert(
			mapped(_cmp.getLhs()), _cmp.getLhs().getType(), _cmp.getLhs().getType());
		mlir::Value rhs = representationConvert(
			mapped(_cmp.getRhs()), _cmp.getRhs().getType(), _cmp.getRhs().getType());
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
	mlir::Value representationConvert(mlir::Value _word, mlir::Type _sourceType, mlir::Type _targetType)
	{
		// Fixed bytes live at the high end of an EVM word; integer-like values
		// live at the low end. Solidity permits equal-width conversions between
		// the two, so crossing that boundary requires an alignment shift rather
		// than only a mask.
		if (auto sourceBytes = llvm::dyn_cast<mlir::solidity::BytesType>(_sourceType))
			if (!llvm::isa<mlir::solidity::BytesType>(_targetType))
				_word = yulShr(
					loc(), wordConstant(uint64_t((32 - sourceBytes.getSize()) * 8)), _word);

		if (auto targetBytes = llvm::dyn_cast<mlir::solidity::BytesType>(_targetType))
		{
			unsigned const width = targetBytes.getSize() * 8;
			if (!llvm::isa<mlir::solidity::BytesType>(_sourceType))
			{
				if (width < 256)
					_word = m_builder.create<mlir::yul::AndOp>(
						loc(), _word, wordConstant(llvm::APInt::getLowBitsSet(256, width)));
				return yulShl(
					loc(), wordConstant(uint64_t(256 - width)), _word);
			}
			// bytesM -> bytesN preserves left alignment and truncates on the
			// right when N is smaller.
			if (width < 256)
				return m_builder.create<mlir::yul::AndOp>(
					loc(), _word, wordConstant(llvm::APInt::getHighBitsSet(256, width)));
			return _word;
		}

		if (auto uintType = llvm::dyn_cast<mlir::solidity::UIntType>(_targetType))
		{
			unsigned width = uintType.getBitWidth();
			if (width >= 256)
				return _word;
			mlir::Value mask = wordConstant(llvm::APInt::getLowBitsSet(256, width));
			return m_builder.create<mlir::yul::AndOp>(loc(), _word, mask);
		}
		if (auto intType = llvm::dyn_cast<mlir::solidity::IntType>(_targetType))
		{
			unsigned width = intType.getBitWidth();
			if (width >= 256)
				return _word;
			// signextend(b, x) extends from byte index b.
			mlir::Value byteIndex = wordConstant(uint64_t(width / 8 - 1));
			return m_builder.create<mlir::yul::SignExtendOp>(loc(), byteIndex, _word);
		}
		if (llvm::isa<mlir::solidity::AddressType>(_targetType))
		{
			mlir::Value mask = wordConstant(llvm::APInt::getLowBitsSet(256, 160));
			return m_builder.create<mlir::yul::AndOp>(loc(), _word, mask);
		}
		if (llvm::isa<mlir::solidity::BoolType>(_targetType))
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

	/// An inline assembly block, imported rather than re-invented.
	///
	/// The generator serialises the Yul it was given and the libyul importer
	/// turns that back into `yul` dialect ops, so this only has to splice them
	/// in and connect the ends: Solidity's values go in through a prologue of
	/// declarations, and whatever the block assigned is read back out by
	/// `solidity.assembly_bind`.
	/// True when the block ended the enclosing one - `assembly { revert(0, 0) }`
	/// is a terminator, and anything emitted after it lands past the end.
	bool convertInlineAssembly(mlir::solidity::InlineAssemblyOp _assembly)
	{
		// Bindings are lexical to one assembly block. Retaining a same-named
		// variable from an earlier function made this block ignore its real
		// Solidity input (for example multiple `ret := val` helpers).
		m_assemblyVars.clear();
		// Every Solidity variable the block mentions is declared ahead of it,
		// so the source is valid strict assembly on its own. The initialisers
		// are replaced with the real values while splicing.
		// Declared in one order: the values coming in first, then anything the
		// block assigns that had no value to start from.
		std::vector<std::string> names;
		llvm::StringMap<mlir::Value> incoming;
		for (auto [name, value]: llvm::zip(_assembly.getInputNames(), _assembly.getInputs()))
		{
			std::string const text = llvm::cast<mlir::StringAttr>(name).getValue().str();
			names.push_back(text);
			incoming[text] = mapped(value);
		}
		for (std::string const& name: assemblyBindingsAfter(_assembly))
			if (!incoming.contains(name))
				names.push_back(name);

		std::string source = "{\n";
		for (std::string const& name: names)
			source += "let " + name + " := 0\n";
		source += _assembly.getYulSource().str() + "\n}";

		std::string error;
		mlir::OwningOpRef<mlir::ModuleOp> imported
			= solidity::mlirgen::importYulSource(
				"inline-assembly", source, *m_builder.getContext(), error, m_evmVersion);
		if (!imported)
			fail("inline assembly: " + error);

		// Every assembly block has its own Yul function namespace. Once its
		// operations are spliced into the surrounding Solidity function, two
		// blocks may otherwise both define `f`, or collide with a generated
		// Solidity symbol. Give the imported lexical namespace a stable unique
		// prefix and retarget all calls before cloning it.
		llvm::StringMap<std::string> renamedFunctions;
		std::string const prefix = "__asm" + std::to_string(m_inlineAssemblySequence++) + "_";
		imported->walk([&](mlir::yul::FuncOp func) {
			std::string const oldName = func.getSymName().str();
			std::string const newName = prefix + oldName;
			renamedFunctions[oldName] = newName;
			func->setAttr(
				mlir::SymbolTable::getSymbolAttrName(), m_builder.getStringAttr(newName));
		});
		imported->walk([&](mlir::yul::FuncCallOp call) {
			auto renamed = renamedFunctions.find(call.getCallee());
			if (renamed != renamedFunctions.end())
				call->setAttr(
					"callee", mlir::FlatSymbolRefAttr::get(m_builder.getContext(), renamed->second));
		});

		mlir::IRMapping mapping;
		bool terminated = false;
		for (mlir::Operation& op: imported->getBody()->getOperations())
		{
			mlir::Operation* cloned = m_builder.clone(op, mapping);
			terminated = cloned->hasTrait<mlir::OpTrait::IsTerminator>();
			if (auto var = llvm::dyn_cast<mlir::yul::VarOp>(cloned))
				if (auto name = var->getAttrOfType<mlir::StringAttr>("yul_name"))
				{
					// The prologue's zero stands in for a value the caller has;
					// this is where the two are joined up.
					auto known = incoming.find(name.getValue());
					if (known != incoming.end() && !m_assemblyVars.contains(name.getValue()))
						var.getInitMutable().assign(known->second);
					m_assemblyVars[name.getValue().str()] = var.getResult();
				}
		}
		return terminated;
	}

	/// The names `solidity.assembly_bind` asks for right after this block -
	/// which is how the generator says which Solidity variables it touches.
	std::vector<std::string> assemblyBindingsAfter(mlir::solidity::InlineAssemblyOp _assembly)
	{
		std::vector<std::string> names;
		for (mlir::Operation* next = _assembly->getNextNode(); next; next = next->getNextNode())
		{
			auto bind = llvm::dyn_cast<mlir::solidity::AssemblyBindOp>(next);
			if (!bind)
				break;
			names.push_back(bind.getVarName().str());
		}
		return names;
	}

	/// The address of `a[i]` when `a` is in memory: a pointer to
	/// [length][data...], so the elements start one word in.
	mlir::Value memoryElement(mlir::Operation& _op, mlir::Value _pointer, mlir::Value _index)
	{
		// Only a dynamic array carries its length in front of the elements.
		bool dynamic = _op.hasAttr("dynamic");
		if (_op.getNumOperands() > 0)
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_op.getOperand(0).getType()))
				dynamic = dynamic || array.isDynamicallySized();
		mlir::Value data = dynamic
			? m_builder.create<mlir::yul::AddOp>(loc(), _pointer, wordConstant(uint64_t(32))).getResult()
			: _pointer;
		return m_builder.create<mlir::yul::AddOp>(
			loc(), data, m_builder.create<mlir::yul::MulOp>(loc(), _index, wordConstant(uint64_t(32))));
	}

	/// Where a dynamic array's elements start: keccak256 of its slot, which is
	/// what keeps them clear of whatever else the layout put nearby.
	mlir::Value dynamicArrayData(mlir::Value _slot)
	{
		m_builder.create<mlir::yul::MStoreOp>(loc(), wordConstant(uint64_t(0)), _slot);
		return m_builder.create<mlir::yul::Keccak256Op>(
			loc(), wordConstant(uint64_t(0)), wordConstant(uint64_t(32)));
	}

	/// The slot of `a[i]`. A dynamic array's data starts away from its slot,
	/// which holds the length; a fixed one starts at the slot itself.
	mlir::Value arrayBaseSlot(mlir::Operation& _op)
	{
		// Every generated storage-array operation carries its base as operand
		// zero.  For a top-level state variable that maps to the declared root
		// slot; for a nested array it is the already-computed element slot.  The
		// diagnostic varName is still the root variable in both cases, so looking
		// it up first incorrectly sends nested indexing back to the outer length
		// slot.
		if (_op.getNumOperands() > 0)
			return mapped(_op.getOperand(0));
		if (auto name = _op.getAttrOfType<mlir::StringAttr>("varName"))
			if (auto state = m_storageSlots.find(name.getValue()); state != m_storageSlots.end())
				return wordConstant(state->second);
		fail("array access has neither a state slot nor a storage-reference operand");
	}

	mlir::Value arrayElementSlot(mlir::Operation& _op, mlir::Value _index)
	{
		mlir::Value slot = arrayBaseSlot(_op);
		mlir::Value base = _op.hasAttr("dynamic") ? dynamicArrayData(slot) : slot;
		if (auto elementSlots = _op.getAttrOfType<mlir::IntegerAttr>("elementSlots"))
			return m_builder.create<mlir::yul::AddOp>(
				loc(), base,
				m_builder.create<mlir::yul::MulOp>(
					loc(), mapped(_index),
					wordConstant(static_cast<uint64_t>(elementSlots.getInt()))));
		mlir::Type elementType = llvm::cast<mlir::solidity::ArrayType>(_op.getOperand(0).getType()).getElementType();
		return storageArrayElementSlot(base, mapped(_index), elementType, elementBytes(_op));
	}

	unsigned elementBytes(mlir::Operation& _op)
	{
		if (auto bytes = _op.getAttrOfType<mlir::IntegerAttr>("elementBytes"))
			return static_cast<unsigned>(bytes.getInt());
		if (_op.getNumOperands() > 0)
			if (auto array = llvm::dyn_cast<mlir::solidity::ArrayType>(_op.getOperand(0).getType()))
				return storageValueBytes(array.getElementType());
		return 32;
	}

	mlir::Value storageArrayElementSlot(
		mlir::Value _base,
		mlir::Value _index,
		mlir::Type _elementType,
		unsigned _bytes)
	{
		mlir::Value offset = _index;
		if (_bytes < 32
			&& !llvm::isa<mlir::solidity::ArrayType, mlir::solidity::StructType>(_elementType))
			offset = m_builder.create<mlir::yul::DivOp>(
				loc(), _index, wordConstant(static_cast<uint64_t>(32 / _bytes)));
		else
		{
			uint64_t const slots = storageSlotSpan(_elementType);
			if (slots != 1)
				offset = m_builder.create<mlir::yul::MulOp>(loc(), _index, wordConstant(slots));
		}
		return m_builder.create<mlir::yul::AddOp>(loc(), _base, offset);
	}

	mlir::Value loadStorageArrayElement(
		mlir::Type _type,
		mlir::Value _slot,
		mlir::Value _index,
		unsigned _bytes)
	{
		mlir::Value value = m_builder.create<mlir::yul::SLoadOp>(loc(), _slot);
		if (_bytes < 32)
		{
			mlir::Value within = m_builder.create<mlir::yul::ModOp>(
				loc(), _index, wordConstant(static_cast<uint64_t>(32 / _bytes)));
			mlir::Value shift = m_builder.create<mlir::yul::MulOp>(
				loc(), within, wordConstant(static_cast<uint64_t>(_bytes * 8)));
			value = yulShr(loc(), shift, value);
			value = m_builder.create<mlir::yul::AndOp>(
				loc(), value, wordConstant(llvm::APInt::getLowBitsSet(256, _bytes * 8)));
		}
		return representationConvert(
			value, mlir::solidity::UIntType::get(m_builder.getContext(), 256), _type);
	}

	void storeStorageArrayElement(
		mlir::Type _type,
		mlir::Value _slot,
		mlir::Value _index,
		mlir::Value _value,
		unsigned _bytes)
	{
		if (_bytes >= 32)
		{
			m_builder.create<mlir::yul::SStoreOp>(loc(), _slot, _value);
			return;
		}
		mlir::Value raw = representationConvert(
			_value, _type, mlir::solidity::UIntType::get(m_builder.getContext(), 256));
		llvm::APInt const fieldMask = llvm::APInt::getLowBitsSet(256, _bytes * 8);
		raw = m_builder.create<mlir::yul::AndOp>(loc(), raw, wordConstant(fieldMask));
		mlir::Value within = m_builder.create<mlir::yul::ModOp>(
			loc(), _index, wordConstant(static_cast<uint64_t>(32 / _bytes)));
		mlir::Value shift = m_builder.create<mlir::yul::MulOp>(
			loc(), within, wordConstant(static_cast<uint64_t>(_bytes * 8)));
		raw = yulShl(loc(), shift, raw);
		mlir::Value shiftedMask = yulShl(
			loc(), shift, wordConstant(fieldMask));
		mlir::Value previous = m_builder.create<mlir::yul::SLoadOp>(loc(), _slot);
		mlir::Value retained = m_builder.create<mlir::yul::AndOp>(
			loc(), previous, m_builder.create<mlir::yul::NotOp>(loc(), shiftedMask));
		m_builder.create<mlir::yul::SStoreOp>(
			loc(), _slot, m_builder.create<mlir::yul::OrOp>(loc(), retained, raw));
	}

	/// `scf.if` with results, which is how a variable assigned in a branch
	/// survives the branch. Yul has no such thing, so it is the same trick as
	/// the loop: a mutable variable per carried value, assigned by each arm.
	void convertIfWithResults(mlir::scf::IfOp _if)
	{
		llvm::SmallVector<mlir::Value, 4> slots;
		for (mlir::Type type: _if.getResultTypes())
		{
			(void) type;
			slots.push_back(m_builder.create<mlir::yul::VarOp>(loc(), wordConstant(uint64_t(0))));
		}

		mlir::Value condition = mapped(_if.getCondition());
		auto emitArm = [&](mlir::Region& _region, mlir::Value _guard) {
			if (_region.empty())
				return;
			auto guardIf = m_builder.create<mlir::yul::IfOp>(loc(), _guard);
			mlir::OpBuilder::InsertionGuard inner(m_builder);
			m_builder.setInsertionPointToStart(&guardIf.getThenRegion().emplaceBlock());
			for (mlir::Operation& op: _region.front().getOperations())
			{
				if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(&op))
				{
					for (auto [index, value]: llvm::enumerate(yield.getOperands()))
						if (index < slots.size())
							m_builder.create<mlir::yul::AssignOp>(loc(), slots[index], mapped(value));
					break;
				}
				if (convertOp(op))
					break;
			}
		};

		emitArm(_if.getThenRegion(), condition);
		emitArm(_if.getElseRegion(), m_builder.create<mlir::yul::IsZeroOp>(loc(), condition));

		for (auto [index, result]: llvm::enumerate(_if.getResults()))
			if (index < slots.size())
				m_map[result] = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), slots[index]);
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

		mlir::Block& after = _while.getAfter().front();
		mlir::solidity::LoopPostOp postMarker;
		for (mlir::Operation& op: after.getOperations())
			if (auto marker = llvm::dyn_cast<mlir::solidity::LoopPostOp>(&op))
			{
				postMarker = marker;
				break;
			}

		bool const isDoWhile = _while->hasAttr("do_while");
		mlir::Value doConditionSlot;
		if (isDoWhile)
			doConditionSlot = m_builder.create<mlir::yul::VarOp>(loc(), wordConstant(uint64_t(1)));

		auto forOp = m_builder.create<mlir::yul::ForOp>(loc());
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());

			mlir::Value cond;
			if (isDoWhile)
				cond = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), doConditionSlot);
			else
			{
				mlir::Block& before = _while.getBefore().front();
				readInto(before);
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
					if (convertOp(op))
						break;
				}
			}
			m_builder.create<mlir::yul::ConditionOp>(loc(), cond ? cond : wordConstant(uint64_t(0)));
		}

		m_loopSlotStack.push_back(slots);
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());
			readInto(after);
			for (mlir::Operation& op: after.getOperations())
			{
				if (auto marker = llvm::dyn_cast<mlir::solidity::LoopPostOp>(&op))
				{
					// Fallthrough reaches the post region with this state.
					// A continue stored its own point-in-time state before
					// jumping here; a break never reaches the post region.
					assignAll(marker.getOperands());
					break;
				}
				if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(&op))
				{
					assignAll(yield.getOperands());
					break;
				}
				if (convertOp(op))
					break;
			}
		}
		{
			mlir::OpBuilder::InsertionGuard guard(m_builder);
			m_builder.setInsertionPointToStart(&forOp.getPostRegion().emplaceBlock());
			if (postMarker)
			{
				// Values defined in the source body do not dominate Yul's
				// sibling post region. Reload the mutable slots and remap the
				// source block arguments plus marker operands so update/condition
				// expressions use the state saved by body fallthrough/continue.
				for (auto [index, argument]: llvm::enumerate(after.getArguments()))
					if (index < slots.size())
						m_map[argument]
							= m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), slots[index]);
				for (auto [index, value]: llvm::enumerate(postMarker.getOperands()))
					if (index < slots.size())
						m_map[value] = m_builder.create<mlir::yul::VarLoadOp>(loc(), wordType(), slots[index]);

				bool inPost = false;
				for (mlir::Operation& op: after.getOperations())
				{
					if (&op == postMarker.getOperation())
					{
						inPost = true;
						continue;
					}
					if (!inPost)
						continue;
					if (auto condition = llvm::dyn_cast<mlir::solidity::LoopConditionOp>(&op))
					{
						if (isDoWhile)
							m_builder.create<mlir::yul::AssignOp>(
								loc(), doConditionSlot, mapped(condition.getCondition()));
						continue;
					}
					if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(&op))
					{
						assignAll(yield.getOperands());
						break;
					}
					if (convertOp(op))
						break;
				}
			}
		}
		m_loopSlotStack.pop_back();

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

mlir::OwningOpRef<mlir::ModuleOp> convertSolToYul(
	mlir::ModuleOp _module,
	std::string& _error,
	bool _creation,
	langutil::EVMVersion _evmVersion)
{
	mlir::MLIRContext& ctx = *_module.getContext();
	ctx.getOrLoadDialect<mlir::yul::YulDialect>();
	referencedContracts().clear();
	return SolToYulConverter(ctx, _creation, _evmVersion).run(_module, _error);
}

std::set<std::string> const& lastReferencedContracts()
{
	return referencedContracts();
}

} // namespace solidity::mlirgen
