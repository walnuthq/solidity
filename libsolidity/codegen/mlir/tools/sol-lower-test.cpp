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
 * sol-lower-test: M2 exit-criteria smoke test - the full dialect ladder.
 *
 * Builds a Counter contract programmatically in the `sol` dialect, then
 * descends every rung:
 *
 *   sol dialect -> (SolToYul) -> yul dialect -> Yul text -> libyul analyze
 *                                     |
 *                                     +-> (promote + YulToEVM) -> evm/arith/cf/func -> verify
 */

#include "SolToYul.h"
#include "SolidityDialect.h"
#include "SolidityOps.h"
#include "YulDialect.h"
#include "YulTextEmitter.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#pragma GCC diagnostic pop

#include <libyul/YulStack.h>

#include <iostream>
#include <string>

using namespace mlir;
namespace sol = mlir::solidity;
namespace mlirgen = ::solidity::mlirgen;

namespace
{

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

/// Generic sol-op creation helper (avoids depending on generated builder
/// signatures for ops with many optional attributes).
Operation* createSolOp(
	OpBuilder& _b,
	StringRef _name,
	ArrayRef<NamedAttribute> _attrs,
	ValueRange _operands,
	TypeRange _results,
	unsigned _numRegions = 0)
{
	OperationState state(_b.getUnknownLoc(), _name);
	state.addAttributes(_attrs);
	state.addOperands(_operands);
	state.addTypes(_results);
	for (unsigned i = 0; i < _numRegions; ++i)
		state.addRegion();
	return _b.create(state);
}

} // anonymous namespace

int main()
{
	MLIRContext ctx;
	ctx.getOrLoadDialect<sol::SolidityDialect>();
	ctx.getOrLoadDialect<yul::YulDialect>();

	OpBuilder b(&ctx);
	Location loc = b.getUnknownLoc();

	auto uint256 = sol::UIntType::get(&ctx, 256);
	auto boolType = sol::BoolType::get(&ctx);

	OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
	b.setInsertionPointToStart(module->getBody());

	// solidity.contract "Counter" { ... }
	Operation* contract = createSolOp(
		b, "solidity.contract", {NamedAttribute(b.getStringAttr("name"), b.getStringAttr("Counter"))}, {}, {}, 1);
	Block* contractBody = &contract->getRegion(0).emplaceBlock();
	b.setInsertionPointToStart(contractBody);

	// state_var count: uint256  (slot 0)
	createSolOp(
		b,
		"solidity.state_var",
		{NamedAttribute(b.getStringAttr("name"), b.getStringAttr("count")),
		 NamedAttribute(b.getStringAttr("type"), TypeAttr::get(uint256))},
		{},
		{});

	// func increment() { count = count + 1; return; }
	{
		auto func = b.create<sol::FunctionOp>(loc, "increment", FunctionType::get(&ctx, {}, {}), "public", "nonpayable");
		Block* body = &func.getBody().emplaceBlock();
		OpBuilder::InsertionGuard guard(b);
		b.setInsertionPointToStart(body);

		Value loaded =
			createSolOp(b, "solidity.load_state", {NamedAttribute(b.getStringAttr("varName"), b.getStringAttr("count"))}, {}, {uint256})
				->getResult(0);
		Value one = b.create<sol::ConstantOp>(loc, b.getIntegerAttr(b.getIntegerType(256), 1), uint256);
		Value sum = createSolOp(b, "solidity.add", {}, {loaded, one}, {uint256})->getResult(0);
		createSolOp(
			b, "solidity.store_state", {NamedAttribute(b.getStringAttr("varName"), b.getStringAttr("count"))}, {sum}, {});
		createSolOp(b, "solidity.return", {}, {}, {});
	}

	// func isPositive(x: uint256) -> bool { return x > 0; }
	{
		auto func = b.create<sol::FunctionOp>(
			loc, "isPositive", FunctionType::get(&ctx, {uint256}, {boolType}), "public", "view");
		Block* body = &func.getBody().emplaceBlock();
		Value x = body->addArgument(uint256, loc);
		OpBuilder::InsertionGuard guard(b);
		b.setInsertionPointToStart(body);

		Value zero = b.create<sol::ConstantOp>(loc, b.getIntegerAttr(b.getIntegerType(256), 0), uint256);
		Value positive =
			createSolOp(
				b, "solidity.cmp", {NamedAttribute(b.getStringAttr("predicate"), b.getStringAttr("gt"))}, {x, zero}, {boolType})
				->getResult(0);
		createSolOp(b, "solidity.return", {}, {positive}, {});
	}

	// func guardedDouble(x: uint256) -> uint256
	//   { require(x != 0); if (x > 100) { return 100; } else { } return x + x; }
	{
		auto func = b.create<sol::FunctionOp>(
			loc, "guardedDouble", FunctionType::get(&ctx, {uint256}, {uint256}), "public", "pure");
		Block* body = &func.getBody().emplaceBlock();
		Value x = body->addArgument(uint256, loc);
		OpBuilder::InsertionGuard guard(b);
		b.setInsertionPointToStart(body);

		Value zero = b.create<sol::ConstantOp>(loc, b.getIntegerAttr(b.getIntegerType(256), 0), uint256);
		Value nonZero =
			createSolOp(
				b, "solidity.cmp", {NamedAttribute(b.getStringAttr("predicate"), b.getStringAttr("ne"))}, {x, zero}, {boolType})
				->getResult(0);
		createSolOp(b, "solidity.require", {}, {nonZero}, {});

		Value hundred = b.create<sol::ConstantOp>(loc, b.getIntegerAttr(b.getIntegerType(256), 100), uint256);
		Value big =
			createSolOp(
				b, "solidity.cmp", {NamedAttribute(b.getStringAttr("predicate"), b.getStringAttr("gt"))}, {x, hundred}, {boolType})
				->getResult(0);
		Operation* ifOp = createSolOp(b, "solidity.if", {}, {big}, {}, 2);
		{
			OpBuilder::InsertionGuard ifGuard(b);
			b.setInsertionPointToStart(&ifOp->getRegion(0).emplaceBlock());
			createSolOp(b, "solidity.return", {}, {hundred}, {});
		}
		ifOp->getRegion(1).emplaceBlock(); // empty else

		Value doubled = createSolOp(b, "solidity.add", {}, {x, x}, {uint256})->getResult(0);
		createSolOp(b, "solidity.return", {}, {doubled}, {});
	}

	bool ok = true;
	ok &= check(succeeded(verify(*module)), "verify hand-built sol-dialect module");

	// Rung 1 -> 2: SolToYul.
	std::string error;
	OwningOpRef<ModuleOp> yulModule = mlirgen::convertSolToYul(*module, error);
	if (!check(static_cast<bool>(yulModule), "sol -> yul conversion + verify"))
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}

	// Rung 2 exit: Yul text accepted by libyul.
	std::string yulText = mlirgen::emitYulText(*yulModule);
	std::cout << "--- emitted Yul ---\n" << yulText << std::endl;
	{
		::solidity::yul::YulStack stack;
		bool parsed = stack.parseAndAnalyze("sol-lower-test.yul", yulText);
		if (!parsed)
			for (auto const& err: stack.errors())
				if (err->comment())
					std::cout << "  libyul error: " << *err->comment() << std::endl;
		ok &= check(parsed, "emitted Yul parses + analyzes with libyul");
	}

	// Rung 2 -> 3: promote + YulToEVM.
	unsigned remaining = mlirgen::promoteBlockLocalVars(*yulModule);
	ok &= check(remaining == 0, "no mutable variables at this rung");
	OwningOpRef<ModuleOp> evmModule = mlirgen::convertYulToEVM(*yulModule, error);
	if (!check(static_cast<bool>(evmModule), "yul -> evm conversion + verify"))
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}

	std::string printed;
	{
		llvm::raw_string_ostream os(printed);
		evmModule->print(os);
	}
	ok &= check(printed.find("evm.sload") != std::string::npos, "storage load survives to evm rung");
	ok &= check(printed.find("evm.sstore") != std::string::npos, "storage store survives to evm rung");
	ok &= check(printed.find("evm.revert") != std::string::npos, "require() became guarded evm.revert");
	ok &= check(printed.find("arith.addi") != std::string::npos, "arith reuse at evm rung");
	ok &= check(printed.find("solidity.") == std::string::npos, "no sol ops remain");
	ok &= check(printed.find("yul.") == std::string::npos, "no yul ops remain");

	std::cout << (ok ? "\nAll sol-lower (full ladder) checks passed." : "\nFAILURES in sol-lower checks.") << std::endl;
	return ok ? 0 : 1;
}
