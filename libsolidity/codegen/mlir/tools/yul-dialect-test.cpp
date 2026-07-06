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
 * yul-dialect-test: M1 exit-criteria smoke test for the `yul` MLIR dialect.
 *
 *  1. Builds sample functions (storage increment, memory-loop sum, guarded
 *     revert) programmatically as `yul` dialect IR.
 *  2. Verifies the module and round-trips it through the MLIR printer/parser.
 *  3. Emits Yul text via the YulText target.
 *  4. Feeds the emitted text to libyul (YulStack parse + analyze) - the
 *     emitted program must be a valid strict-assembly Yul program.
 */

#include "YulDialect.h"
#include "YulOps.h"
#include "YulTextEmitter.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#pragma GCC diagnostic pop

#include <libyul/YulStack.h>

#include <iostream>
#include <string>

using namespace mlir;

namespace
{

IntegerType wordType(MLIRContext& _ctx) { return IntegerType::get(&_ctx, 256); }

Value word(OpBuilder& _b, Location _loc, uint64_t _value)
{
	IntegerType type = wordType(*_b.getContext());
	return _b.create<yul::ConstOp>(_loc, type, IntegerAttr::get(type, APInt(256, _value)));
}

/// function increment() { sstore(0, add(sload(0), 1)) leave }
void buildIncrement(OpBuilder& _b, Location _loc, MLIRContext& _ctx)
{
	auto func = _b.create<yul::FuncOp>(_loc, "increment", FunctionType::get(&_ctx, {}, {}));
	Block* body = &func.getBody().emplaceBlock();
	OpBuilder::InsertionGuard guard(_b);
	_b.setInsertionPointToStart(body);

	Value slot = word(_b, _loc, 0);
	Value loaded = _b.create<yul::SLoadOp>(_loc, slot);
	Value one = word(_b, _loc, 1);
	Value sum = _b.create<yul::AddOp>(_loc, loaded, one);
	_b.create<yul::SStoreOp>(_loc, slot, sum);
	_b.create<yul::LeaveOp>(_loc, ValueRange{});
}

/// function sumToN(n) -> r: sums 0..n-1 via a mutable-variable loop.
void buildSumToN(OpBuilder& _b, Location _loc, MLIRContext& _ctx)
{
	IntegerType word256 = wordType(_ctx);
	auto func = _b.create<yul::FuncOp>(_loc, "sumToN", FunctionType::get(&_ctx, {word256}, {word256}));
	Block* body = &func.getBody().emplaceBlock();
	Value n = body->addArgument(word256, _loc);
	OpBuilder::InsertionGuard guard(_b);
	_b.setInsertionPointToStart(body);

	Value zero = word(_b, _loc, 0);
	Value acc = _b.create<yul::VarOp>(_loc, zero);
	Value i = _b.create<yul::VarOp>(_loc, zero);

	auto forOp = _b.create<yul::ForOp>(_loc);
	// cond: continue while lt(i, n)
	{
		OpBuilder::InsertionGuard forGuard(_b);
		_b.setInsertionPointToStart(&forOp.getCondRegion().emplaceBlock());
		Value iv = _b.create<yul::VarLoadOp>(_loc, i);
		Value cond = _b.create<yul::LtOp>(_loc, iv, n);
		_b.create<yul::ConditionOp>(_loc, cond);
	}
	// body: acc := add(acc, i)
	{
		OpBuilder::InsertionGuard forGuard(_b);
		_b.setInsertionPointToStart(&forOp.getBodyRegion().emplaceBlock());
		Value accV = _b.create<yul::VarLoadOp>(_loc, acc);
		Value iv = _b.create<yul::VarLoadOp>(_loc, i);
		Value sum = _b.create<yul::AddOp>(_loc, accV, iv);
		_b.create<yul::AssignOp>(_loc, acc, sum);
	}
	// post: i := add(i, 1)
	{
		OpBuilder::InsertionGuard forGuard(_b);
		_b.setInsertionPointToStart(&forOp.getPostRegion().emplaceBlock());
		Value iv = _b.create<yul::VarLoadOp>(_loc, i);
		Value one = word(_b, _loc, 1);
		Value next = _b.create<yul::AddOp>(_loc, iv, one);
		_b.create<yul::AssignOp>(_loc, i, next);
	}

	Value result = _b.create<yul::VarLoadOp>(_loc, acc);
	_b.create<yul::LeaveOp>(_loc, ValueRange{result});
}

/// function guarded(x) { if iszero(x) { revert(0, 0) } sstore(1, x) }
void buildGuarded(OpBuilder& _b, Location _loc, MLIRContext& _ctx)
{
	IntegerType word256 = wordType(_ctx);
	auto func = _b.create<yul::FuncOp>(_loc, "guarded", FunctionType::get(&_ctx, {word256}, {}));
	Block* body = &func.getBody().emplaceBlock();
	Value x = body->addArgument(word256, _loc);
	OpBuilder::InsertionGuard guard(_b);
	_b.setInsertionPointToStart(body);

	Value isZero = _b.create<yul::IsZeroOp>(_loc, x);
	auto ifOp = _b.create<yul::IfOp>(_loc, isZero);
	{
		OpBuilder::InsertionGuard ifGuard(_b);
		_b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
		Value zero = word(_b, _loc, 0);
		_b.create<yul::RevertOp>(_loc, zero, zero);
	}
	Value slot = word(_b, _loc, 1);
	_b.create<yul::SStoreOp>(_loc, slot, x);
	_b.create<yul::LeaveOp>(_loc, ValueRange{});
}

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

} // anonymous namespace

int main()
{
	MLIRContext ctx;
	ctx.getOrLoadDialect<yul::YulDialect>();

	OpBuilder builder(&ctx);
	Location loc = builder.getUnknownLoc();
	OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
	builder.setInsertionPointToStart(module->getBody());

	buildIncrement(builder, loc, ctx);
	buildSumToN(builder, loc, ctx);
	buildGuarded(builder, loc, ctx);

	bool ok = true;

	// 1. Verify the constructed module.
	ok &= check(succeeded(verify(*module)), "verify constructed yul-dialect module");

	// 2. Round-trip through the MLIR printer/parser.
	std::string printed;
	{
		llvm::raw_string_ostream os(printed);
		module->print(os);
	}
	std::cout << "\n--- yul dialect IR ---\n" << printed << std::endl;

	OwningOpRef<ModuleOp> reparsed = parseSourceString<ModuleOp>(printed, &ctx);
	ok &= check(reparsed && succeeded(verify(*reparsed)), "MLIR print -> parse round-trip");

	if (reparsed)
	{
		unsigned before = 0, after = 0;
		module->walk([&](Operation*) { ++before; });
		reparsed->walk([&](Operation*) { ++after; });
		ok &= check(before == after, "round-trip preserves op count (" + std::to_string(before) + ")");
	}

	// 3. Emit Yul text.
	std::string yulText = solidity::mlirgen::emitYulText(*module);
	std::cout << "--- emitted Yul ---\n" << yulText << std::endl;
	ok &= check(!yulText.empty(), "Yul text emission");

	// 4. The emitted text must be a valid strict-assembly Yul program.
	solidity::yul::YulStack stack;
	bool parsed = stack.parseAndAnalyze("yul-dialect-test.yul", yulText);
	if (!parsed)
		for (auto const& error: stack.errors())
			std::cout << "libyul error: " << (error->comment() ? *error->comment() : std::string("<no comment>"))
					  << std::endl;
	ok &= check(parsed, "libyul parse + analyze of emitted Yul");

	std::cout << (ok ? "\nAll yul-dialect scaffold checks passed." : "\nFAILURES in yul-dialect scaffold checks.")
			  << std::endl;
	return ok ? 0 : 1;
}
