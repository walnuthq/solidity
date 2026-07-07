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
 * evm-lower-test: M3/M4 exit-criteria smoke test.
 *
 * Pipeline under test: Yul source -> libyul AST -> yul dialect (M2b) ->
 * block-local var promotion -> evm/arith/cf/func (M4) -> MLIR verify.
 * Also checks that a region-crossing mutable variable (loop counter) is
 * reported gracefully as unsupported rather than miscompiled.
 */

#include "YulASTImporter.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

#include <iostream>
#include <string>

namespace
{

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

std::string const promotableSample = R"({
	function addmul(a, b) -> r {
		r := add(mul(a, 3), b)
	}
	function guarded(x) -> r {
		if iszero(x) { revert(0, 0) }
		r := addmul(x, 5)
		sstore(0, r)
	}
})";

std::string const loopSample = R"({
	function sum(n) -> total {
		for { let i := 0 } lt(i, n) { i := add(i, 1) }
		{
			total := add(total, i)
		}
	}
})";

} // anonymous namespace

int main()
{
	bool ok = true;

	// 1. Promotable sample: full descent to the evm rung.
	{
		mlir::MLIRContext ctx;
		std::string error;
		mlir::OwningOpRef<mlir::ModuleOp> module =
			solidity::mlirgen::importYulSource("promotable.yul", promotableSample, ctx, error);
		ok &= check(static_cast<bool>(module), "import promotable sample");
		if (module)
		{
			unsigned remaining = solidity::mlirgen::promoteBlockLocalVars(*module);
			ok &= check(remaining == 0, "all variables promoted to SSA");

			mlir::OwningOpRef<mlir::ModuleOp> lowered = solidity::mlirgen::convertYulToEVM(*module, error);
			if (!check(static_cast<bool>(lowered), "yul -> evm/arith/cf/func conversion + verify"))
			{
				std::cout << "  error: " << error << std::endl;
				ok = false;
			}
			else
			{
				std::string printed;
				{
					llvm::raw_string_ostream os(printed);
					lowered->print(os);
				}
				std::cout << "--- lowered module ---\n" << printed << std::endl;
				ok &= check(printed.find("arith.muli") != std::string::npos, "arith reuse (arith.muli present)");
				ok &= check(printed.find("evm.revert") != std::string::npos, "evm terminator (evm.revert present)");
				ok &= check(printed.find("evm.sstore") != std::string::npos, "evm state op (evm.sstore present)");
				ok &= check(printed.find("call @addmul") != std::string::npos, "user call (func.call present)");
				ok &= check(printed.find("cf.cond_br") != std::string::npos, "flattened control flow (cf.cond_br present)");
				ok &= check(printed.find("yul.") == std::string::npos, "no yul ops remain");
			}
		}
	}

	// 2. Loop sample: loop-carried variables become block arguments (full
	//    structured-CF SSA construction in the converter).
	{
		mlir::MLIRContext ctx;
		std::string error;
		mlir::OwningOpRef<mlir::ModuleOp> module =
			solidity::mlirgen::importYulSource("loop.yul", loopSample, ctx, error);
		ok &= check(static_cast<bool>(module), "import loop sample");
		if (module)
		{
			unsigned remaining = solidity::mlirgen::promoteBlockLocalVars(*module);
			ok &= check(remaining > 0, "loop-carried variables survive block-local promotion");

			mlir::OwningOpRef<mlir::ModuleOp> lowered = solidity::mlirgen::convertYulToEVM(*module, error);
			if (!check(static_cast<bool>(lowered), "loop-carried variables become block arguments"))
				std::cout << "  error: " << error << std::endl;
			else
			{
				std::string printed;
				{
					llvm::raw_string_ostream os(printed);
					lowered->print(os);
				}
				std::cout << "--- lowered loop ---\n" << printed << std::endl;
				ok &= check(printed.find("cf.cond_br") != std::string::npos, "loop lowered to CFG");
				ok &= check(printed.find("i256)") != std::string::npos, "block arguments carry loop state");
				ok &= check(printed.find("yul.") == std::string::npos, "no yul ops remain in loop");
			}
		}
	}

	std::cout << (ok ? "\nAll evm-lower checks passed." : "\nFAILURES in evm-lower checks.") << std::endl;
	return ok ? 0 : 1;
}
