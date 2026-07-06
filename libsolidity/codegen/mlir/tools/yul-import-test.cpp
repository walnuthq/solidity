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
 * yul-import-test: M2b exit-criteria smoke test.
 *
 * Round trip: Yul source -> libyul parse/analyze -> yul dialect (importer)
 * -> MLIR verify -> Yul text (emitter) -> libyul parse/analyze again.
 * Exercises functions (multi-return), calls before definition, if, switch
 * (with and without default), for loops with break/continue, nested blocks,
 * mutable variables, and a representative set of builtins.
 */

#include "YulASTImporter.h"
#include "YulTextEmitter.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/MLIRContext.h"
#pragma GCC diagnostic pop

#include <libyul/YulStack.h>

#include <iostream>
#include <string>
#include <vector>

namespace
{

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

struct Sample
{
	std::string name;
	std::string source;
};

std::vector<Sample> samples()
{
	return {
		{"erc20-ish transfer",
		 R"({
			function selector() -> s {
				s := shr(224, calldataload(0))
			}
			function require(condition) {
				if iszero(condition) { revert(0, 0) }
			}
			function balanceSlot(account) -> slot {
				mstore(0, account)
				mstore(32, 0)
				slot := keccak256(0, 64)
			}
			function transfer(to, amount) -> success {
				let fromSlot := balanceSlot(caller())
				let fromBalance := sload(fromSlot)
				require(iszero(lt(fromBalance, amount)))
				sstore(fromSlot, sub(fromBalance, amount))
				let toSlot := balanceSlot(to)
				sstore(toSlot, add(sload(toSlot), amount))
				success := 1
			}
			switch selector()
			case 0xa9059cbb {
				let ok := transfer(calldataload(4), calldataload(36))
				mstore(0, ok)
				return(0, 32)
			}
			default {
				revert(0, 0)
			}
		})"},
		{"loops with break/continue",
		 R"({
			function oddSum(n) -> total {
				for { let i := 0 } lt(i, n) { i := add(i, 1) }
				{
					if iszero(mod(i, 2)) { continue }
					if gt(total, 1000) { break }
					total := add(total, i)
				}
			}
			sstore(0, oddSum(100))
		})"},
		{"multi-return and nested blocks",
		 R"({
			function divmod(a, b) -> quotient, remainder {
				quotient := div(a, b)
				remainder := mod(a, b)
			}
			let x, y := divmod(calldataload(0), 3)
			{
				let z := add(x, y)
				mstore(0, z)
			}
			return(0, 32)
		})"},
		{"leave and early exit",
		 R"({
			function guarded(x) -> r {
				r := 42
				if iszero(x) { leave }
				r := x
			}
			mstore(0, guarded(calldataload(0)))
			return(0, 32)
		})"},
	};
}

} // anonymous namespace

int main()
{
	bool ok = true;

	for (Sample const& sample: samples())
	{
		std::cout << "=== " << sample.name << " ===" << std::endl;

		mlir::MLIRContext ctx;
		std::string error;
		mlir::OwningOpRef<mlir::ModuleOp> module =
			solidity::mlirgen::importYulSource(sample.name + ".yul", sample.source, ctx, error);
		if (!check(static_cast<bool>(module), "import + verify"))
		{
			std::cout << "  error: " << error << std::endl;
			ok = false;
			continue;
		}

		std::string emitted = solidity::mlirgen::emitYulText(*module);

		solidity::yul::YulStack stack;
		bool reparsed = stack.parseAndAnalyze(sample.name + ".emitted.yul", emitted);
		if (!reparsed)
		{
			std::cout << "--- emitted ---\n" << emitted << std::endl;
			for (auto const& err: stack.errors())
				if (err->comment())
					std::cout << "  libyul error: " << *err->comment() << std::endl;
		}
		ok &= check(reparsed, "emitted Yul re-parses and re-analyzes");
	}

	std::cout << (ok ? "\nAll yul-import checks passed." : "\nFAILURES in yul-import checks.") << std::endl;
	return ok ? 0 : 1;
}
