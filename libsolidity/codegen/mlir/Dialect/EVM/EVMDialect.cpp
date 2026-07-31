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
 * Implementation of the `evm` MLIR dialect.
 */

#include "EVMDialect.h"
#include "EVMOps.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::evm;

// Generated dialect definitions.
#include "EVMDialect.cpp.inc"

namespace
{

/// Nothing in this dialect depends on the frame it runs in - the ops read and
/// write machine state that is shared by caller and callee alike - so copying
/// one into another function changes nothing about what it does. Declaring that
/// is all the upstream inliner needs, and it replaces the Yul inliner rather
/// than reimplementing it.
///
/// A halting terminator is left alone: `evm.return` and friends end execution
/// rather than the function, so there is no continuation to branch to.
struct EVMInlinerInterface: public DialectInlinerInterface
{
	using DialectInlinerInterface::DialectInlinerInterface;

	bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final { return true; }
	bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final { return true; }
	bool isLegalToInline(Operation*, Operation*, bool) const final { return true; }

	// Every terminator this dialect defines halts execution outright - stop,
	// return, revert, invalid, selfdestruct. None of them returns to a caller,
	// so an inlined copy needs neither a branch to the continuation nor values
	// to hand back, and both hooks are deliberately empty. The base class
	// declares them unreachable, which is what a function containing a `revert`
	// used to trip over.
	void handleTerminator(Operation*, Block*) const final {}
	void handleTerminator(Operation*, ValueRange) const final {}
};

} // anonymous namespace

void EVMDialect::initialize()
{
	addOperations<
#define GET_OP_LIST
#include "EVMOps.cpp.inc"
		>();
	addInterfaces<EVMInlinerInterface>();
}
