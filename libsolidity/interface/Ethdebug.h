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
 * The ethdebug resources of a contract: the type documents and pointer
 * templates of ethdebug/format/info/resources, derived from analysis results.
 */

#pragma once

#include <libsolidity/ast/ASTForward.h>

#include <liblangutil/EthdebugSchema.h>
#include <liblangutil/SemanticDebugData.h>

#include <map>
#include <optional>
#include <string>

namespace solidity::frontend::ethdebug
{

/// The two resource tables of ethdebug/format/info/resources.
struct Resources
{
	/// Type documents keyed by the compiler's type identifier. Composed types
	/// reference their component types by that identifier.
	std::map<std::string, langutil::ethdebug::schema::Type> types;
	/// The pointer template of every struct, array and mapping type the state
	/// variables have or are composed of, keyed by the compiler's type
	/// identifier: the layout of a value of the type from a base slot, which
	/// the template expects as `slot`. A mapping's template expects `key` as
	/// well. Value types have no template; they are single regions.
	std::map<std::string, langutil::ethdebug::schema::Pointer::Template> pointers;

	/// Adds the tables of @a _other, replacing entries with the same key.
	void merge(Resources _other);
};

/// The resources of @a _contract: the types of its state variables and of the
/// parameters and return variables of every function and modifier compiled
/// into it, and the pointer templates of the storage types of its state
/// variables. @a _sourceIndices maps source unit names to the source IDs of
/// the ethdebug compilation record, which definition locations refer to.
Resources resources(ContractDefinition const& _contract, std::map<std::string, unsigned> const& _sourceIndices);

/// The scope record of @a _contract's state variables in storage and
/// transient storage: each materialized at its slot, a value type as a
/// region, a mapping as the region of its base slot and any other type
/// through the template of the type in resources(), with its declaration
/// and type reference.
langutil::SemanticDebugScope stateVariableScope(
	ContractDefinition const& _contract,
	std::map<std::string, unsigned> const& _sourceIndices
);

/// The program-level context of ethdebug/format/program: every materialized
/// variable of @a _scope whose pointer is closed, i.e. reads no name that is
/// not bound within it, with its declaration, type reference and pointer.
/// The templates the pointer references are in @a _resources, with their
/// parameters bound by the pointer.
/// @returns nothing when no variable qualifies.
std::optional<langutil::ethdebug::schema::program::Context> programContext(
	langutil::SemanticDebugScope const& _scope,
	Resources const& _resources
);

}
