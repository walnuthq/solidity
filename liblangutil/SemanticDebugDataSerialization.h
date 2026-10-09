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
 * The JSON sidecar that carries the semantic debug info side table across a
 * textual Yul boundary, as specified in
 * docs/internals/ethdebug_internal_debug_info.rst.
 */

#pragma once

#include <liblangutil/SemanticDebugDataTable.h>

#include <libsolutil/JSON.h>

#include <string_view>

namespace solidity::langutil
{

constexpr std::string_view SemanticDebugDataFormat = "solidity-ethdebug-semantic-data";
constexpr unsigned SemanticDebugDataFormatVersion = 1;

/// Writes @a _table as the sidecar: the format and version, the contract
/// name, the resource tables in the shape of ethdebug.resources, and the
/// scope records under the decimal AST ID and instance of their scope ID.
/// Throws ethdebug::schema::EthdebugException for a record that cannot be
/// expressed, which the producer never creates.
Json semanticDebugDataToJson(SemanticDebugDataTable const& _table);

/// Reads a sidecar written by semanticDebugDataToJson, checking it against
/// the format: throws util::JsonValidationError, whose message names the
/// offending location, for an unknown format or version, a malformed
/// member, a scope key that is not a decimal integer in range, a variable
/// whose phase and pointer disagree, a template reference that resolves
/// neither locally nor in the pointer table, or a compiler-internal
/// expression in the pointer table.
SemanticDebugDataTable semanticDebugDataFromJson(Json const& _json);

}
