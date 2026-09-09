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

#include <liblangutil/SemanticDebugDataSerialization.h>

#include <liblangutil/Exceptions.h>

#include <libsolutil/Visitor.h>

#include <charconv>
#include <set>
#include <string>
#include <utility>

using namespace solidity;
using namespace solidity::langutil;
using namespace solidity::langutil::ethdebug;

namespace
{

using Phase = SemanticDebugVariablePhase;

[[noreturn]] void invalid(std::string _message)
{
	BOOST_THROW_EXCEPTION(util::JsonValidationError() << util::errinfo_comment(std::move(_message)));
}

std::string member(std::string_view _path, std::string_view _name)
{
	return std::string(_path) + "." + std::string(_name);
}

std::string element(std::string_view _path, size_t _index)
{
	return std::string(_path) + "[" + std::to_string(_index) + "]";
}

/// Phase values use the compiler's dash-separated convention for
/// enumeration values, like `ast-id` in the debug info selection.
constexpr std::pair<Phase, std::string_view> phaseNames[]{
	{Phase::Materialized, "materialized"},
	{Phase::Computed, "computed"},
	{Phase::OptimizedOut, "optimized-out"},
};

std::string_view phaseName(Phase _phase)
{
	for (auto const& [phase, name]: phaseNames)
		if (phase == _phase)
			return name;
	solAssert(false);
}

Phase phaseFromJson(Json const& _json, std::string_view _path)
{
	std::string const name = util::valueOfType<std::string>(_json, _path);
	for (auto const& [phase, phaseName]: phaseNames)
		if (name == phaseName)
			return phase;
	invalid(std::string(_path) + " has the unknown value \"" + name + "\".");
}

/// A materialized variable has a pointer; a computed or optimized-out
/// variable does not.
void requirePhaseConsistency(std::optional<Phase> _phase, bool _hasPointer, std::string_view _path)
{
	if (!_phase)
		return;
	if (*_phase == Phase::Materialized && !_hasPointer)
		invalid(std::string(_path) + " is materialized but has no pointer.");
	if (*_phase != Phase::Materialized && _hasPointer)
		invalid(std::string(_path) + " is " + std::string(phaseName(*_phase)) + " but has a pointer.");
}

// Writing

Json variableToJson(SemanticDebugVariable const& _variable)
{
	solAssert(!_variable.identifier || !_variable.identifier->empty(), "Variable identifier must not be empty.");
	solAssert((_variable.phase == Phase::Materialized) == _variable.pointer.has_value(), "Variable phase and pointer disagree.");
	Json result = Json::object();
	if (_variable.identifier)
		result["identifier"] = *_variable.identifier;
	if (_variable.declarationASTID)
		result["declarationASTID"] = *_variable.declarationASTID;
	if (_variable.declarationSourceRange)
		result["declarationSourceRange"] = *_variable.declarationSourceRange;
	if (_variable.typeID)
		result["typeID"] = *_variable.typeID;
	result["phase"] = phaseName(_variable.phase);
	if (_variable.pointer)
		result["pointer"] = *_variable.pointer;
	return result;
}

Json variableUpdateToJson(SemanticDebugVariableUpdate const& _update)
{
	solAssert(_update.phase || _update.pointer, "A variable update must change the phase or the pointer.");
	solAssert(!_update.phase || (*_update.phase == Phase::Materialized) == _update.pointer.has_value(), "Update phase and pointer disagree.");
	Json result{{"variableASTID", _update.variableASTID}};
	if (_update.phase)
		result["phase"] = phaseName(*_update.phase);
	if (_update.pointer)
		result["pointer"] = *_update.pointer;
	return result;
}

Json scopeToJson(SemanticDebugScope const& _scope)
{
	Json result{{"variableDefinitions", Json::array()}};
	for (SemanticDebugVariable const& variable: _scope.variableDefinitions)
		result["variableDefinitions"].emplace_back(variableToJson(variable));
	// Omitted while empty, which is all code generation produces: only
	// optimizer passes emit updates.
	if (!_scope.variableUpdates.empty())
	{
		result["variableUpdates"] = Json::array();
		for (SemanticDebugVariableUpdate const& update: _scope.variableUpdates)
			result["variableUpdates"].emplace_back(variableUpdateToJson(update));
	}
	return result;
}

// Reading

/// The keys of the two-level scope map are the canonical decimal forms of
/// the AST ID and the instance: an optional sign, digits without leading
/// zeros, within the 64-bit range.
int64_t decimalKey(std::string const& _text, std::string_view _path)
{
	std::string_view digits = _text;
	bool const negative = !digits.empty() && digits.front() == '-';
	if (negative)
		digits.remove_prefix(1);
	bool canonical = !digits.empty() && (digits == "0" ? !negative : digits.front() != '0');
	for (char const digit: digits)
		if (!std::isdigit(static_cast<unsigned char>(digit)))
			canonical = false;
	if (!canonical)
		invalid(std::string(_path) + " has the key \"" + _text + "\", which is not a decimal integer.");

	int64_t value = 0;
	auto const [end, error] = std::from_chars(_text.data(), _text.data() + _text.size(), value);
	if (error != std::errc{} || end != _text.data() + _text.size())
		invalid(std::string(_path) + " has the key \"" + _text + "\", which is out of range.");
	return value;
}

constexpr schema::Pointer::ReadOptions sidecarPointers{.internalExpressions = true};

std::optional<schema::Pointer> optionalPointer(Json const& _json, std::string_view _path)
{
	if (Json const* pointer = util::optionalMember(_json, "pointer", _path))
		return schema::pointerFromJson(*pointer, member(_path, "pointer"), sidecarPointers);
	return std::nullopt;
}

SemanticDebugVariable variableFromJson(Json const& _json, std::string_view _path)
{
	util::requireOnlyMembers(_json, {"identifier", "declarationASTID", "declarationSourceRange", "typeID", "phase", "pointer"}, _path);
	SemanticDebugVariable variable;
	variable.identifier = util::optionalValue<std::string>(_json, "identifier", _path);
	if (variable.identifier && variable.identifier->empty())
		invalid(member(_path, "identifier") + " must not be empty.");
	variable.declarationASTID = util::optionalValue<int64_t>(_json, "declarationASTID", _path);
	if (Json const* range = util::optionalMember(_json, "declarationSourceRange", _path))
		variable.declarationSourceRange = schema::materials::sourceRangeFromJson(*range, member(_path, "declarationSourceRange"));
	variable.typeID = util::optionalValue<std::string>(_json, "typeID", _path);
	variable.phase = phaseFromJson(util::requiredMember(_json, "phase", _path), member(_path, "phase"));
	variable.pointer = optionalPointer(_json, _path);
	requirePhaseConsistency(variable.phase, variable.pointer.has_value(), _path);
	return variable;
}

SemanticDebugVariableUpdate variableUpdateFromJson(Json const& _json, std::string_view _path)
{
	util::requireOnlyMembers(_json, {"variableASTID", "phase", "pointer"}, _path);
	SemanticDebugVariableUpdate update;
	update.variableASTID = util::requiredValue<int64_t>(_json, "variableASTID", _path);
	if (Json const* phase = util::optionalMember(_json, "phase", _path))
		update.phase = phaseFromJson(*phase, member(_path, "phase"));
	update.pointer = optionalPointer(_json, _path);
	if (!update.phase && !update.pointer)
		invalid(std::string(_path) + " must change the phase or the pointer.");
	requirePhaseConsistency(update.phase, update.pointer.has_value(), _path);
	return update;
}

SemanticDebugScope scopeFromJson(Json const& _json, std::string_view _path)
{
	util::requireOnlyMembers(_json, {"variableDefinitions", "variableUpdates"}, _path);
	SemanticDebugScope scope;
	Json const& variables = util::requireArray(util::requiredMember(_json, "variableDefinitions", _path), member(_path, "variableDefinitions"));
	for (size_t index = 0; index < variables.size(); ++index)
		scope.variableDefinitions.emplace_back(variableFromJson(variables.at(index), element(member(_path, "variableDefinitions"), index)));
	if (Json const* updates = util::optionalMember(_json, "variableUpdates", _path))
	{
		util::requireArray(*updates, member(_path, "variableUpdates"));
		for (size_t index = 0; index < updates->size(); ++index)
			scope.variableUpdates.emplace_back(variableUpdateFromJson(updates->at(index), element(member(_path, "variableUpdates"), index)));
	}
	return scope;
}

/// A template reference must resolve against a local templates block in the
/// same pointer document or against the sidecar's pointer table.
void requireResolvableTemplateReferences(
	schema::Pointer const& _pointer,
	std::set<std::string> const& _localTemplates,
	SemanticDebugDataTable const& _table,
	std::string const& _path
)
{
	using Pointer = schema::Pointer;
	auto const recurse = [&](std::shared_ptr<Pointer const> const& _sub, std::string const& _subPath) {
		if (_sub)
			requireResolvableTemplateReferences(*_sub, _localTemplates, _table, _subPath);
	};
	std::visit(util::GenericVisitor{
		[&](Pointer::TemplateReference const& _reference)
		{
			if (!_localTemplates.count(_reference.name) && !_table.findPointerTemplate(_reference.name))
				invalid(_path + " references the unknown pointer template \"" + _reference.name + "\".");
		},
		[&](Pointer::Templates const& _templates)
		{
			std::set<std::string> localTemplates = _localTemplates;
			for (auto const& [templateName, definition]: _templates.templates)
			{
				recurse(definition.body, member(member(_path, "templates"), templateName));
				localTemplates.insert(templateName);
			}
			requireResolvableTemplateReferences(*_templates.in, localTemplates, _table, member(_path, "in"));
		},
		[&](Pointer::Group const& _group)
		{
			for (size_t index = 0; index < _group.members.size(); ++index)
				requireResolvableTemplateReferences(_group.members[index], _localTemplates, _table, element(member(_path, "group"), index));
		},
		[&](Pointer::List const& _list) { recurse(_list.is, member(member(_path, "list"), "is")); },
		[&](Pointer::Conditional const& _conditional)
		{
			recurse(_conditional.then, member(_path, "then"));
			recurse(_conditional.otherwise, member(_path, "else"));
		},
		[&](Pointer::Scope const& _scope) { recurse(_scope.in, member(_path, "in")); },
		[](Pointer::Region const&) {}
	}, _pointer.value);
}

void requireResolvableTemplateReferences(SemanticDebugDataTable const& _table)
{
	for (auto const& [templateName, definition]: _table.pointerTemplates())
		requireResolvableTemplateReferences(*definition.body, {}, _table, member("sidecar.resources.pointers", templateName) + ".for");
	for (auto const& [id, scope]: _table.scopes())
	{
		std::string const path = "sidecar.scopes." + std::to_string(id.astID) + "." + std::to_string(id.instance);
		for (size_t index = 0; index < scope->variableDefinitions.size(); ++index)
			if (auto const& pointer = scope->variableDefinitions[index].pointer)
				requireResolvableTemplateReferences(*pointer, {}, _table, member(element(member(path, "variableDefinitions"), index), "pointer"));
		for (size_t index = 0; index < scope->variableUpdates.size(); ++index)
			if (auto const& pointer = scope->variableUpdates[index].pointer)
				requireResolvableTemplateReferences(*pointer, {}, _table, member(element(member(path, "variableUpdates"), index), "pointer"));
	}
}

}

Json langutil::semanticDebugDataToJson(SemanticDebugDataTable const& _table)
{
	Json result{
		{"format", std::string(SemanticDebugDataFormat)},
		{"version", SemanticDebugDataFormatVersion},
	};
	if (_table.contractName())
		result["contractName"] = *_table.contractName();

	// The resource tables mirror ethdebug.resources; the templates may carry
	// internal expressions here, unlike in the public output.
	if (!_table.types().empty() || !_table.pointerTemplates().empty())
	{
		Json types = Json::object();
		for (auto const& [id, document]: _table.types())
			types[id] = document;
		Json pointers = Json::object();
		for (auto const& [name, definition]: _table.pointerTemplates())
			pointers[name] = definition;
		result["resources"] = Json{{"types", std::move(types)}, {"pointers", std::move(pointers)}};
	}

	// Scope records form a two-level map from the decimal AST ID to the
	// decimal instance, with instance 0 written explicitly. The JSON library
	// orders object members by key, which keeps the output deterministic.
	Json scopes = Json::object();
	for (auto const& [id, scope]: _table.scopes())
	{
		solAssert(scope, "Semantic debug data table contains a null scope record.");
		scopes[std::to_string(id.astID)][std::to_string(id.instance)] = scopeToJson(*scope);
	}
	result["scopes"] = std::move(scopes);
	return result;
}

SemanticDebugDataTable langutil::semanticDebugDataFromJson(Json const& _json)
{
	std::string const root = "sidecar";
	util::requireOnlyMembers(_json, {"format", "version", "contractName", "resources", "scopes"}, root);
	std::string const format = util::requiredValue<std::string>(_json, "format", root);
	if (format != SemanticDebugDataFormat)
		invalid("Unsupported semantic debug data format \"" + format + "\".");
	unsigned const version = util::requiredValue<unsigned>(_json, "version", root);
	if (version != SemanticDebugDataFormatVersion)
		invalid("Unsupported semantic debug data format version " + std::to_string(version) + ".");

	SemanticDebugDataTable table;
	if (std::optional<std::string> contractName = util::optionalValue<std::string>(_json, "contractName", root))
		table.setContractName(std::move(*contractName));

	if (Json const* resources = util::optionalMember(_json, "resources", root))
	{
		std::string const resourcesPath = member(root, "resources");
		util::requireOnlyMembers(*resources, {"types", "pointers"}, resourcesPath);
		if (Json const* types = util::optionalMember(*resources, "types", resourcesPath))
			for (auto& [id, document]: schema::info::typesFromJson(*types, member(resourcesPath, "types")))
				table.setType(id, std::move(document));
		// Pointer templates are public resources, published verbatim: an
		// expression internal to the compiler has no place in them, so the
		// reader does not accept one there.
		if (Json const* pointers = util::optionalMember(*resources, "pointers", resourcesPath))
			for (auto& [name, definition]: schema::info::pointersFromJson(*pointers, member(resourcesPath, "pointers")))
				table.setPointerTemplate(name, std::move(definition));
	}

	Json const& scopes = util::requireObject(util::requiredMember(_json, "scopes", root), member(root, "scopes"));
	for (auto const& [astIDText, instances]: scopes.items())
	{
		std::string const astIDPath = member(member(root, "scopes"), astIDText);
		int64_t const astID = decimalKey(astIDText, member(root, "scopes"));
		util::requireObject(instances, astIDPath);
		for (auto const& [instanceText, scope]: instances.items())
		{
			int64_t const instance = decimalKey(instanceText, astIDPath);
			if (instance < 0)
				invalid(astIDPath + " has the key \"" + instanceText + "\", which must not be negative.");
			table.set(
				ScopeID{astID, instance},
				std::make_shared<SemanticDebugScope const>(scopeFromJson(scope, member(astIDPath, instanceText)))
			);
		}
	}

	requireResolvableTemplateReferences(table);
	return table;
}
