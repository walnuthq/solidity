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
 * Component that translates Solidity AST into MLIR.
 */

#pragma once

#include <libsolidity/ast/ASTForward.h>
#include <libsolidity/ast/ASTVisitor.h>
#include <libsolidity/interface/OptimiserSettings.h>

#include <liblangutil/EVMVersion.h>
#include <liblangutil/SourceLocation.h>

#include <memory>
#include <string>
#include <vector>

namespace solidity::yul
{
class Object;
}

// Forward declarations for MLIR types
namespace mlir
{
class MLIRContext;
class ModuleOp;
class OpBuilder;
class Value;
class Type;
class Operation;
}

namespace solidity::frontend
{

class CompilerStack;
class ContractDefinition;
class FunctionDefinition;
class VariableDeclaration;
class Statement;
class Expression;

/**
 * AST to MLIR converter.
 * Converts Solidity AST nodes into MLIR operations using the Solidity MLIR dialect.
 */
class MLIRGenerator: public ASTConstVisitor
{
public:
	MLIRGenerator(
		CompilerStack const& _compilerStack,
		langutil::EVMVersion _evmVersion,
		OptimiserSettings const& _optimiserSettings,
		bool _legacyCodegen = false);

	~MLIRGenerator() override;

	/// Generate MLIR module from contract
	std::string generate(ContractDefinition const& _contract);


protected:
	// ASTVisitor interface implementations
	bool visit(ContractDefinition const& _contract) override;
	void endVisit(ContractDefinition const& _contract) override;

	bool visit(FunctionDefinition const& _function) override;
	void endVisit(FunctionDefinition const& _function) override;

	bool visit(VariableDeclaration const& _variable) override;
	void endVisit(VariableDeclaration const& _variable) override;

	bool visit(Block const& _block) override;
	void endVisit(Block const& _block) override;

	bool visit(IfStatement const& _ifStatement) override;
	void endVisit(IfStatement const& _ifStatement) override;

	bool visit(WhileStatement const& _whileStatement) override;
	void endVisit(WhileStatement const& _whileStatement) override;

	bool visit(ForStatement const& _forStatement) override;
	void endVisit(ForStatement const& _forStatement) override;

	bool visit(Return const& _return) override;
	void endVisit(Return const& _return) override;

	bool visit(Assignment const& _assignment) override;
	void endVisit(Assignment const& _assignment) override;

	bool visit(BinaryOperation const& _operation) override;
	void endVisit(BinaryOperation const& _operation) override;

	bool visit(UnaryOperation const& _operation) override;
	void endVisit(UnaryOperation const& _operation) override;

	bool visit(FunctionCall const& _functionCall) override;
	void endVisit(FunctionCall const& _functionCall) override;

	bool visit(Identifier const& _identifier) override;
	void endVisit(Identifier const& _identifier) override;

	bool visit(Literal const& _literal) override;
	void endVisit(Literal const& _literal) override;

private:
	class MLIRGeneratorImpl;
	std::unique_ptr<MLIRGeneratorImpl> m_impl;

	[[maybe_unused]] CompilerStack const& m_compilerStack;
	[[maybe_unused]] langutil::EVMVersion m_evmVersion;
	[[maybe_unused]] OptimiserSettings const& m_optimiserSettings;
};

} // namespace solidity::frontend
