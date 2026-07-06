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
 * RISC-V target emitter implementation.
 */

#include "RISCVTargetEmitter.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#pragma GCC diagnostic pop

#include <string>

using namespace mlir;

unsigned solidity::mlirgen::addI256TestWrappers(mlir::ModuleOp _module)
{
	MLIRContext* ctx = _module.getContext();
	OpBuilder builder(ctx);
	auto i256 = IntegerType::get(ctx, 256);
	auto ptrType = LLVM::LLVMPointerType::get(ctx);
	Location loc = _module.getLoc();

	SmallVector<LLVM::LLVMFuncOp, 8> targets;
	for (auto func: _module.getOps<LLVM::LLVMFuncOp>())
	{
		if (func.isExternal() || func.getName().starts_with("__"))
			continue;
		LLVM::LLVMFunctionType type = func.getFunctionType();
		if (type.getReturnType() != i256)
			continue;
		bool allWords = true;
		for (Type param: type.getParams())
			if (param != i256)
				allWords = false;
		if (allWords)
			targets.push_back(func);
	}

	for (LLVM::LLVMFuncOp target: targets)
	{
		unsigned numArgs = target.getFunctionType().getParams().size();
		SmallVector<Type, 5> wrapperParams(numArgs + 1, ptrType);
		auto wrapperType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), wrapperParams);

		builder.setInsertionPointToEnd(_module.getBody());
		auto wrapper =
			builder.create<LLVM::LLVMFuncOp>(loc, ("__test_" + target.getName()).str(), wrapperType);
		Block* body = wrapper.addEntryBlock(builder);
		builder.setInsertionPointToStart(body);

		SmallVector<Value, 4> loadedArgs;
		for (unsigned i = 0; i < numArgs; ++i)
			loadedArgs.push_back(builder.create<LLVM::LoadOp>(loc, i256, body->getArgument(i + 1)));
		auto call = builder.create<LLVM::CallOp>(loc, target, loadedArgs);
		builder.create<LLVM::StoreOp>(loc, call.getResult(), body->getArgument(0));
		builder.create<LLVM::ReturnOp>(loc, ValueRange{});
	}
	return targets.size();
}

bool solidity::mlirgen::emitRISCVObject(mlir::ModuleOp _module, std::string const& _objectPath, std::string& _error)
{
	// Register the MLIR->LLVM-IR translation interfaces.
	registerBuiltinDialectTranslation(*_module.getContext());
	registerLLVMDialectTranslation(*_module.getContext());

	llvm::LLVMContext llvmCtx;
	std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(_module, llvmCtx, "evm-contract");
	if (!llvmModule)
	{
		_error = "failed to translate MLIR to LLVM IR";
		return false;
	}

	LLVMInitializeRISCVTargetInfo();
	LLVMInitializeRISCVTarget();
	LLVMInitializeRISCVTargetMC();
	LLVMInitializeRISCVAsmPrinter();

	std::string const triple = "riscv32-unknown-elf";
	std::string lookupError;
	llvm::Target const* target = llvm::TargetRegistry::lookupTarget(triple, lookupError);
	if (!target)
	{
		_error = "RISC-V target unavailable: " + lookupError;
		return false;
	}

	llvm::TargetOptions options;
	options.MCOptions.ABIName = "ilp32";
	std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
		llvm::Triple(triple), /*CPU=*/"generic-rv32", /*Features=*/"+m", options, llvm::Reloc::Static,
		std::nullopt, llvm::CodeGenOptLevel::Default));
	if (!machine)
	{
		_error = "failed to create RISC-V target machine";
		return false;
	}

	llvmModule->setTargetTriple(llvm::Triple(triple));
	llvmModule->setDataLayout(machine->createDataLayout());

	std::error_code ec;
	llvm::raw_fd_ostream out(_objectPath, ec, llvm::sys::fs::OF_None);
	if (ec)
	{
		_error = "cannot open output file: " + ec.message();
		return false;
	}

	llvm::legacy::PassManager passManager;
	if (machine->addPassesToEmitFile(passManager, out, nullptr, llvm::CodeGenFileType::ObjectFile))
	{
		_error = "RISC-V target cannot emit object files";
		return false;
	}
	passManager.run(*llvmModule);
	out.flush();
	return true;
}
