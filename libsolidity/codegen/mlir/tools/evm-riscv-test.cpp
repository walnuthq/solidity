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
 * evm-riscv-test: M5 exit-criteria test - the RISC-V rung.
 *
 * Pipeline: Yul source -> yul dialect -> evm dialect -> LLVM dialect ->
 *  (a) host ORC JIT: every landmine function executed against an
 *      independent APInt-based oracle registered as the evm-rt symbols -
 *      the semantics differential;
 *  (b) RV32IM object emitted in-process, evm-rt cross-compiled with the
 *      LLVM-toolchain clang (_BitInt(256)), the whole thing linked into a
 *      static ELF with ld.lld and inspected - the artifact proof.
 *      (Execution under qemu-riscv32 runs in Linux CI; qemu user-mode is
 *      not available on macOS.)
 */

#include "EVMToLLVM.h"
#include "RISCVTargetEmitter.h"
#include "YulASTImporter.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#pragma GCC diagnostic pop

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using llvm::APInt;

//===----------------------------------------------------------------------===//
// Host evm-rt oracle: an independent APInt-based implementation of the
// runtime, registered with the JIT. (The RISC-V build links the C
// _BitInt(256) implementation instead - two implementations, one spec.)
//===----------------------------------------------------------------------===//

namespace
{
APInt loadW(void const* _p) { return APInt(256, llvm::ArrayRef<uint64_t>(static_cast<uint64_t const*>(_p), 4)); }
void storeW(void* _p, APInt const& _v) { std::memcpy(_p, _v.getRawData(), 32); }
} // anonymous namespace

extern "C"
{
	void __evm_rt_div(void* r, void const* a, void const* b)
	{
		APInt B = loadW(b);
		storeW(r, B.isZero() ? APInt(256, 0) : loadW(a).udiv(B));
	}
	void __evm_rt_sdiv(void* r, void const* a, void const* b)
	{
		APInt A = loadW(a), B = loadW(b);
		if (B.isZero())
			storeW(r, APInt(256, 0));
		else if (A.isMinSignedValue() && B.isAllOnes())
			storeW(r, A);
		else
			storeW(r, A.sdiv(B));
	}
	void __evm_rt_mod(void* r, void const* a, void const* b)
	{
		APInt B = loadW(b);
		storeW(r, B.isZero() ? APInt(256, 0) : loadW(a).urem(B));
	}
	void __evm_rt_smod(void* r, void const* a, void const* b)
	{
		APInt A = loadW(a), B = loadW(b);
		if (B.isZero() || (A.isMinSignedValue() && B.isAllOnes()))
			storeW(r, APInt(256, 0));
		else
			storeW(r, A.srem(B));
	}
	void __evm_rt_addmod(void* r, void const* a, void const* b, void const* c)
	{
		APInt C = loadW(c);
		if (C.isZero())
			return storeW(r, APInt(256, 0));
		storeW(r, (loadW(a).zext(512) + loadW(b).zext(512)).urem(C.zext(512)).trunc(256));
	}
	void __evm_rt_mulmod(void* r, void const* a, void const* b, void const* c)
	{
		APInt C = loadW(c);
		if (C.isZero())
			return storeW(r, APInt(256, 0));
		storeW(r, (loadW(a).zext(512) * loadW(b).zext(512)).urem(C.zext(512)).trunc(256));
	}
	void __evm_rt_exp(void* r, void const* base, void const* exponent)
	{
		APInt b = loadW(base), e = loadW(exponent), result(256, 1);
		while (!e.isZero())
		{
			if (e[0])
				result *= b;
			b *= b;
			e.lshrInPlace(1);
		}
		storeW(r, result);
	}
	void __evm_rt_byte(void* r, void const* index, void const* word)
	{
		APInt i = loadW(index);
		if (i.uge(32))
			return storeW(r, APInt(256, 0));
		storeW(r, loadW(word).lshr(8 * (31 - i.getZExtValue())) & APInt(256, 0xff));
	}
	void __evm_rt_signextend(void* r, void const* byteIndex, void const* word)
	{
		APInt b = loadW(byteIndex);
		APInt x = loadW(word);
		if (b.uge(31))
			return storeW(r, x);
		unsigned bit = 8 * unsigned(b.getZExtValue()) + 7;
		storeW(r, x.trunc(bit + 1).sext(256));
	}
}

//===----------------------------------------------------------------------===//
// Test program & vectors
//===----------------------------------------------------------------------===//

namespace
{

std::string const yulSource = R"({
	function div2(a, b) -> r { r := div(a, b) }
	function sdiv2(a, b) -> r { r := sdiv(a, b) }
	function mod2(a, b) -> r { r := mod(a, b) }
	function smod2(a, b) -> r { r := smod(a, b) }
	function exp2(a, b) -> r { r := exp(a, b) }
	function shl2(a, b) -> r { r := shl(a, b) }
	function shr2(a, b) -> r { r := shr(a, b) }
	function sar2(a, b) -> r { r := sar(a, b) }
	function byte2(a, b) -> r { r := byte(a, b) }
	function se2(a, b) -> r { r := signextend(a, b) }
	function am3(a, b, c) -> r { r := addmod(a, b, c) }
	function mm3(a, b, c) -> r { r := mulmod(a, b, c) }
	function addmul(a, b) -> r { r := add(mul(a, 3), b) }
})";

char const* MIN = "8000000000000000000000000000000000000000000000000000000000000000";
char const* MAX = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

struct Vector
{
	std::string func;
	std::vector<std::string> args; // hex, no 0x
	std::string expected;          // hex, no 0x
};

// Landmine-exact expectations (independently computed).
std::vector<Vector> vectors()
{
	return {
		{"div2", {"7", "2"}, "3"},
		{"div2", {"5", "0"}, "0"},                                              // div by zero = 0
		{"sdiv2", {MIN, MAX}, MIN},                                             // MIN / -1 wraps
		{"sdiv2", {"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa", "2"},
		 "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd"},  // -6 / 2 = -3
		{"mod2", {"9", "4"}, "1"},
		{"mod2", {"7", "0"}, "0"},                                              // mod by zero = 0
		{"smod2", {"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff9", "2"},
		 "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},  // -7 smod 2 = -1
		{"exp2", {"3", "c8"}, "c21a937a76f3432ffd73d97e447606b683ecf6f6e4a7ae225bfaff1eaaf8b0a1"},
		{"exp2", {"2", "100"}, "0"},                                            // 2^256 wraps to 0
		{"shl2", {"4", "1"}, "10"},
		{"shl2", {"12c", "1"}, "0"},                                            // shift 300 -> 0
		{"shr2", {"4", "100"}, "10"},
		{"shr2", {"12c", MAX}, "0"},
		{"sar2", {"ff", MIN}, MAX},                                             // sign fill
		{"sar2", {"12c", MIN}, MAX},                                            // shift >= 256, negative
		{"byte2", {"0", "aa00000000000000000000000000000000000000000000000000000000000000"}, "aa"},
		{"byte2", {"1f", "bb"}, "bb"},
		{"byte2", {"20", MAX}, "0"},                                            // index 32 -> 0
		{"se2", {"0", "ff"}, MAX},                                              // sign-extend byte 0
		{"se2", {"0", "7f"}, "7f"},
		{"am3", {MAX, MAX, "7"}, "2"},                                          // 257-bit intermediate
		{"am3", {"1", "2", "0"}, "0"},                                          // mod zero = 0
		{"mm3", {MAX, MAX, "7"}, "1"},                                          // 512-bit intermediate
		{"addmul", {"2", "5"}, "b"},
	};
}

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

void toBuffer(std::string const& _hex, uint64_t _out[4])
{
	APInt value(256, _hex, 16);
	std::memcpy(_out, value.getRawData(), 32);
}

int runCommand(std::string const& _cmd)
{
	return std::system(_cmd.c_str());
}

} // anonymous namespace

int main()
{
	bool ok = true;

	// Descend the ladder: Yul source -> yul -> evm -> LLVM dialect.
	mlir::MLIRContext ctx;
	std::string error;
	mlir::OwningOpRef<mlir::ModuleOp> yulModule =
		solidity::mlirgen::importYulSource("landmines.yul", yulSource, ctx, error);
	ok &= check(static_cast<bool>(yulModule), "import Yul landmine functions");
	if (!yulModule)
		return 1;

	unsigned remaining = solidity::mlirgen::promoteBlockLocalVars(*yulModule);
	ok &= check(remaining == 0, "variables promoted");

	mlir::OwningOpRef<mlir::ModuleOp> module = solidity::mlirgen::convertYulToEVM(*yulModule, error);
	ok &= check(static_cast<bool>(module), "yul -> evm conversion");
	if (!module)
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}

	if (!check(solidity::mlirgen::convertEVMToLLVM(*module, error), "evm -> LLVM dialect conversion"))
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}

	unsigned wrappers = solidity::mlirgen::addI256TestWrappers(*module);
	ok &= check(wrappers == 13, "test wrappers generated (" + std::to_string(wrappers) + "/13)");

	// (a) Host JIT differential against the APInt oracle.
	mlir::registerBuiltinDialectTranslation(ctx);
	mlir::registerLLVMDialectTranslation(ctx);
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	mlir::ExecutionEngineOptions engineOptions;
	// IR-level optimization via makeOptimizingTransformer crashes against
	// this LLVM build's analysis registration; backend codegen optimization
	// is sufficient for the differential.
	engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
	auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
	if (!check(static_cast<bool>(maybeEngine), "JIT engine creation"))
	{
		llvm::errs() << llvm::toString(maybeEngine.takeError()) << "\n";
		return 1;
	}
	auto& engine = *maybeEngine;

	engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
		llvm::orc::SymbolMap map;
		auto add = [&](char const* name, void* fn) {
			map[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn), llvm::JITSymbolFlags::Exported};
		};
		add("__evm_rt_div", reinterpret_cast<void*>(&__evm_rt_div));
		add("__evm_rt_sdiv", reinterpret_cast<void*>(&__evm_rt_sdiv));
		add("__evm_rt_mod", reinterpret_cast<void*>(&__evm_rt_mod));
		add("__evm_rt_smod", reinterpret_cast<void*>(&__evm_rt_smod));
		add("__evm_rt_addmod", reinterpret_cast<void*>(&__evm_rt_addmod));
		add("__evm_rt_mulmod", reinterpret_cast<void*>(&__evm_rt_mulmod));
		add("__evm_rt_exp", reinterpret_cast<void*>(&__evm_rt_exp));
		add("__evm_rt_byte", reinterpret_cast<void*>(&__evm_rt_byte));
		add("__evm_rt_signextend", reinterpret_cast<void*>(&__evm_rt_signextend));
		return map;
	});

	unsigned failures = 0;
	for (Vector const& vec: vectors())
	{
		uint64_t out[4] = {0, 0, 0, 0};
		uint64_t argBufs[3][4];
		void* outPtr = out;
		void* argPtrs[3];
		llvm::SmallVector<void*, 4> packed{&outPtr};
		for (size_t i = 0; i < vec.args.size(); ++i)
		{
			toBuffer(vec.args[i], argBufs[i]);
			argPtrs[i] = argBufs[i];
			packed.push_back(&argPtrs[i]);
		}

		llvm::Error err = engine->invokePacked("__test_" + vec.func, llvm::MutableArrayRef<void*>(packed));
		if (err)
		{
			std::cout << "[FAIL] invoke " << vec.func << ": " << llvm::toString(std::move(err)) << std::endl;
			++failures;
			continue;
		}

		uint64_t expected[4];
		toBuffer(vec.expected, expected);
		if (std::memcmp(out, expected, 32) != 0)
		{
			llvm::SmallString<80> got;
			APInt(256, llvm::ArrayRef<uint64_t>(out, 4)).toStringUnsigned(got, 16);
			std::cout << "[FAIL] " << vec.func << "(";
			for (auto const& a: vec.args)
				std::cout << "0x" << a << ",";
			std::cout << ") got 0x" << std::string(got.str()) << " want 0x" << vec.expected << std::endl;
			++failures;
		}
	}
	ok &= check(failures == 0, "JIT landmine differential: " + std::to_string(vectors().size() - failures) + "/"
								   + std::to_string(vectors().size()) + " vectors exact");

	// (b) RISC-V artifact: object, cross-compiled runtime, linked ELF.
	std::string scratch = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/evm-riscv-test";
	llvm::sys::fs::create_directories(scratch);
	std::string objectPath = scratch + "/contract.o";

	ok &= check(solidity::mlirgen::emitRISCVObject(*module, objectPath, error), "RV32IM object emitted in-process");
	if (!error.empty())
		std::cout << "  error: " << error << std::endl;

	std::string bin = LLVM_BIN_DIR;
	std::string lld = LLD_PATH;
	std::string rtDir = EVM_RT_DIR;
	std::string flags = " --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 ";

	ok &= check(
		runCommand(bin + "/clang" + flags + "-O2 -c " + rtDir + "/evm_rt.ll -o " + scratch + "/evm_rt.o") == 0,
		"evm-rt (i256/i512 LLVM IR) cross-compiles for rv32im");
	ok &= check(
		runCommand(bin + "/clang" + flags + "-c " + rtDir + "/start.S -o " + scratch + "/start.o") == 0,
		"start.S assembles");
	ok &= check(
		runCommand(
			lld + " " + scratch + "/contract.o " + scratch + "/evm_rt.o " + scratch + "/start.o -o " + scratch
			+ "/landmines.elf")
			== 0,
		"static RV32 ELF links with ld.lld");
	ok &= check(
		runCommand(bin + "/llvm-nm " + scratch + "/landmines.elf | grep -q __test_div2") == 0,
		"linked ELF exports the test entry points");

	std::cout << "\n--- rv32im disassembly (head) ---" << std::endl;
	runCommand(bin + "/llvm-objdump -d --mattr=+m " + scratch + "/landmines.elf | sed -n '6,24p'");
	std::cout << "(full ELF at " << scratch << "/landmines.elf; qemu-riscv32 execution runs in Linux CI)"
			  << std::endl;

	std::cout << (ok ? "\nAll evm-riscv checks passed." : "\nFAILURES in evm-riscv checks.") << std::endl;
	return ok ? 0 : 1;
}
