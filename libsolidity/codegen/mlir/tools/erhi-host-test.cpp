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
 * erhi-host-test: M6-lite exit criteria - a realistic deployed contract
 * (selector dispatcher, keccak mapping slots, storage, a loop, reverts)
 * compiled down the full ladder and EXECUTED against a native ERHI v0 host:
 * EVM memory arena (big-endian bytes), storage map, env block, calldata,
 * halting via host unwind. Assertions cover state changes, returndata, and
 * revert behavior across multiple calls into the same storage.
 *
 * Word ABI: 32-byte little-endian-limb i256 (LLVM memory layout). EVM
 * memory/calldata/returndata are big-endian byte arrays - the host converts
 * at the boundaries, exactly like the RISC-V target must (landmine #7).
 */

#include "EVMToLLVM.h"
#include "RISCVTargetEmitter.h"
#include "YulASTImporter.h"
#include "YulToEVM.h"

// Disable warnings for LLVM/MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#pragma GCC diagnostic pop

#include <libsolutil/Keccak256.h>

#include <array>
#include <csetjmp>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Word = std::array<uint8_t, 32>; // little-endian-limb ABI word

//===----------------------------------------------------------------------===//
// ERHI v0 native host state
//===----------------------------------------------------------------------===//

namespace
{
std::vector<uint8_t> g_memory;      // EVM memory: big-endian byte arena
std::map<Word, Word> g_storage;     // keyed/valued by ABI words
std::vector<uint8_t> g_calldata;    // big-endian bytes
std::vector<uint8_t> g_returndata;  // big-endian bytes
bool g_reverted = false;
Word g_caller{};
std::jmp_buf g_halt;

uint64_t wordToU64(uint8_t const* _w)
{
	uint64_t v = 0;
	for (int i = 7; i >= 0; --i)
		v = (v << 8) | _w[i];
	return v;
}

void u64ToWord(uint64_t _v, uint8_t* _w)
{
	std::memset(_w, 0, 32);
	for (int i = 0; i < 8; ++i)
		_w[i] = uint8_t(_v >> (8 * i));
}

void ensureMemory(uint64_t _end)
{
	if (g_memory.size() < _end)
		g_memory.resize(((_end + 31) / 32) * 32, 0);
}

} // anonymous namespace

extern "C"
{
	void __evm_mstore(void const* offset, void const* value)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		ensureMemory(off + 32);
		auto const* v = static_cast<uint8_t const*>(value);
		for (int i = 0; i < 32; ++i)
			g_memory[off + i] = v[31 - i]; // LE word -> BE memory bytes
	}

	void __evm_mload(void* result, void const* offset)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		ensureMemory(off + 32);
		auto* r = static_cast<uint8_t*>(result);
		for (int i = 0; i < 32; ++i)
			r[i] = g_memory[off + 31 - i];
	}

	void __evm_keccak256(void* result, void const* offset, void const* length)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		uint64_t len = wordToU64(static_cast<uint8_t const*>(length));
		ensureMemory(off + len);
		auto hash = solidity::util::keccak256(
			solidity::bytesConstRef(g_memory.data() + off, static_cast<size_t>(len)));
		auto* r = static_cast<uint8_t*>(result);
		for (int i = 0; i < 32; ++i)
			r[i] = hash.data()[31 - i]; // BE digest -> LE word
	}

	void __evm_calldataload(void* result, void const* offset)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		uint8_t be[32] = {0};
		for (uint64_t i = 0; i < 32; ++i)
			if (off + i < g_calldata.size())
				be[i] = g_calldata[off + i]; // out-of-range zero-padded
		auto* r = static_cast<uint8_t*>(result);
		for (int i = 0; i < 32; ++i)
			r[i] = be[31 - i];
	}

	void __evm_calldatasize(void* result) { u64ToWord(g_calldata.size(), static_cast<uint8_t*>(result)); }

	void __evm_caller(void* result) { std::memcpy(result, g_caller.data(), 32); }

	void __evm_sload(void* result, void const* slot)
	{
		Word key;
		std::memcpy(key.data(), slot, 32);
		auto it = g_storage.find(key);
		if (it == g_storage.end())
			std::memset(result, 0, 32);
		else
			std::memcpy(result, it->second.data(), 32);
	}

	void __evm_sstore(void const* slot, void const* value)
	{
		Word key, val;
		std::memcpy(key.data(), slot, 32);
		std::memcpy(val.data(), value, 32);
		g_storage[key] = val;
	}

	void __evm_return(void const* offset, void const* length)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		uint64_t len = wordToU64(static_cast<uint8_t const*>(length));
		ensureMemory(off + len);
		g_returndata.assign(g_memory.begin() + off, g_memory.begin() + off + len);
		g_reverted = false;
		std::longjmp(g_halt, 1);
	}

	void __evm_revert(void const* offset, void const* length)
	{
		uint64_t off = wordToU64(static_cast<uint8_t const*>(offset));
		uint64_t len = wordToU64(static_cast<uint8_t const*>(length));
		ensureMemory(off + len);
		g_returndata.assign(g_memory.begin() + off, g_memory.begin() + off + len);
		g_reverted = true;
		std::longjmp(g_halt, 1);
	}

	void __evm_stop()
	{
		g_returndata.clear();
		g_reverted = false;
		std::longjmp(g_halt, 1);
	}
}

//===----------------------------------------------------------------------===//
// The contract: a realistic deployed object
//===----------------------------------------------------------------------===//

namespace
{

std::string const tokenYul = R"({
	if lt(calldatasize(), 4) { revert(0, 0) }
	switch shr(224, calldataload(0))
	case 0x70a08231 /* balanceOf(address) */ {
		mstore(0, calldataload(4))
		mstore(32, 0)
		mstore(0, sload(keccak256(0, 64)))
		return(0, 32)
	}
	case 0xa9059cbb /* transfer(address,uint256) */ {
		let to := calldataload(4)
		let amount := calldataload(36)
		mstore(0, caller())
		mstore(32, 0)
		let fromSlot := keccak256(0, 64)
		let fromBalance := sload(fromSlot)
		if lt(fromBalance, amount) { revert(0, 0) }
		sstore(fromSlot, sub(fromBalance, amount))
		mstore(0, to)
		mstore(32, 0)
		let toSlot := keccak256(0, 64)
		sstore(toSlot, add(sload(toSlot), amount))
		mstore(0, 1)
		return(0, 32)
	}
	case 0x0d15fd77 /* sumTo(uint256) - loop-carried state */ {
		let n := calldataload(4)
		let total := 0
		for { let i := 0 } lt(i, n) { i := add(i, 1) } {
			total := add(total, i)
		}
		mstore(0, total)
		return(0, 32)
	}
	default { revert(0, 0) }
})";

bool check(bool _ok, std::string const& _what)
{
	std::cout << (_ok ? "[PASS] " : "[FAIL] ") << _what << std::endl;
	return _ok;
}

/// Mapping slot keccak256(BE(key) . BE(0)) as an ABI word, replicating what
/// the compiled contract computes in its memory arena.
Word mappingSlot(Word const& _keyWord)
{
	uint8_t buffer[64] = {0};
	for (int i = 0; i < 32; ++i)
		buffer[i] = _keyWord[31 - i]; // LE word -> BE bytes
	auto hash = solidity::util::keccak256(solidity::bytesConstRef(buffer, 64));
	Word slot;
	for (int i = 0; i < 32; ++i)
		slot[i] = hash.data()[31 - i];
	return slot;
}

Word wordFromU64(uint64_t _v)
{
	Word w{};
	for (int i = 0; i < 8; ++i)
		w[i] = uint8_t(_v >> (8 * i));
	return w;
}

std::vector<uint8_t> encodeCall(uint32_t _selector, std::vector<Word> const& _args)
{
	std::vector<uint8_t> data;
	for (int i = 3; i >= 0; --i)
		data.push_back(uint8_t(_selector >> (8 * i)));
	for (Word const& arg: _args)
		for (int i = 31; i >= 0; --i)
			data.push_back(arg[i]); // LE word -> BE calldata
	return data;
}

uint64_t returndataAsU64()
{
	uint64_t v = 0;
	for (uint8_t byte: g_returndata)
		v = (v << 8) | byte;
	return v;
}

} // anonymous namespace

int main()
{
	bool ok = true;

	// Descend the ladder.
	mlir::MLIRContext ctx;
	std::string error;
	mlir::OwningOpRef<mlir::ModuleOp> yulModule =
		solidity::mlirgen::importYulSource("token.yul", tokenYul, ctx, error);
	ok &= check(static_cast<bool>(yulModule), "import token deployed object");
	if (!yulModule)
		return 1;
	solidity::mlirgen::promoteBlockLocalVars(*yulModule);

	mlir::OwningOpRef<mlir::ModuleOp> module = solidity::mlirgen::convertYulToEVM(*yulModule, error);
	if (!check(static_cast<bool>(module), "yul -> evm (dispatcher, loop, storage)"))
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}
	if (!check(solidity::mlirgen::convertEVMToLLVM(*module, error), "evm -> LLVM (ERHI v0 host calls)"))
	{
		std::cout << "  error: " << error << std::endl;
		return 1;
	}

	// Also prove the artifact leg: the same module emits an RV32IM object.
	std::string scratch = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp");
	ok &= check(
		solidity::mlirgen::emitRISCVObject(*module, scratch + "/erhi-token.o", error),
		"token compiles to an RV32IM object");

	// JIT with the native ERHI host.
	mlir::registerBuiltinDialectTranslation(ctx);
	mlir::registerLLVMDialectTranslation(ctx);
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	mlir::ExecutionEngineOptions engineOptions;
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
		add("__evm_mstore", reinterpret_cast<void*>(&__evm_mstore));
		add("__evm_mload", reinterpret_cast<void*>(&__evm_mload));
		add("__evm_keccak256", reinterpret_cast<void*>(&__evm_keccak256));
		add("__evm_calldataload", reinterpret_cast<void*>(&__evm_calldataload));
		add("__evm_calldatasize", reinterpret_cast<void*>(&__evm_calldatasize));
		add("__evm_caller", reinterpret_cast<void*>(&__evm_caller));
		add("__evm_sload", reinterpret_cast<void*>(&__evm_sload));
		add("__evm_sstore", reinterpret_cast<void*>(&__evm_sstore));
		add("__evm_return", reinterpret_cast<void*>(&__evm_return));
		add("__evm_revert", reinterpret_cast<void*>(&__evm_revert));
		add("__evm_stop", reinterpret_cast<void*>(&__evm_stop));
		return map;
	});

	Word alice = wordFromU64(0xA11CE);
	Word bob = wordFromU64(0xB0B);
	g_caller = alice;

	auto call = [&](std::vector<uint8_t> _calldata) -> bool {
		g_calldata = std::move(_calldata);
		g_memory.clear();
		g_returndata.clear();
		g_reverted = false;
		if (setjmp(g_halt) == 0)
		{
			llvm::SmallVector<void*, 1> packed;
			llvm::Error err = engine->invokePacked("__entry", llvm::MutableArrayRef<void*>(packed));
			if (err)
			{
				std::cout << "  invoke error: " << llvm::toString(std::move(err)) << std::endl;
				return false;
			}
			return false; // fell off without halting - should not happen
		}
		return true;
	};

	// Mint: prepare storage balances[alice] = 100 the way the contract
	// computes the slot (keccak over the BE key/slot pair).
	g_storage[mappingSlot(alice)] = wordFromU64(100);

	// balanceOf(alice) == 100
	ok &= check(call(encodeCall(0x70a08231, {alice})), "balanceOf(alice) halts");
	ok &= check(!g_reverted && returndataAsU64() == 100, "balanceOf(alice) == 100");

	// transfer(bob, 60) succeeds
	ok &= check(call(encodeCall(0xa9059cbb, {bob, wordFromU64(60)})), "transfer(bob, 60) halts");
	ok &= check(!g_reverted && returndataAsU64() == 1, "transfer(bob, 60) returns true");
	ok &= check(g_storage[mappingSlot(alice)] == wordFromU64(40), "balances[alice] == 40 after transfer");
	ok &= check(g_storage[mappingSlot(bob)] == wordFromU64(60), "balances[bob] == 60 after transfer");

	// transfer(bob, 1000) reverts, state unchanged
	ok &= check(call(encodeCall(0xa9059cbb, {bob, wordFromU64(1000)})), "transfer(bob, 1000) halts");
	ok &= check(g_reverted, "insufficient balance reverts");
	ok &= check(g_storage[mappingSlot(alice)] == wordFromU64(40), "revert leaves balances[alice] intact");

	// sumTo(10) == 45 - the loop-carried-variable proof under execution
	ok &= check(call(encodeCall(0x0d15fd77, {wordFromU64(10)})), "sumTo(10) halts");
	ok &= check(!g_reverted && returndataAsU64() == 45, "sumTo(10) == 45 (loop-carried SSA)");

	// Unknown selector reverts.
	ok &= check(call(encodeCall(0xdeadbeef, {})), "unknown selector halts");
	ok &= check(g_reverted, "unknown selector reverts");

	std::cout << (ok ? "\nAll ERHI host-execution checks passed." : "\nFAILURES in ERHI host-execution checks.")
			  << std::endl;
	return ok ? 0 : 1;
}
