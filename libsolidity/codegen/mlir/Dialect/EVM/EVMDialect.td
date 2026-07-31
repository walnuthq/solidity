//===- EVMDialect.td - EVM dialect definition --------------*- tablegen -*-===//
//
// This file is part of solidity.
//
// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0
//
//===----------------------------------------------------------------------===//
//
// The `evm` MLIR dialect - rung 3 of the solc dialect ladder:
//
//   sol dialect -> yul dialect -> evm dialect -> RISC-V (via LLVM)
//
// Design rule (ADR-003): upstream dialects are reused wherever EVM semantics
// coincide exactly - wrapping add/sub/mul and bitwise ops are arith.*,
// comparisons are arith.cmpi (+ext), control flow is cf.*, functions are
// func.func. This dialect defines ONLY the ops where EVM differs from or is
// absent in upstream MLIR:
//
//  - the semantic landmines (SolcRISCV.md §5): div/mod-by-zero yields 0,
//    sdiv MIN/-1 wraps, shifts >= 256 are defined, addmod/mulmod use wide
//    intermediates, exp wraps;
//  - the machine state: memory (byte-addressed, big-endian, observable
//    msize), storage/transient storage, calldata/code/returndata;
//  - the environment and the call family (lowered to ERHI host calls);
//  - halting terminators;
//  - the program container (creation + runtime artifacts).
//
//===----------------------------------------------------------------------===//

#ifndef EVM_DIALECT
#define EVM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// EVM dialect definition
//===----------------------------------------------------------------------===//

def EVM_Dialect : Dialect {
  let name = "evm";
  let summary = "Machine-level EVM semantics";
  let description = [{
    Machine-level EVM ops with EVM-exact semantics over signless i256 words,
    designed to compose with upstream arith/cf/func. The hub of the solc
    dialect ladder: entered from the yul dialect (and later from external
    EVM artifacts), exits to RISC-V via the LLVM dialect.
  }];

  let cppNamespace = "mlir::evm";
}

// The EVM word.
def EVM_Word : I<256>;

// EVM state spaces as distinct side-effect resources (see EVMDialect.h), so a
// pass can tell that storage, transient storage and memory do not alias.
def EVM_StorageResource : Resource<"::mlir::evm::StorageResource">;
def EVM_TransientResource : Resource<"::mlir::evm::TransientStorageResource">;
def EVM_MemoryResource : Resource<"::mlir::evm::MemoryResource">;

// 256-bit signless integer attribute for literals.
def EVM_WordAttr : SignlessIntegerAttrBase<I<256>,
    "256-bit signless integer attribute">;

//===----------------------------------------------------------------------===//
// Base op classes (mirroring the yul dialect's builder conventions)
//===----------------------------------------------------------------------===//

class EVM_Op<string mnemonic, list<Trait> traits = []> :
    Op<EVM_Dialect, mnemonic, traits>;

class EVM_NullaryOp<string mnemonic, list<Trait> traits = []> :
    EVM_Op<mnemonic, traits> {
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

class EVM_UnaryOp<string mnemonic, list<Trait> traits = []> :
    EVM_Op<mnemonic, traits> {
  let arguments = (ins EVM_Word:$operand);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$operand attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$operandValue), [{
      $_state.addOperands(operandValue);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

class EVM_BinaryOp<string mnemonic, list<Trait> traits = []> :
    EVM_Op<mnemonic, traits> {
  let arguments = (ins EVM_Word:$lhs, EVM_Word:$rhs);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhsValue, "::mlir::Value":$rhsValue), [{
      $_state.addOperands({lhsValue, rhsValue});
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

class EVM_TernaryOp<string mnemonic, list<Trait> traits = []> :
    EVM_Op<mnemonic, traits> {
  let arguments = (ins EVM_Word:$a, EVM_Word:$b, EVM_Word:$c);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$aValue, "::mlir::Value":$bValue,
                   "::mlir::Value":$cValue), [{
      $_state.addOperands({aValue, bValue, cValue});
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

// builtin(a, b, c) - statement form, no result (the copy family)
class EVM_TernaryStmtOp<string mnemonic, list<Trait> traits = []> :
    EVM_Op<mnemonic, traits> {
  let arguments = (ins EVM_Word:$a, EVM_Word:$b, EVM_Word:$c);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict";
}

//===----------------------------------------------------------------------===//
// Landmine arithmetic (EVM-exact semantics upstream arith cannot express)
//===----------------------------------------------------------------------===//

def EVM_DivOp : EVM_BinaryOp<"div", [Pure]> {
  let summary = "Unsigned division; x / 0 yields 0";
  let hasFolder = 1;
}
def EVM_SDivOp : EVM_BinaryOp<"sdiv", [Pure]> {
  let summary = "Signed division; x / 0 yields 0, MIN / -1 yields MIN";
  let hasFolder = 1;
}
def EVM_ModOp : EVM_BinaryOp<"mod", [Pure]> {
  let summary = "Unsigned modulo; x mod 0 yields 0";
  let hasFolder = 1;
}
def EVM_SModOp : EVM_BinaryOp<"smod", [Pure]> {
  let summary = "Signed modulo; x smod 0 yields 0";
  let hasFolder = 1;
}
def EVM_AddModOp : EVM_TernaryOp<"addmod", [Pure]> {
  let summary = "(a + b) mod c with 257-bit intermediate; c == 0 yields 0";
  let hasFolder = 1;
}
def EVM_MulModOp : EVM_TernaryOp<"mulmod", [Pure]> {
  let summary = "(a * b) mod c with 512-bit intermediate; c == 0 yields 0";
  let hasFolder = 1;
}
def EVM_ExpOp : EVM_BinaryOp<"exp", [Pure]> {
  let summary = "Wrapping exponentiation";
  let hasFolder = 1;
}
def EVM_ShlOp : EVM_BinaryOp<"shl", [Pure]> {
  let summary = "Shift left; shift >= 256 yields 0 (upstream shift would be poison)";
  let hasFolder = 1;
}
def EVM_ShrOp : EVM_BinaryOp<"shr", [Pure]> {
  let summary = "Logical shift right; shift >= 256 yields 0";
  let hasFolder = 1;
}
def EVM_SarOp : EVM_BinaryOp<"sar", [Pure]> {
  let summary = "Arithmetic shift right; shift >= 256 yields 0 or -1";
  let hasFolder = 1;
}
def EVM_ByteOp : EVM_BinaryOp<"byte", [Pure]> {
  let summary = "Nth byte of a word, big-endian indexing";
  let hasFolder = 1;
}
def EVM_SignExtendOp : EVM_BinaryOp<"signextend", [Pure]> {
  let summary = "Sign-extend from byte index";
  let hasFolder = 1;
}
def EVM_ClzOp : EVM_UnaryOp<"clz", [Pure]> {
  let summary = "Count of leading zero bits; clz(0) is 256";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Keccak and memory (byte-addressed, big-endian, zero-initialized)
//===----------------------------------------------------------------------===//

def EVM_Keccak256Op : EVM_BinaryOp<"keccak256", [MemoryEffects<[MemRead<EVM_MemoryResource>]>]> {
  let summary = "keccak256 over memory range (offset, length)";
}
def EVM_MLoadOp : EVM_UnaryOp<"mload", [MemoryEffects<[MemRead<EVM_MemoryResource>]>]> {
  let summary = "Load a word from memory (big-endian byte order)";
}
def EVM_MStoreOp : EVM_Op<"mstore", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$value);
  let assemblyFormat = "$offset `,` $value attr-dict";
}
def EVM_MStore8Op : EVM_Op<"mstore8", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$value);
  let assemblyFormat = "$offset `,` $value attr-dict";
}
def EVM_MCopyOp : EVM_TernaryStmtOp<"mcopy", [MemoryEffects<[MemRead<EVM_MemoryResource>, MemWrite<EVM_MemoryResource>]>]> {
  let summary = "Memory copy (dst, src, length)";
}
def EVM_MSizeOp : EVM_NullaryOp<"msize", [MemoryEffects<[MemRead<EVM_MemoryResource>]>]> {
  let summary = "Highest accessed memory index rounded to words (observable)";
}

//===----------------------------------------------------------------------===//
// Storage / transient storage (ERHI host state)
//===----------------------------------------------------------------------===//

def EVM_SLoadOp : EVM_UnaryOp<"sload", [MemoryEffects<[MemRead<EVM_StorageResource>]>]>;
def EVM_SStoreOp : EVM_Op<"sstore", [MemoryEffects<[MemWrite<EVM_StorageResource>]>]> {
  let arguments = (ins EVM_Word:$slot, EVM_Word:$value);
  let assemblyFormat = "$slot `,` $value attr-dict";
}
def EVM_TLoadOp : EVM_UnaryOp<"tload", [MemoryEffects<[MemRead<EVM_TransientResource>]>]>;
def EVM_TStoreOp : EVM_Op<"tstore", [MemoryEffects<[MemWrite<EVM_TransientResource>]>]> {
  let arguments = (ins EVM_Word:$slot, EVM_Word:$value);
  let assemblyFormat = "$slot `,` $value attr-dict";
}

//===----------------------------------------------------------------------===//
// Calldata / code / returndata
//===----------------------------------------------------------------------===//

def EVM_CallDataLoadOp : EVM_UnaryOp<"calldataload", [Pure]> {
  let summary = "Load a word from calldata; out-of-range bytes are zero";
}
def EVM_CallDataSizeOp : EVM_NullaryOp<"calldatasize", [Pure]>;
def EVM_CallDataCopyOp : EVM_TernaryStmtOp<"calldatacopy", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]> {
  let summary = "Copy calldata to memory (dst, src, length); zero-padded";
}
def EVM_CodeSizeOp : EVM_NullaryOp<"codesize", [Pure]>;
def EVM_CodeCopyOp : EVM_TernaryStmtOp<"codecopy", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]>;
def EVM_ReturnDataSizeOp : EVM_NullaryOp<"returndatasize">;
def EVM_ReturnDataCopyOp : EVM_TernaryStmtOp<"returndatacopy", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]> {
  let summary = "Copy returndata to memory; out-of-range REVERTS (landmine #10)";
}
def EVM_ExtCodeSizeOp : EVM_UnaryOp<"extcodesize">;
def EVM_ExtCodeHashOp : EVM_UnaryOp<"extcodehash">;
def EVM_ExtCodeCopyOp : EVM_Op<"extcodecopy"> {
  let arguments = (ins EVM_Word:$address, EVM_Word:$dst, EVM_Word:$src,
                       EVM_Word:$length);
  let assemblyFormat = "$address `,` $dst `,` $src `,` $length attr-dict";
}

//===----------------------------------------------------------------------===//
// Environment (per-tx constants become ERHI env-block loads; state-dependent
// reads become syscalls)
//===----------------------------------------------------------------------===//

def EVM_AddressOp : EVM_NullaryOp<"address", [Pure]>;
def EVM_CallerOp : EVM_NullaryOp<"caller", [Pure]>;
def EVM_CallValueOp : EVM_NullaryOp<"callvalue", [Pure]>;
def EVM_OriginOp : EVM_NullaryOp<"origin", [Pure]>;
def EVM_GasPriceOp : EVM_NullaryOp<"gasprice", [Pure]>;
def EVM_CoinbaseOp : EVM_NullaryOp<"coinbase", [Pure]>;
def EVM_TimestampOp : EVM_NullaryOp<"timestamp", [Pure]>;
def EVM_NumberOp : EVM_NullaryOp<"number", [Pure]>;
def EVM_PrevRandaoOp : EVM_NullaryOp<"prevrandao", [Pure]>;
def EVM_GasLimitOp : EVM_NullaryOp<"gaslimit", [Pure]>;
def EVM_ChainIdOp : EVM_NullaryOp<"chainid", [Pure]>;
def EVM_SelfBalanceOp : EVM_NullaryOp<"selfbalance">;
def EVM_BaseFeeOp : EVM_NullaryOp<"basefee", [Pure]>;
def EVM_BlobBaseFeeOp : EVM_NullaryOp<"blobbasefee", [Pure]>;
def EVM_GasOp : EVM_NullaryOp<"gas"> {
  let summary = "Remaining gas (gas-mode dependent, see proposal)";
}
def EVM_BalanceOp : EVM_UnaryOp<"balance">;
def EVM_BlockHashOp : EVM_UnaryOp<"blockhash", [Pure]>;
def EVM_BlobHashOp : EVM_UnaryOp<"blobhash", [Pure]>;

//===----------------------------------------------------------------------===//
// Logs
//===----------------------------------------------------------------------===//

def EVM_Log0Op : EVM_Op<"log0"> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def EVM_Log1Op : EVM_Op<"log1"> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length, EVM_Word:$topic1);
  let assemblyFormat = "$offset `,` $length `,` $topic1 attr-dict";
}
def EVM_Log2Op : EVM_Op<"log2"> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length, EVM_Word:$topic1,
                       EVM_Word:$topic2);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 attr-dict";
}
def EVM_Log3Op : EVM_Op<"log3"> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length, EVM_Word:$topic1,
                       EVM_Word:$topic2, EVM_Word:$topic3);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 `,` $topic3 attr-dict";
}
def EVM_Log4Op : EVM_Op<"log4"> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length, EVM_Word:$topic1,
                       EVM_Word:$topic2, EVM_Word:$topic3, EVM_Word:$topic4);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 `,` $topic3 `,` $topic4 attr-dict";
}

//===----------------------------------------------------------------------===//
// Call family (ERHI host re-entry)
//===----------------------------------------------------------------------===//

def EVM_CallOp : EVM_Op<"call"> {
  let arguments = (ins EVM_Word:$gas, EVM_Word:$target, EVM_Word:$value,
                       EVM_Word:$inOffset, EVM_Word:$inSize,
                       EVM_Word:$outOffset, EVM_Word:$outSize);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $value `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def EVM_CallCodeOp : EVM_Op<"callcode"> {
  let arguments = (ins EVM_Word:$gas, EVM_Word:$target, EVM_Word:$value,
                       EVM_Word:$inOffset, EVM_Word:$inSize,
                       EVM_Word:$outOffset, EVM_Word:$outSize);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $value `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def EVM_DelegateCallOp : EVM_Op<"delegatecall"> {
  let arguments = (ins EVM_Word:$gas, EVM_Word:$target,
                       EVM_Word:$inOffset, EVM_Word:$inSize,
                       EVM_Word:$outOffset, EVM_Word:$outSize);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def EVM_StaticCallOp : EVM_Op<"staticcall"> {
  let arguments = (ins EVM_Word:$gas, EVM_Word:$target,
                       EVM_Word:$inOffset, EVM_Word:$inSize,
                       EVM_Word:$outOffset, EVM_Word:$outSize);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def EVM_CreateOp : EVM_Op<"create"> {
  let arguments = (ins EVM_Word:$value, EVM_Word:$offset, EVM_Word:$length);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$value `,` $offset `,` $length attr-dict";
}
def EVM_Create2Op : EVM_Op<"create2"> {
  let arguments = (ins EVM_Word:$value, EVM_Word:$offset, EVM_Word:$length,
                       EVM_Word:$salt);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$value `,` $offset `,` $length `,` $salt attr-dict";
}

//===----------------------------------------------------------------------===//
// Object data segments and immutables (resolved by the ERHI host / linker)
//===----------------------------------------------------------------------===//

def EVM_DataOffsetOp : EVM_Op<"dataoffset", [Pure]> {
  let summary = "Offset of a named object/data segment";
  let arguments = (ins StrAttr:$segment);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$segment attr-dict";
}
def EVM_DataSizeOp : EVM_Op<"datasize", [Pure]> {
  let summary = "Size of a named object/data segment";
  let arguments = (ins StrAttr:$segment);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$segment attr-dict";
}
def EVM_DataCopyOp : EVM_TernaryStmtOp<"datacopy", [MemoryEffects<[MemWrite<EVM_MemoryResource>]>]> {
  let summary = "Copy an object/data segment to memory (dst, src, length)";
}
def EVM_MemoryGuardOp : EVM_Op<"memoryguard", [Pure]> {
  let summary = "Start of the contract's heap; memory below it is reserved";
  let description = [{
    Kept as an op rather than folded to its literal because a backend that
    wants memory of its own has to place it below this boundary and move the
    boundary up, which means rewriting the value the contract allocates from.
  }];
  let arguments = (ins EVM_WordAttr:$size);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$size attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::IntegerAttr":$sizeAttr), [{
      $_state.getOrAddProperties<Properties>().size = sizeAttr;
      $_state.addTypes(sizeAttr.getType());
    }]>
  ];
}

def EVM_LinkerSymbolOp : EVM_Op<"linkersymbol", [Pure]> {
  let summary = "Address of a named library, resolved at link time";
  let arguments = (ins StrAttr:$symbol);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$symbol attr-dict";
}
def EVM_LoadImmutableOp : EVM_Op<"loadimmutable", [Pure]> {
  let summary = "Load a named immutable (deployed-code objects)";
  let arguments = (ins StrAttr:$immutable_name);
  let results = (outs EVM_Word:$result);
  let assemblyFormat = "$immutable_name attr-dict";
}
def EVM_SetImmutableOp : EVM_Op<"setimmutable"> {
  let summary = "Patch a named immutable into deployed code (creation objects)";
  let arguments = (ins EVM_Word:$offset, StrAttr:$immutable_name,
                       EVM_Word:$value);
  let assemblyFormat = "$offset `,` $immutable_name `,` $value attr-dict";
}

//===----------------------------------------------------------------------===//
// Halting terminators
//===----------------------------------------------------------------------===//

def EVM_StopOp : EVM_Op<"stop", [Terminator]> {
  let assemblyFormat = "attr-dict";
}
def EVM_ReturnOp : EVM_Op<"return", [Terminator]> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def EVM_RevertOp : EVM_Op<"revert", [Terminator]> {
  let arguments = (ins EVM_Word:$offset, EVM_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def EVM_InvalidOp : EVM_Op<"invalid", [Terminator]> {
  let assemblyFormat = "attr-dict";
}
def EVM_SelfDestructOp : EVM_Op<"selfdestruct", [Terminator]> {
  let summary = "Selfdestruct (EIP-6780 semantics post-Cancun)";
  let arguments = (ins EVM_Word:$beneficiary);
  let assemblyFormat = "$beneficiary attr-dict";
}

//===----------------------------------------------------------------------===//
// Program container: the two EVM artifacts of a contract
//===----------------------------------------------------------------------===//

def EVM_ProgramOp : EVM_Op<"program", [
    IsolatedFromAbove,
    Symbol,
    NoTerminator
  ]> {
  let summary = "An EVM contract: creation and runtime artifacts";
  let description = [{
    Container for the two compiled artifacts of a contract. On EVM, creation
    code returns the runtime code; on RISC-V targets the two regions become
    the ERHI entry points evm_deploy and evm_call.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$creation, AnyRegion:$runtime);
  let assemblyFormat = "$sym_name `creation` $creation `runtime` $runtime attr-dict";
}

#endif // EVM_DIALECT
