//===- YulDialect.td - Yul dialect definition --------------*- tablegen -*-===//
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
// The `yul` MLIR dialect - rung 2 of the solc dialect ladder:
//
//   sol dialect  ->  yul dialect  ->  evm dialect  ->  RISC-V (via LLVM)
//
// Design (ADR-003): ops mirror Yul builtins 1:1 over signless i256 words with
// Yul/EVM-exact semantics (e.g. yul.div by zero yields 0). This makes the
// Yul-text emitter a trivial printer and a future libyul-AST importer a
// trivial walk. Reuse of upstream `arith` happens one rung down, in the
// `evm` dialect, where MLIR's canonicalization infrastructure earns its keep.
//
// Design (ADR-004): control flow uses region ops (yul.if / yul.for) with
// yul.break / yul.continue / yul.leave terminators, mirroring Yul's
// structured control flow. Regions carry NoTerminator so that plain
// fall-through blocks remain valid while explicit terminators are allowed
// at block ends. Mutable Yul variables are not modeled at this rung yet;
// SSA values + memory ops cover the scaffold (yul.var/yul.assign are a
// planned M1 follow-up; see README).
//
//===----------------------------------------------------------------------===//

#ifndef YUL_DIALECT
#define YUL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Yul dialect definition
//===----------------------------------------------------------------------===//

def Yul_Dialect : Dialect {
  let name = "yul";
  let summary = "A faithful MLIR model of the Yul intermediate language";
  let description = [{
    This dialect models Yul ("Yul with SSA values"): 256-bit signless words,
    ops 1:1 with Yul builtins carrying Yul/EVM-exact semantics, structured
    control flow (if/for with break/continue/leave), and functions.
    It is the second rung of the solc MLIR dialect ladder
    (sol -> yul -> evm -> RISC-V).
  }];

  let cppNamespace = "mlir::yul";
  let hasConstantMaterializer = 1;

  // Hand-written type parsing/printing for the (single, parameterless)
  // !yul.varref type - mirrors the Sol dialect's approach of hand-rolled
  // types (tablegen typedef generation is avoided; see Dialect/Sol notes).
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

//===----------------------------------------------------------------------===//
// Types and attributes
//===----------------------------------------------------------------------===//

// The Yul/EVM word: a 256-bit signless integer (builtin type; no custom
// dialect type needed).
def Yul_Word : I<256>;

// 256-bit signless integer attribute for literals.
def Yul_WordAttr : SignlessIntegerAttrBase<I<256>,
    "256-bit signless integer attribute">;

// Reference to a mutable Yul variable (hand-written C++ type, see
// YulDialect.h). Yul locals are mutable; modeling them as first-class
// variable ops (clang/flang style) makes the libyul-AST importer a trivial
// walk with no SSA construction. An SSA-promotion pass (mem2reg style)
// runs before lowering to the evm dialect.
def Yul_VarRef : Type<CPred<"::llvm::isa<::mlir::yul::VarRefType>($_self)">,
                      "yul variable reference",
                      "::mlir::yul::VarRefType"> {
  let builderCall = "$_builder.getType<::mlir::yul::VarRefType>()";
}

//===----------------------------------------------------------------------===//
// Base op classes
//===----------------------------------------------------------------------===//

class Yul_Op<string mnemonic, list<Trait> traits = []> :
    Op<Yul_Dialect, mnemonic, traits>;

// value = builtin()
class Yul_NullaryOp<string mnemonic, list<Trait> traits = []> :
    Yul_Op<mnemonic, traits> {
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

// value = builtin(operand)
class Yul_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Yul_Op<mnemonic, traits> {
  let arguments = (ins Yul_Word:$operand);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$operand attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$operandValue), [{
      $_state.addOperands(operandValue);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

// value = builtin(lhs, rhs)
class Yul_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Yul_Op<mnemonic, traits> {
  let arguments = (ins Yul_Word:$lhs, Yul_Word:$rhs);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhsValue, "::mlir::Value":$rhsValue), [{
      $_state.addOperands({lhsValue, rhsValue});
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

// value = builtin(a, b, c)
class Yul_TernaryOp<string mnemonic, list<Trait> traits = []> :
    Yul_Op<mnemonic, traits> {
  let arguments = (ins Yul_Word:$a, Yul_Word:$b, Yul_Word:$c);
  let results = (outs Yul_Word:$result);
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
class Yul_TernaryStmtOp<string mnemonic, list<Trait> traits = []> :
    Yul_Op<mnemonic, traits> {
  let arguments = (ins Yul_Word:$a, Yul_Word:$b, Yul_Word:$c);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict";
}

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

def Yul_ConstOp : Yul_Op<"const", [Pure, ConstantLike]> {
  let summary = "A 256-bit Yul literal";
  let arguments = (ins Yul_WordAttr:$value);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "::mlir::IntegerAttr":$valueAttr), [{
      $_state.getOrAddProperties<Properties>().value = valueAttr;
      $_state.addTypes(valueAttr.getType());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Arithmetic builtins (Yul/EVM-exact semantics; all wrap mod 2^256)
//===----------------------------------------------------------------------===//

def Yul_AddOp : Yul_BinaryOp<"add", [Pure, Commutative]> {
  let summary = "Wrapping addition";
}
def Yul_SubOp : Yul_BinaryOp<"sub", [Pure]> {
  let summary = "Wrapping subtraction";
}
def Yul_MulOp : Yul_BinaryOp<"mul", [Pure, Commutative]> {
  let summary = "Wrapping multiplication";
}
def Yul_DivOp : Yul_BinaryOp<"div", [Pure]> {
  let summary = "Unsigned division; x / 0 yields 0 (EVM semantics)";
}
def Yul_SDivOp : Yul_BinaryOp<"sdiv", [Pure]> {
  let summary = "Signed division; x / 0 yields 0, MIN / -1 yields MIN";
}
def Yul_ModOp : Yul_BinaryOp<"mod", [Pure]> {
  let summary = "Unsigned modulo; x mod 0 yields 0 (EVM semantics)";
}
def Yul_SModOp : Yul_BinaryOp<"smod", [Pure]> {
  let summary = "Signed modulo; x smod 0 yields 0 (EVM semantics)";
}
def Yul_ExpOp : Yul_BinaryOp<"exp", [Pure]> {
  let summary = "Wrapping exponentiation";
}
def Yul_SignExtendOp : Yul_BinaryOp<"signextend", [Pure]> {
  let summary = "Sign-extend from byte index (EVM SIGNEXTEND)";
}
def Yul_AddModOp : Yul_TernaryOp<"addmod", [Pure]> {
  let summary = "(a + b) mod c with 257-bit intermediate; c == 0 yields 0";
}
def Yul_MulModOp : Yul_TernaryOp<"mulmod", [Pure]> {
  let summary = "(a * b) mod c with 512-bit intermediate; c == 0 yields 0";
}

//===----------------------------------------------------------------------===//
// Comparison builtins (produce 0 or 1 as a word)
//===----------------------------------------------------------------------===//

def Yul_LtOp : Yul_BinaryOp<"lt", [Pure]> {
  let summary = "Unsigned less-than; yields 0 or 1";
}
def Yul_GtOp : Yul_BinaryOp<"gt", [Pure]> {
  let summary = "Unsigned greater-than; yields 0 or 1";
}
def Yul_SLtOp : Yul_BinaryOp<"slt", [Pure]> {
  let summary = "Signed less-than; yields 0 or 1";
}
def Yul_SGtOp : Yul_BinaryOp<"sgt", [Pure]> {
  let summary = "Signed greater-than; yields 0 or 1";
}
def Yul_EqOp : Yul_BinaryOp<"eq", [Pure, Commutative]> {
  let summary = "Equality; yields 0 or 1";
}
def Yul_IsZeroOp : Yul_UnaryOp<"iszero", [Pure]> {
  let summary = "Logical negation of a word; yields 0 or 1";
}

//===----------------------------------------------------------------------===//
// Bitwise builtins
//===----------------------------------------------------------------------===//

def Yul_AndOp : Yul_BinaryOp<"and", [Pure, Commutative]> {
  let summary = "Bitwise and";
}
def Yul_OrOp : Yul_BinaryOp<"or", [Pure, Commutative]> {
  let summary = "Bitwise or";
}
def Yul_XorOp : Yul_BinaryOp<"xor", [Pure, Commutative]> {
  let summary = "Bitwise xor";
}
def Yul_NotOp : Yul_UnaryOp<"not", [Pure]> {
  let summary = "Bitwise negation";
}
def Yul_ByteOp : Yul_BinaryOp<"byte", [Pure]> {
  let summary = "Nth byte of a word, big-endian indexing (EVM BYTE)";
}
def Yul_ShlOp : Yul_BinaryOp<"shl", [Pure]> {
  let summary = "Shift left; shift >= 256 yields 0 (EVM semantics; operands are (shift, value))";
}
def Yul_ShrOp : Yul_BinaryOp<"shr", [Pure]> {
  let summary = "Logical shift right; shift >= 256 yields 0 (operands are (shift, value))";
}
def Yul_SarOp : Yul_BinaryOp<"sar", [Pure]> {
  let summary = "Arithmetic shift right; shift >= 256 yields 0 or -1 (operands are (shift, value))";
}

//===----------------------------------------------------------------------===//
// Keccak
//===----------------------------------------------------------------------===//

def Yul_Keccak256Op : Yul_BinaryOp<"keccak256"> {
  let summary = "keccak256 hash of memory range (offset, length)";
}

//===----------------------------------------------------------------------===//
// Memory builtins
//===----------------------------------------------------------------------===//

def Yul_MLoadOp : Yul_UnaryOp<"mload"> {
  let summary = "Load a word from memory (big-endian byte order)";
}
def Yul_MStoreOp : Yul_Op<"mstore"> {
  let summary = "Store a word to memory (big-endian byte order)";
  let arguments = (ins Yul_Word:$offset, Yul_Word:$value);
  let assemblyFormat = "$offset `,` $value attr-dict";
}
def Yul_MStore8Op : Yul_Op<"mstore8"> {
  let summary = "Store a single byte to memory";
  let arguments = (ins Yul_Word:$offset, Yul_Word:$value);
  let assemblyFormat = "$offset `,` $value attr-dict";
}
def Yul_MCopyOp : Yul_TernaryStmtOp<"mcopy"> {
  let summary = "Memory-to-memory copy (dst, src, length); EIP-5656";
}
def Yul_MSizeOp : Yul_NullaryOp<"msize"> {
  let summary = "Highest accessed memory index rounded to words (observable!)";
}

//===----------------------------------------------------------------------===//
// Storage and transient storage builtins
//===----------------------------------------------------------------------===//

def Yul_SLoadOp : Yul_UnaryOp<"sload"> {
  let summary = "Load a word from storage";
}
def Yul_SStoreOp : Yul_Op<"sstore"> {
  let summary = "Store a word to storage";
  let arguments = (ins Yul_Word:$slot, Yul_Word:$value);
  let assemblyFormat = "$slot `,` $value attr-dict";
}
def Yul_TLoadOp : Yul_UnaryOp<"tload"> {
  let summary = "Load a word from transient storage (EIP-1153)";
}
def Yul_TStoreOp : Yul_Op<"tstore"> {
  let summary = "Store a word to transient storage (EIP-1153)";
  let arguments = (ins Yul_Word:$slot, Yul_Word:$value);
  let assemblyFormat = "$slot `,` $value attr-dict";
}

//===----------------------------------------------------------------------===//
// Calldata / code / returndata builtins
//===----------------------------------------------------------------------===//

def Yul_CallDataLoadOp : Yul_UnaryOp<"calldataload"> {
  let summary = "Load a word from calldata; out-of-range bytes are zero";
}
def Yul_CallDataSizeOp : Yul_NullaryOp<"calldatasize">;
def Yul_CallDataCopyOp : Yul_TernaryStmtOp<"calldatacopy"> {
  let summary = "Copy calldata to memory (dst, src, length); zero-padded";
}
def Yul_CodeSizeOp : Yul_NullaryOp<"codesize">;
def Yul_CodeCopyOp : Yul_TernaryStmtOp<"codecopy"> {
  let summary = "Copy own code to memory (dst, src, length); zero-padded";
}
def Yul_ReturnDataSizeOp : Yul_NullaryOp<"returndatasize">;
def Yul_ReturnDataCopyOp : Yul_TernaryStmtOp<"returndatacopy"> {
  let summary = "Copy returndata to memory (dst, src, length); OOB reverts (unlike calldatacopy!)";
}
def Yul_ExtCodeSizeOp : Yul_UnaryOp<"extcodesize">;
def Yul_ExtCodeHashOp : Yul_UnaryOp<"extcodehash">;
def Yul_ExtCodeCopyOp : Yul_Op<"extcodecopy"> {
  let summary = "Copy external code to memory (addr, dst, src, length)";
  let arguments = (ins Yul_Word:$address, Yul_Word:$dst, Yul_Word:$src,
                       Yul_Word:$length);
  let assemblyFormat = "$address `,` $dst `,` $src `,` $length attr-dict";
}

//===----------------------------------------------------------------------===//
// Environment builtins (per-call/per-tx constants and state reads)
//===----------------------------------------------------------------------===//

def Yul_AddressOp : Yul_NullaryOp<"address">;
def Yul_CallerOp : Yul_NullaryOp<"caller">;
def Yul_CallValueOp : Yul_NullaryOp<"callvalue">;
def Yul_OriginOp : Yul_NullaryOp<"origin">;
def Yul_GasPriceOp : Yul_NullaryOp<"gasprice">;
def Yul_CoinbaseOp : Yul_NullaryOp<"coinbase">;
def Yul_TimestampOp : Yul_NullaryOp<"timestamp">;
def Yul_NumberOp : Yul_NullaryOp<"number">;
def Yul_PrevRandaoOp : Yul_NullaryOp<"prevrandao">;
def Yul_GasLimitOp : Yul_NullaryOp<"gaslimit">;
def Yul_ChainIdOp : Yul_NullaryOp<"chainid">;
def Yul_SelfBalanceOp : Yul_NullaryOp<"selfbalance">;
def Yul_BaseFeeOp : Yul_NullaryOp<"basefee">;
def Yul_BlobBaseFeeOp : Yul_NullaryOp<"blobbasefee">;
def Yul_GasOp : Yul_NullaryOp<"gas"> {
  let summary = "Remaining gas (observable - see gas-mode notes in the proposal)";
}
def Yul_BalanceOp : Yul_UnaryOp<"balance">;
def Yul_BlockHashOp : Yul_UnaryOp<"blockhash">;
def Yul_BlobHashOp : Yul_UnaryOp<"blobhash">;

//===----------------------------------------------------------------------===//
// Log builtins
//===----------------------------------------------------------------------===//

def Yul_Log0Op : Yul_Op<"log0"> {
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def Yul_Log1Op : Yul_Op<"log1"> {
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length, Yul_Word:$topic1);
  let assemblyFormat = "$offset `,` $length `,` $topic1 attr-dict";
}
def Yul_Log2Op : Yul_Op<"log2"> {
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length, Yul_Word:$topic1,
                       Yul_Word:$topic2);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 attr-dict";
}
def Yul_Log3Op : Yul_Op<"log3"> {
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length, Yul_Word:$topic1,
                       Yul_Word:$topic2, Yul_Word:$topic3);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 `,` $topic3 attr-dict";
}
def Yul_Log4Op : Yul_Op<"log4"> {
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length, Yul_Word:$topic1,
                       Yul_Word:$topic2, Yul_Word:$topic3, Yul_Word:$topic4);
  let assemblyFormat = "$offset `,` $length `,` $topic1 `,` $topic2 `,` $topic3 `,` $topic4 attr-dict";
}

//===----------------------------------------------------------------------===//
// Call-family builtins
//===----------------------------------------------------------------------===//

def Yul_CallOp : Yul_Op<"call"> {
  let summary = "External call; yields 0 on failure, 1 on success";
  let arguments = (ins Yul_Word:$gas, Yul_Word:$target, Yul_Word:$value,
                       Yul_Word:$inOffset, Yul_Word:$inSize,
                       Yul_Word:$outOffset, Yul_Word:$outSize);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $value `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def Yul_CallCodeOp : Yul_Op<"callcode"> {
  let summary = "callcode (legacy); yields 0 on failure, 1 on success";
  let arguments = (ins Yul_Word:$gas, Yul_Word:$target, Yul_Word:$value,
                       Yul_Word:$inOffset, Yul_Word:$inSize,
                       Yul_Word:$outOffset, Yul_Word:$outSize);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $value `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def Yul_DelegateCallOp : Yul_Op<"delegatecall"> {
  let summary = "Delegatecall; yields 0 on failure, 1 on success";
  let arguments = (ins Yul_Word:$gas, Yul_Word:$target,
                       Yul_Word:$inOffset, Yul_Word:$inSize,
                       Yul_Word:$outOffset, Yul_Word:$outSize);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def Yul_StaticCallOp : Yul_Op<"staticcall"> {
  let summary = "Static call; yields 0 on failure, 1 on success";
  let arguments = (ins Yul_Word:$gas, Yul_Word:$target,
                       Yul_Word:$inOffset, Yul_Word:$inSize,
                       Yul_Word:$outOffset, Yul_Word:$outSize);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$gas `,` $target `,` $inOffset `,` $inSize `,` $outOffset `,` $outSize attr-dict";
}
def Yul_CreateOp : Yul_Op<"create"> {
  let summary = "Create a contract; yields new address or 0 on failure";
  let arguments = (ins Yul_Word:$value, Yul_Word:$offset, Yul_Word:$length);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$value `,` $offset `,` $length attr-dict";
}
def Yul_Create2Op : Yul_Op<"create2"> {
  let summary = "Create2; yields new address or 0 on failure";
  let arguments = (ins Yul_Word:$value, Yul_Word:$offset, Yul_Word:$length,
                       Yul_Word:$salt);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$value `,` $offset `,` $length `,` $salt attr-dict";
}

//===----------------------------------------------------------------------===//
// Halting builtins (terminate the surrounding block)
//===----------------------------------------------------------------------===//

def Yul_StopOp : Yul_Op<"stop", [Terminator]> {
  let summary = "Stop execution successfully with empty returndata";
  let assemblyFormat = "attr-dict";
}
def Yul_ReturnOp : Yul_Op<"return", [Terminator]> {
  let summary = "Stop execution, returning memory range (offset, length)";
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def Yul_RevertOp : Yul_Op<"revert", [Terminator]> {
  let summary = "Revert state, returning memory range (offset, length)";
  let arguments = (ins Yul_Word:$offset, Yul_Word:$length);
  let assemblyFormat = "$offset `,` $length attr-dict";
}
def Yul_InvalidOp : Yul_Op<"invalid", [Terminator]> {
  let summary = "Invalid instruction; consumes all gas";
  let assemblyFormat = "attr-dict";
}
def Yul_SelfDestructOp : Yul_Op<"selfdestruct", [Terminator]> {
  let summary = "Selfdestruct (EIP-6780 semantics post-Cancun)";
  let arguments = (ins Yul_Word:$beneficiary);
  let assemblyFormat = "$beneficiary attr-dict";
}

//===----------------------------------------------------------------------===//
// Object builtins (Yul object model; full yul.object support is an M1
// follow-up - these ops already carry the segment-name semantics)
//===----------------------------------------------------------------------===//

def Yul_DataOffsetOp : Yul_Op<"dataoffset", [Pure]> {
  let summary = "Offset of a named object/data segment";
  let arguments = (ins StrAttr:$segment);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$segment attr-dict";
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$segmentName), [{
      $_state.getOrAddProperties<Properties>().segment =
          $_builder.getStringAttr(segmentName);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}
def Yul_DataSizeOp : Yul_Op<"datasize", [Pure]> {
  let summary = "Size of a named object/data segment";
  let arguments = (ins StrAttr:$segment);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$segment attr-dict";
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$segmentName), [{
      $_state.getOrAddProperties<Properties>().segment =
          $_builder.getStringAttr(segmentName);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}
def Yul_DataCopyOp : Yul_TernaryStmtOp<"datacopy"> {
  let summary = "Copy an object/data segment to memory (dst, src, length)";
}
def Yul_LoadImmutableOp : Yul_Op<"loadimmutable", [Pure]> {
  let summary = "Load a named immutable (deployed-code objects)";
  let arguments = (ins StrAttr:$immutable_name);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$immutable_name attr-dict";
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$immutableName), [{
      $_state.getOrAddProperties<Properties>().immutable_name =
          $_builder.getStringAttr(immutableName);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}
def Yul_SetImmutableOp : Yul_Op<"setimmutable"> {
  let summary = "Patch a named immutable into deployed code (creation objects)";
  let arguments = (ins Yul_Word:$offset, StrAttr:$immutable_name,
                       Yul_Word:$value);
  let assemblyFormat = "$offset `,` $immutable_name `,` $value attr-dict";
}
def Yul_MemoryGuardOp : Yul_Op<"memoryguard", [Pure]> {
  let summary = "Declare the memory region reserved by the compiler";
  let arguments = (ins Yul_WordAttr:$size);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$size attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::IntegerAttr":$sizeAttr), [{
      $_state.getOrAddProperties<Properties>().size = sizeAttr;
      $_state.addTypes(sizeAttr.getType());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Mutable variables (Yul locals; promoted to SSA before evm lowering)
//===----------------------------------------------------------------------===//

def Yul_VarOp : Yul_Op<"var"> {
  let summary = "Declare a mutable Yul variable with an initial value";
  let arguments = (ins Yul_Word:$init);
  let results = (outs Yul_VarRef:$ref);
  let assemblyFormat = "$init attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$initValue), [{
      $_state.addOperands(initValue);
      $_state.addTypes(::mlir::yul::VarRefType::get($_builder.getContext()));
    }]>
  ];
}

def Yul_AssignOp : Yul_Op<"assign"> {
  let summary = "Assign a new value to a mutable Yul variable";
  let arguments = (ins Yul_VarRef:$var, Yul_Word:$value);
  let assemblyFormat = "$var `,` $value attr-dict";
}

def Yul_VarLoadOp : Yul_Op<"var_load"> {
  let summary = "Read the current value of a mutable Yul variable";
  let arguments = (ins Yul_VarRef:$var);
  let results = (outs Yul_Word:$result);
  let assemblyFormat = "$var attr-dict";
  let builders = [
    OpBuilder<(ins "::mlir::Value":$varValue), [{
      $_state.addOperands(varValue);
      $_state.addTypes($_builder.getIntegerType(256));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// User-defined function calls
//===----------------------------------------------------------------------===//

def Yul_FuncCallOp : Yul_Op<"func_call"> {
  let summary = "Call a user-defined Yul function";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Yul_Word>:$operands);
  let results = (outs Variadic<Yul_Word>:$results);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)";
}

//===----------------------------------------------------------------------===//
// Structured control flow (ADR-004)
//===----------------------------------------------------------------------===//

def Yul_IfOp : Yul_Op<"if", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "Yul if statement (no else branch, per Yul grammar)";
  let description = [{
    Executes the body when the condition word is non-zero. Yul has no else
    branch; switch covers multi-way dispatch.
  }];
  let arguments = (ins Yul_Word:$condition);
  let regions = (region SizedRegion<1>:$thenRegion);
  let assemblyFormat = "$condition `then` $thenRegion attr-dict";
}

def Yul_ForOp : Yul_Op<"for", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "Yul for loop (cond / body / post regions)";
  let description = [{
    Models Yul's `for {init} cond {post} {body}`. The init block is hoisted
    before the loop by construction (MLIR SSA names are unique, so Yul's
    init-scoping is preserved trivially). The cond region computes a word and
    ends with yul.condition; the loop continues while it is non-zero.
    yul.continue jumps to the post region; yul.break exits the loop.
  }];
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion,
                        SizedRegion<1>:$postRegion);
  let assemblyFormat = [{
    `cond` $condRegion
    `body` $bodyRegion
    `post` $postRegion
    attr-dict
  }];
}

def Yul_ConditionOp : Yul_Op<"condition", [Terminator]> {
  let summary = "Terminates a yul.for condition region with the loop condition";
  let arguments = (ins Yul_Word:$condition);
  let assemblyFormat = "$condition attr-dict";
}

def Yul_BreakOp : Yul_Op<"break", [Terminator]> {
  let summary = "Break out of the innermost yul.for";
  let assemblyFormat = "attr-dict";
}

def Yul_ContinueOp : Yul_Op<"continue", [Terminator]> {
  let summary = "Continue with the post region of the innermost yul.for";
  let assemblyFormat = "attr-dict";
}

def Yul_LeaveOp : Yul_Op<"leave", [Terminator]> {
  let summary = "Exit the enclosing function, yielding the given return values";
  let description = [{
    Yul's `leave` returns whatever the named return variables hold. In SSA
    form the values are carried explicitly as operands; the Yul-text emitter
    materializes the return-variable assignments.
  }];
  let arguments = (ins Variadic<Yul_Word>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

def Yul_FuncOp : Yul_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    NoTerminator
  ]> {
  let summary = "Yul function definition";
  let description = [{
    A Yul function. Parameters are the entry-block arguments of the body
    region; results are described by the function type and produced by
    yul.leave. Yul functions cannot access variables of enclosing scopes,
    hence IsolatedFromAbove.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().getResults();
    }
  }];
}

#endif // YUL_DIALECT
