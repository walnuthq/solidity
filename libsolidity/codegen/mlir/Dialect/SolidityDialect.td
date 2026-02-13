//===- SolidityDialect.td - Solidity dialect definition ---*- tablegen -*-===//
//
// This file defines the Solidity MLIR dialect using TableGen
//
//===----------------------------------------------------------------------===//

#ifndef SOLIDITY_DIALECT
#define SOLIDITY_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// Solidity dialect definition
//===----------------------------------------------------------------------===//

def Solidity_Dialect : Dialect {
  let name = "solidity";
  let summary = "A dialect for Solidity smart contract language";
  let description = [{
    This dialect provides operations and types for representing Solidity
    smart contracts in MLIR. It captures Solidity-specific semantics including
    state variables, functions, modifiers, events, and blockchain interactions.
  }];
  
  let cppNamespace = "mlir::solidity";
  
  // We'll use 32-bit integer types for type parameters
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Solidity types
//===----------------------------------------------------------------------===//

class Solidity_Type<string name, string typeMnemonic> 
    : TypeDef<Solidity_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Base class for integer types
class Solidity_IntegerType<string name, string typeMnemonic>
    : Solidity_Type<name, typeMnemonic> {
  let parameters = (ins "unsigned":$bitwidth);
  let builders = [
    TypeBuilder<(ins "unsigned":$bitwidth), [{
      return $_get($_ctxt, bitwidth);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

def Solidity_UIntType : Solidity_IntegerType<"UInt", "uint">;
def Solidity_IntType : Solidity_IntegerType<"Int", "int">;

def Solidity_AddressType : Solidity_Type<"Address", "address"> {
  let summary = "Solidity address type";
  let description = [{
    Represents a 160-bit Ethereum address.
  }];
  let hasCustomAssemblyFormat = 1;
}

def Solidity_BoolType : Solidity_Type<"Bool", "bool"> {
  let summary = "Solidity boolean type";
  let hasCustomAssemblyFormat = 1;
}

def Solidity_BytesType : Solidity_Type<"Bytes", "bytes"> {
  let summary = "Fixed-size bytes type";
  let parameters = (ins "unsigned":$size);
  let builders = [
    TypeBuilder<(ins "unsigned":$size), [{
      return $_get($_ctxt, size);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

def Solidity_DynamicBytesType : Solidity_Type<"DynamicBytes", "dynbytes"> {
  let summary = "Dynamic bytes type";
  let hasCustomAssemblyFormat = 1;
}

def Solidity_StringType : Solidity_Type<"String", "string"> {
  let summary = "Solidity string type";
  let hasCustomAssemblyFormat = 1;
}

def Solidity_ArrayType : Solidity_Type<"Array", "array"> {
  let summary = "Solidity array type";
  // Simplified - parameters causing issues with mlir-tblgen
  let hasCustomAssemblyFormat = 1;
}

// Type constraints
def Solidity_AnyType : AnyTypeOf<[
  Solidity_UIntType,
  Solidity_IntType,
  Solidity_AddressType,
  Solidity_BoolType,
  Solidity_BytesType,
  Solidity_DynamicBytesType,
  Solidity_StringType,
  Solidity_ArrayType
]>;

//===----------------------------------------------------------------------===//
// Solidity operations
//===----------------------------------------------------------------------===//

class Solidity_Op<string mnemonic, list<Trait> traits = []> :
    Op<Solidity_Dialect, mnemonic, traits>;

// Contract definition
def Solidity_ContractOp : Solidity_Op<"contract", [
    SingleBlock, NoTerminator, IsolatedFromAbove
  ]> {
  let summary = "Solidity contract definition";
  let description = [{
    Represents a Solidity contract with its state variables and functions.
  }];
  
  let arguments = (ins StrAttr:$name, OptionalAttr<I64Attr>:$id);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$name $body attr-dict";
}

// State variable declaration
def Solidity_StateVarOp : Solidity_Op<"state_var", [HasParent<"ContractOp">]> {
  let summary = "State variable declaration";
  let description = [{
    Declares a state variable in a contract.
  }];
  
  let arguments = (ins 
    StrAttr:$name,
    TypeAttr:$type,
    OptionalAttr<StrAttr>:$visibility,
    UnitAttr:$isConstant,
    OptionalAttr<AnyAttr>:$initialValue
  );
  
  let assemblyFormat = [{
    $name `:` $type 
    (`visibility` `=` $visibility^)? 
    (`constant` $isConstant^)?
    (`=` $initialValue^)?
    attr-dict
  }];
}

// Function definition
def Solidity_FunctionOp : Solidity_Op<"func", [
    AutomaticAllocationScope,
    IsolatedFromAbove
  ]> {
  let summary = "Solidity function definition";
  let description = [{
    Represents a Solidity function with its parameters and body.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$visibility,
    OptionalAttr<StrAttr>:$stateMutability
  );
  
  let regions = (region AnyRegion:$body);
  
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"StringRef", "\"public\"">:$visibility,
      CArg<"StringRef", "\"nonpayable\"">:$stateMutability
    )>
  ];
  
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

// Load state variable
def Solidity_LoadStateVarOp : Solidity_Op<"load_state", [Pure]> {
  let summary = "Load state variable";
  let description = [{
    Loads the value of a state variable.
  }];
  
  let arguments = (ins StrAttr:$varName);
  let results = (outs Solidity_AnyType:$result);
  
  // TableGen automatically generates build methods
  
  let assemblyFormat = "$varName attr-dict `:` type($result)";
}

// Store state variable
def Solidity_StoreStateVarOp : Solidity_Op<"store_state"> {
  let summary = "Store to state variable";
  let description = [{
    Stores a value to a state variable.
  }];
  
  let arguments = (ins StrAttr:$varName, Solidity_AnyType:$value);
  
  let assemblyFormat = "$varName `,` $value attr-dict `:` type($value)";
}

// Arithmetic operations
def Solidity_AddOp : Solidity_Op<"add", [Pure, Commutative]> {
  let summary = "Addition operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  
  // TableGen automatically generates build methods
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_SubOp : Solidity_Op<"sub", [Pure]> {
  let summary = "Subtraction operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_MulOp : Solidity_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiplication operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_DivOp : Solidity_Op<"div", [Pure]> {
  let summary = "Division operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_ModOp : Solidity_Op<"mod", [Pure]> {
  let summary = "Modulo operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_ExpOp : Solidity_Op<"exp", [Pure]> {
  let summary = "Exponentiation operation";
  let arguments = (ins Solidity_AnyType:$base, Solidity_AnyType:$exponent);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$base `,` $exponent attr-dict `:` type($base) `,` type($exponent) `->` type($result)";
}

// Bitwise operations
def Solidity_AndOp : Solidity_Op<"and", [Pure, Commutative]> {
  let summary = "Bitwise AND operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_OrOp : Solidity_Op<"or", [Pure, Commutative]> {
  let summary = "Bitwise OR operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_XorOp : Solidity_Op<"xor", [Pure, Commutative]> {
  let summary = "Bitwise XOR operation";
  let arguments = (ins Solidity_AnyType:$lhs, Solidity_AnyType:$rhs);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Solidity_NotOp : Solidity_Op<"not", [Pure]> {
  let summary = "Bitwise NOT operation";
  let arguments = (ins Solidity_AnyType:$operand);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

def Solidity_ShlOp : Solidity_Op<"shl", [Pure]> {
  let summary = "Left shift operation";
  let arguments = (ins Solidity_AnyType:$value, Solidity_AnyType:$shift);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$value `,` $shift attr-dict `:` type($value) `,` type($shift) `->` type($result)";
}

def Solidity_ShrOp : Solidity_Op<"shr", [Pure]> {
  let summary = "Right shift operation";
  let arguments = (ins Solidity_AnyType:$value, Solidity_AnyType:$shift);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$value `,` $shift attr-dict `:` type($value) `,` type($shift) `->` type($result)";
}

def Solidity_SarOp : Solidity_Op<"sar", [Pure]> {
  let summary = "Arithmetic right shift operation";
  let arguments = (ins Solidity_AnyType:$value, Solidity_AnyType:$shift);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$value `,` $shift attr-dict `:` type($value) `,` type($shift) `->` type($result)";
}

// Comparison operations
def Solidity_CmpOp : Solidity_Op<"cmp", [Pure]> {
  let summary = "Comparison operation";
  let arguments = (ins 
    Solidity_AnyType:$lhs,
    Solidity_AnyType:$rhs,
    StrAttr:$predicate
  );
  let results = (outs Solidity_BoolType:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

// Control flow
def Solidity_IfOp : Solidity_Op<"if", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "If statement";
  let arguments = (ins Solidity_BoolType:$condition);
  let regions = (region SizedRegion<1>:$thenRegion,
                        AnyRegion:$elseRegion);
  
  let assemblyFormat = [{
    $condition `then` $thenRegion
    (`else` $elseRegion^)?
    attr-dict
  }];
}

def Solidity_ForOp : Solidity_Op<"for", [RecursiveMemoryEffects]> {
  let summary = "For loop";
  let regions = (region SizedRegion<1>:$init,
                        SizedRegion<1>:$condition,
                        SizedRegion<1>:$update,
                        SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    `init` $init
    `condition` $condition  
    `update` $update
    `do` $body
    attr-dict
  }];
}

// Return operation
// Note: HasParent<"FunctionOp"> removed to allow returns inside nested regions (if blocks, etc.)
def Solidity_ReturnOp : Solidity_Op<"return", [Pure, Terminator]> {
  let summary = "Return from function";
  let arguments = (ins Variadic<Solidity_AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

// Require operation
def Solidity_RequireOp : Solidity_Op<"require"> {
  let summary = "Require condition check";
  let arguments = (ins Solidity_BoolType:$condition, OptionalAttr<StrAttr>:$message);
  let assemblyFormat = "$condition (`,` $message^)? attr-dict `:` type($condition)";
}

// Assert operation
def Solidity_AssertOp : Solidity_Op<"assert"> {
  let summary = "Assert condition check";
  let arguments = (ins Solidity_BoolType:$condition);
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
}

// Revert operation
def Solidity_RevertOp : Solidity_Op<"revert", [Terminator]> {
  let summary = "Revert transaction";
  let arguments = (ins OptionalAttr<StrAttr>:$reason);
  let assemblyFormat = "($reason^)? attr-dict";
}

// Constant operation
def Solidity_ConstantOp : Solidity_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant value";
  let arguments = (ins AnyAttr:$value);
  let results = (outs Solidity_AnyType:$result);
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$type)>
  ];
  
  // Use custom printer/parser to handle the constant value
  let hasCustomAssemblyFormat = 1;
}

// Type conversion operations
def Solidity_ToI1Op : Solidity_Op<"to_i1", [Pure]> {
  let summary = "Convert Solidity bool to i1";
  let description = [{
    Converts a Solidity boolean type to MLIR's standard i1 type.
    This is needed for interoperability with SCF and other standard dialects.
  }];
  let arguments = (ins Solidity_BoolType:$input);
  let results = (outs I1:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Solidity_ConvertOp : Solidity_Op<"convert", [Pure]> {
  let summary = "Type conversion between Solidity types";
  let description = [{
    Converts a value from one Solidity type to another.
    Used for explicit type conversions like address(x), uint256(x), etc.
  }];
  let arguments = (ins Solidity_AnyType:$input);
  let results = (outs Solidity_AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Array operations
def Solidity_ArrayPushOp : Solidity_Op<"array_push"> {
  let summary = "Push element to dynamic array";
  let description = [{
    Pushes a new element to the end of a dynamic array.
  }];
  let arguments = (ins Solidity_ArrayType:$array, Solidity_AnyType:$value);
  let assemblyFormat = "$array `,` $value attr-dict `:` type($array) `,` type($value)";
}

def Solidity_ArrayLengthOp : Solidity_Op<"array_length", [Pure]> {
  let summary = "Get array length";
  let description = [{
    Returns the length of an array.
  }];
  let arguments = (ins Solidity_ArrayType:$array);
  let results = (outs Solidity_UIntType:$length);
  let assemblyFormat = "$array attr-dict `:` type($array) `->` type($length)";
}

// Struct operations
def Solidity_StructCreateOp : Solidity_Op<"struct_create", [Pure]> {
  let summary = "Create struct instance";
  let description = [{
    Creates a new struct instance with the given field values.
  }];
  let arguments = (ins 
    StrAttr:$name,
    Variadic<Solidity_AnyType>:$fields
  );
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$name `(` $fields `)` attr-dict `:` functional-type($fields, $result)";
}

def Solidity_MemberAccessOp : Solidity_Op<"member_access", [Pure]> {
  let summary = "Access struct member or array property";
  let description = [{
    Accesses a member of a struct or a property (like length) of an array.
  }];
  let arguments = (ins
    Solidity_AnyType:$object,
    StrAttr:$member
  );
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$object $member attr-dict `:` type($object) `->` type($result)";
}

// Address member access operations
def Solidity_AddressBalanceOp : Solidity_Op<"address_balance", [Pure]> {
  let summary = "Get balance of an address";
  let description = [{
    Returns the balance in wei of the given address.
  }];
  let arguments = (ins Solidity_AddressType:$address);
  let results = (outs Solidity_UIntType:$balance);
  let assemblyFormat = "$address attr-dict `:` type($address) `->` type($balance)";
}

def Solidity_AddressCodeOp : Solidity_Op<"address_code", [Pure]> {
  let summary = "Get code at an address";
  let description = [{
    Returns the contract code at the given address as a bytes array.
  }];
  let arguments = (ins Solidity_AddressType:$address);
  let results = (outs Solidity_ArrayType:$code);
  let assemblyFormat = "$address attr-dict `:` type($address) `->` type($code)";
}

def Solidity_AddressCodehashOp : Solidity_Op<"address_codehash", [Pure]> {
  let summary = "Get codehash of an address";
  let description = [{
    Returns the keccak256 hash of the contract code at the given address.
  }];
  let arguments = (ins Solidity_AddressType:$address);
  let results = (outs Solidity_BytesType:$codehash);
  let assemblyFormat = "$address attr-dict `:` type($address) `->` type($codehash)";
}

// Control flow operations
def Solidity_UncheckedOp : Solidity_Op<"unchecked", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "Unchecked arithmetic block";
  let description = [{
    Represents an unchecked block where arithmetic operations don't check for overflow.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def Solidity_BreakOp : Solidity_Op<"break", [Terminator]> {
  let summary = "Break from loop";
  let description = [{
    Breaks out of the current loop.
  }];
  let assemblyFormat = "attr-dict";
}

def Solidity_ContinueOp : Solidity_Op<"continue", [Terminator]> {
  let summary = "Continue to next loop iteration";
  let description = [{
    Continues to the next iteration of the current loop.
  }];
  let assemblyFormat = "attr-dict";
}

// Logical operations (fixing the logical_not issue)
def Solidity_LogicalNotOp : Solidity_Op<"logical_not", [Pure]> {
  let summary = "Logical NOT operation";
  let arguments = (ins Solidity_BoolType:$operand);
  let results = (outs Solidity_BoolType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Built-in functions
//===----------------------------------------------------------------------===//

// Modular arithmetic operations
def Solidity_AddModOp : Solidity_Op<"addmod", [Pure]> {
  let summary = "Modular addition without overflow";
  let description = [{
    Computes (a + b) % n without intermediate overflow.
    Equivalent to the Yul addmod(a, b, n) opcode.
  }];
  let arguments = (ins Solidity_AnyType:$a, Solidity_AnyType:$b, Solidity_AnyType:$n);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$a `,` $b `,` $n attr-dict `:` type($a) `,` type($b) `,` type($n) `->` type($result)";
}

def Solidity_MulModOp : Solidity_Op<"mulmod", [Pure]> {
  let summary = "Modular multiplication without overflow";
  let description = [{
    Computes (a * b) % n without intermediate overflow.
    Equivalent to the Yul mulmod(a, b, n) opcode.
  }];
  let arguments = (ins Solidity_AnyType:$a, Solidity_AnyType:$b, Solidity_AnyType:$n);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$a `,` $b `,` $n attr-dict `:` type($a) `,` type($b) `,` type($n) `->` type($result)";
}

// Block/Transaction functions
def Solidity_GasLeftOp : Solidity_Op<"gasleft", [Pure]> {
  let summary = "Get remaining gas";
  let description = [{
    Returns the amount of gas remaining in the current execution context.
    Equivalent to the Yul gas() opcode.
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_BlockhashOp : Solidity_Op<"blockhash", [Pure]> {
  let summary = "Get block hash";
  let description = [{
    Returns the hash of the given block number. Only works for the 256 most recent blocks.
    Equivalent to the Yul blockhash(blockNumber) opcode.
  }];
  let arguments = (ins Solidity_AnyType:$blockNumber);
  let results = (outs Solidity_BytesType:$result);
  let assemblyFormat = "$blockNumber attr-dict `:` type($blockNumber) `->` type($result)";
}

// Hash functions
def Solidity_Keccak256Op : Solidity_Op<"keccak256", [Pure]> {
  let summary = "Compute keccak256 hash";
  let description = [{
    Computes the Keccak-256 hash of the input data.
    For bytes in memory, this involves storing data and calling keccak256(offset, length).
  }];
  let arguments = (ins Solidity_AnyType:$data);
  let results = (outs Solidity_BytesType:$result);
  let assemblyFormat = "$data attr-dict `:` type($data) `->` type($result)";
}

def Solidity_Sha256Op : Solidity_Op<"sha256", [Pure]> {
  let summary = "Compute SHA-256 hash";
  let description = [{
    Computes the SHA-256 hash of the input data using the precompile at address 0x02.
  }];
  let arguments = (ins Solidity_AnyType:$data);
  let results = (outs Solidity_BytesType:$result);
  let assemblyFormat = "$data attr-dict `:` type($data) `->` type($result)";
}

def Solidity_Ripemd160Op : Solidity_Op<"ripemd160", [Pure]> {
  let summary = "Compute RIPEMD-160 hash";
  let description = [{
    Computes the RIPEMD-160 hash of the input data using the precompile at address 0x03.
  }];
  let arguments = (ins Solidity_AnyType:$data);
  let results = (outs Solidity_BytesType:$result);
  let assemblyFormat = "$data attr-dict `:` type($data) `->` type($result)";
}

// Cryptographic recovery
def Solidity_EcrecoverOp : Solidity_Op<"ecrecover", [Pure]> {
  let summary = "Recover address from signature";
  let description = [{
    Recovers the address associated with the public key from an elliptic curve signature.
    Uses the precompile at address 0x01.
    Arguments: hash (bytes32), v (uint8), r (bytes32), s (bytes32)
  }];
  let arguments = (ins
    Solidity_AnyType:$hash,
    Solidity_AnyType:$v,
    Solidity_AnyType:$r,
    Solidity_AnyType:$s
  );
  let results = (outs Solidity_AddressType:$result);
  let assemblyFormat = "$hash `,` $v `,` $r `,` $s attr-dict `:` type($hash) `,` type($v) `,` type($r) `,` type($s) `->` type($result)";
}

// Self-destruct (deprecated but needed for compatibility)
def Solidity_SelfdestructOp : Solidity_Op<"selfdestruct"> {
  let summary = "Destroy contract and send balance";
  let description = [{
    Terminates the contract and sends its remaining balance to the specified address.
    Note: This opcode is deprecated since EIP-6049 but still needed for compatibility.
  }];
  let arguments = (ins Solidity_AddressType:$recipient);
  let assemblyFormat = "$recipient attr-dict `:` type($recipient)";
}

//===----------------------------------------------------------------------===//
// Mapping operations
//===----------------------------------------------------------------------===//

def Solidity_MappingAccessOp : Solidity_Op<"mapping_access", [Pure]> {
  let summary = "Access mapping element";
  let description = [{
    Accesses a mapping element by key. Returns the value stored at the key.
  }];
  let arguments = (ins StrAttr:$varName, Solidity_AnyType:$key);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "$varName `[` $key `]` attr-dict `:` type($key) `->` type($result)";
}

def Solidity_MappingStoreOp : Solidity_Op<"mapping_store"> {
  let summary = "Store to mapping element";
  let description = [{
    Stores a value to a mapping element at the given key.
  }];
  let arguments = (ins StrAttr:$varName, Solidity_AnyType:$key, Solidity_AnyType:$value);
  let assemblyFormat = "$varName `[` $key `]` `,` $value attr-dict `:` type($key) `,` type($value)";
}

//===----------------------------------------------------------------------===//
// Message context operations
//===----------------------------------------------------------------------===//

def Solidity_MsgSenderOp : Solidity_Op<"msg_sender", [Pure]> {
  let summary = "Get message sender address";
  let description = [{
    Returns the address of the message sender (msg.sender).
  }];
  let results = (outs Solidity_AddressType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_MsgValueOp : Solidity_Op<"msg_value", [Pure]> {
  let summary = "Get message value in wei";
  let description = [{
    Returns the amount of wei sent with the message (msg.value).
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_MsgDataOp : Solidity_Op<"msg_data", [Pure]> {
  let summary = "Get complete calldata";
  let description = [{
    Returns the complete calldata (msg.data).
  }];
  let results = (outs Solidity_ArrayType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_MsgSigOp : Solidity_Op<"msg_sig", [Pure]> {
  let summary = "Get function selector from calldata";
  let description = [{
    Returns the first four bytes of the calldata (msg.sig).
  }];
  let results = (outs Solidity_BytesType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Block context operations
//===----------------------------------------------------------------------===//

def Solidity_BlockTimestampOp : Solidity_Op<"block_timestamp", [Pure]> {
  let summary = "Get current block timestamp";
  let description = [{
    Returns the current block's timestamp (block.timestamp).
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_BlockNumberOp : Solidity_Op<"block_number", [Pure]> {
  let summary = "Get current block number";
  let description = [{
    Returns the current block number (block.number).
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_BlockChainIdOp : Solidity_Op<"block_chainid", [Pure]> {
  let summary = "Get current chain ID";
  let description = [{
    Returns the current chain ID (block.chainid).
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Transaction context operations
//===----------------------------------------------------------------------===//

def Solidity_TxOriginOp : Solidity_Op<"tx_origin", [Pure]> {
  let summary = "Get transaction origin address";
  let description = [{
    Returns the origin address of the transaction (tx.origin).
  }];
  let results = (outs Solidity_AddressType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Solidity_TxGasPriceOp : Solidity_Op<"tx_gasprice", [Pure]> {
  let summary = "Get transaction gas price";
  let description = [{
    Returns the gas price of the transaction (tx.gasprice).
  }];
  let results = (outs Solidity_UIntType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ABI encoding/decoding operations
//===----------------------------------------------------------------------===//

def Solidity_AbiEncodeOp : Solidity_Op<"abi_encode", [Pure]> {
  let summary = "ABI encode arguments";
  let description = [{
    ABI-encodes the given arguments. Equivalent to abi.encode(...).
  }];
  let arguments = (ins Variadic<Solidity_AnyType>:$args);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $result)";
}

def Solidity_AbiEncodePackedOp : Solidity_Op<"abi_encode_packed", [Pure]> {
  let summary = "ABI encode packed arguments";
  let description = [{
    ABI-encodes the given arguments with tight packing. Equivalent to abi.encodePacked(...).
  }];
  let arguments = (ins Variadic<Solidity_AnyType>:$args);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $result)";
}

def Solidity_AbiEncodeWithSelectorOp : Solidity_Op<"abi_encode_with_selector", [Pure]> {
  let summary = "ABI encode with selector";
  let description = [{
    ABI-encodes the given arguments with a function selector. Equivalent to abi.encodeWithSelector(...).
  }];
  let arguments = (ins Variadic<Solidity_AnyType>:$args);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $result)";
}

def Solidity_AbiEncodeWithSignatureOp : Solidity_Op<"abi_encode_with_signature", [Pure]> {
  let summary = "ABI encode with signature";
  let description = [{
    ABI-encodes the given arguments with a function signature string. Equivalent to abi.encodeWithSignature(...).
  }];
  let arguments = (ins Variadic<Solidity_AnyType>:$args);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $result)";
}

def Solidity_AbiDecodeOp : Solidity_Op<"abi_decode", [Pure]> {
  let summary = "ABI decode data";
  let description = [{
    ABI-decodes the given data. Equivalent to abi.decode(...).
  }];
  let arguments = (ins Variadic<Solidity_AnyType>:$args);
  let results = (outs Solidity_AnyType:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $result)";
}

//===----------------------------------------------------------------------===//
// Function call operation
//===----------------------------------------------------------------------===//

def Solidity_FunctionCallOp : Solidity_Op<"function_call"> {
  let summary = "Call a Solidity function";
  let description = [{
    Calls a Solidity function by name with the given arguments.
  }];
  let arguments = (ins StrAttr:$callee, Variadic<Solidity_AnyType>:$args);
  let results = (outs Variadic<Solidity_AnyType>:$results);
  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

//===----------------------------------------------------------------------===//
// Event emission operation
//===----------------------------------------------------------------------===//

def Solidity_EmitOp : Solidity_Op<"emit"> {
  let summary = "Emit an event";
  let description = [{
    Emits a Solidity event with the given arguments.
  }];
  let arguments = (ins
    StrAttr:$event,
    OptionalAttr<StrAttr>:$eventSignature,
    OptionalAttr<ArrayAttr>:$indexed,
    Variadic<Solidity_AnyType>:$args
  );
  let assemblyFormat = "$event `(` $args `)` attr-dict `:` type($args)";
}

#endif // SOLIDITY_DIALECT