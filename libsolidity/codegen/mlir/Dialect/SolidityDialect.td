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
  
  let arguments = (ins StrAttr:$name);
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
def Solidity_ReturnOp : Solidity_Op<"return", [
    Pure, HasParent<"FunctionOp">, Terminator
  ]> {
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

#endif // SOLIDITY_DIALECT