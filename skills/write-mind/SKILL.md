# Write MIND Code

Write correct `.mind` source files for the MIND programming language — a statically typed, tensor-oriented language that compiles to LLVM IR via MLIR.

## When to Use

Activate when the user asks to:
- Write, generate, or create `.mind` files
- Implement algorithms, models, or solvers in MIND
- Port code from Python/Rust/C to MIND
- Explain MIND syntax or semantics

## Language Overview

MIND is a Rust-inspired language designed for numerical computing, ML, and scientific applications. It compiles through MLIR to native code with full autodiff support.

Key characteristics:
- Statically typed with type inference
- First-class tensor types with compile-time shape checking
- Reverse-mode automatic differentiation via `diff` types and `backward()`
- Rust-like syntax: `fn`, `let`, `struct`, `enum`, `trait`, `match`
- No garbage collector — deterministic memory

## Keywords

```
fn  let  type  struct  trait  if  else  match  while  for
return  defer  import  export  where  true  false
```

## Primitive Types

| Type | Description |
|------|-------------|
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |
| `f32` | 32-bit IEEE 754 float |
| `f64` | 64-bit IEEE 754 float |
| `bool` | Boolean (`true` / `false`) |
| `unit` | Unit type (void equivalent) |

## Tensor Types

Tensors are the core primitive. Shape is part of the type:

```mind
let x: tensor<f32[3, 224, 224]>;        // 3D tensor
let scalar: tensor<f64[]>;               // Rank-0 scalar
let batch: tensor<f32[batch, 1, 28, 28]>; // Symbolic batch dim
```

Function signatures:
```mind
fn layer(x: diff tensor<f32[batch, 16, 13, 13]>) -> diff tensor<f32[batch, 32, 5, 5]>
```

> **Note:** The EBNF grammar uses `Tensor<dtype, [dims]>` (uppercase, comma-separated). Surface `.mind` files use the shorthand `tensor<dtype[dims]>` (lowercase, bracket dims). Both are accepted; prefer the lowercase form.

## Differentiable Types

Prefix `diff` marks tensors that participate in automatic differentiation:

```mind
let x: diff tensor<f32> = 3.0;
let y = x * x;
let grad = backward(y, x);  // dy/dx = 2x = 6.0
```

The `backward(loss, parameter)` intrinsic computes gradients via reverse-mode autodiff.

## Composite Types

### Structs
```mind
struct Model {
    layers: i32,
    learning_rate: f32,
}
```

### Enums
```mind
enum Action {
    Read,
    Write,
    Delete,
    Execute,
}
```

Enums can have explicit discriminants:
```mind
enum DenyCode {
    InvalidInput = 1,
    SuspiciousJustification = 2,
    DefaultDeny = 255,
}
```

### Traits
```mind
trait Solver {
    fn solve(self, x: tensor<f64[N]>) -> tensor<f64[N]>;
}
```

## Type Aliases and Generics

```mind
type Matrix<T> = Tensor<T, [N, M]>;
type Vector = Tensor<f64, [N]>;
```

## Functions

```mind
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

Implicit return (last expression without semicolon):
```mind
fn square(x: i32) -> i32 {
    x * x
}
```

Anonymous functions:
```mind
fn(x: f64) -> f64 { x * x }
```

## Control Flow

### If expressions (return values)
```mind
let y = if x > 0 { 1 } else { -1 };
```

### While loops
```mind
let mut i = 0;
while i < n {
    // ...
    i += 1;
}
```

### For loops
```mind
for i in 0..n {
    grid[i] = start + (i as f64) * step;
}
```

### Match (exhaustive pattern matching)
```mind
match action {
    Action::Read => Effect { tag: EffectTag::Allow, code: 0 },
    Action::Write => check_write_permission(req),
    _ => Effect { tag: EffectTag::Deny, code: DenyCode::DefaultDeny as u32 },
}
```

## Statements

### Let bindings
```mind
let x = 42;                    // Type inferred
let y: f64 = 3.14;            // Explicit type
let (a, b) = (1, 2);          // Destructuring
let _ = unused_result();       // Wildcard
```

### Return
```mind
return Effect { tag: EffectTag::Deny, code: 1 };
```

### Defer
```mind
defer { cleanup_resources(); }
```

## Operators

### Precedence (highest to lowest)

| Prec | Operators | Description |
|------|-----------|-------------|
| 1 | `()` `[]` `.` | Grouping, indexing, field access |
| 2 | `-` `!` | Unary negation, logical NOT |
| 3 | `*` `/` `%` | Multiplication, division, modulo |
| 4 | `+` `-` | Addition, subtraction |
| 5 | `==` `!=` `<` `>` `<=` `>=` | Comparison |
| 6 | `&&` | Logical AND |
| 7 | `\|\|` | Logical OR |
| 8 | `=` `+=` `-=` `*=` `/=` `:=` | Assignment (right-to-left) |

### Other operators
- `->` return type annotation
- `=>` match arm
- `::` path separator (imports, enum variants)
- `@` attribute/annotation
- `^` differentiable literal suffix
- `as` type cast

## Imports and Exports

```mind
import std.tensor;
import std.math;

export my_function;
```

Path syntax uses `::` for nested modules:
```mind
import std::tensor::zeros;
```

## Standard Library

### std.tensor
- `tensor.zeros[dtype, shape]` — zero-filled tensor
- `tensor.ones[dtype, shape]` — one-filled tensor
- `reshape(x, shape)` — reshape tensor
- `matmul(a, b)` — matrix multiplication
- `conv2d(x, w, stride, padding)` — 2D convolution
- `maxpool2d(x, kernel, stride)` — max pooling
- `sum(x)` / `sum(x, axis=N)` — reduction
- `mean(x)` / `mean(x, axis=N)` — mean reduction
- `transpose(x, perm)` — transpose
- `expand_dims(x, axis)` / `squeeze(x, axis)` — shape ops
- `gather(x, indices, axis)` — gather elements
- `random_normal(shape, stddev)` — random initialization

### std.math
- `sqrt(x)`, `exp(x)`, `log(x)`, `abs(x)`
- `sin(x)`, `cos(x)`, `tanh(x)`
- Constants: `PI`, `E`

### Activation functions
- `relu(x)` — max(0, x)
- `sigmoid(x)` — 1/(1+e^(-x))
- `log_softmax(x, axis)` — numerically stable log-softmax
- `softmax(x, axis)` — softmax

### Core
- `print(args...)` — stdout output
- `panic!(msg)` — terminate

## Tensor Operations on Types

All arithmetic operators work elementwise on tensors with broadcasting:
```mind
let result = alpha * X + beta * Y;  // Broadcasts scalar to tensor shape
```

Matrix multiplication uses function syntax (not operator):
```mind
let y = matmul(W, x);  // NOT W @ x
```

## Device Placement

```mind
on(gpu0) {
    let result = matmul(A, B);
}
```

## Comments

```mind
// Single-line comment
/* Block comment (nestable) */
```

## Integer Literals

```mind
let dec = 1_000_000;    // Decimal with separators
let bin = 0b1010_1100;  // Binary
let oct = 0o777;        // Octal
let hex = 0xFF_AA;      // Hexadecimal
```

## Full EBNF Grammar — Lexical

Source: `star-ga/mind-spec/spec/v1.0/grammar-lexical.ebnf` (Apache 2.0, STARGA Inc.)

```ebnf
(* Source text structure *)
SourceFile = [ ByteOrderMark ] , { Token | Whitespace | Comment } ;

(* Tokens *)
Token = Identifier | Keyword | Literal | Operator | Punctuation ;

(* Identifiers *)
Identifier = IdentifierStart , { IdentifierContinue } ;
IdentifierStart = Letter | "_" ;
IdentifierContinue = Letter | Digit | "_" ;
Letter = ? Unicode XID_Start ? ;
Digit = ? Unicode XID_Continue & Nd ? | "0"-"9" ;

(* Keywords *)
Keyword = "fn" | "let" | "type" | "struct" | "trait"
        | "if" | "else" | "match" | "while" | "for"
        | "return" | "defer" | "import" | "export" | "where" ;

(* Literals *)
Literal = IntegerLiteral | FloatingPointLiteral | StringLiteral
        | BooleanLiteral | DifferentiableLiteral ;

(* Integer literals *)
IntegerLiteral = [ Sign ] , ( DecimalInteger | BinaryInteger | OctalInteger | HexInteger ) ;
Sign = "+" | "-" ;
DecimalInteger = DecimalDigit , { DecimalDigit | "_" } ;
BinaryInteger = "0b" , BinaryDigit , { BinaryDigit | "_" } ;
OctalInteger = "0o" , OctalDigit , { OctalDigit | "_" } ;
HexInteger = "0x" , HexDigit , { HexDigit | "_" } ;

DecimalDigit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
BinaryDigit = "0" | "1" ;
OctalDigit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" ;
HexDigit = DecimalDigit | "a"-"f" | "A"-"F" ;

(* Floating-point literals *)
FloatingPointLiteral = [ Sign ] , DecimalDigits , "." , DecimalDigits , [ Exponent ] ;
DecimalDigits = DecimalDigit , { DecimalDigit | "_" } ;
Exponent = ( "e" | "E" ) , [ Sign ] , DecimalDigits ;

(* String literals *)
StringLiteral = '"' , { StringCharacter | EscapeSequence } , '"' ;
StringCharacter = ? any Unicode scalar value except '"', '\', newline ? ;
EscapeSequence = "\" , ( "n" | "r" | "t" | "\" | '"' | "0" | UnicodeEscape ) ;
UnicodeEscape = "u" , "{" , HexDigit , { HexDigit } , "}" ;

(* Boolean literals *)
BooleanLiteral = "true" | "false" ;

(* Differentiable literals *)
DifferentiableLiteral = Literal , "^" ;

(* Operators *)
Operator = "+" | "-" | "*" | "/" | "%"
         | "==" | "!=" | "<" | ">" | "<=" | ">="
         | "&&" | "||" | "!"
         | "=" | "+=" | "-=" | "*=" | "/="
         | "->" | "=>" | ":="
         | "." | "::" | "@"
         | "^" ;

(* Punctuation *)
Punctuation = "(" | ")" | "{" | "}" | "[" | "]" | "," | ";" | ":" ;

(* Comments *)
Comment = LineComment | BlockComment ;
LineComment = "//" , { ? any character except newline ? } , LineTerminator ;
BlockComment = "/*" , { ? any character ? | BlockComment } , "*/" ;

(* Whitespace *)
Whitespace = Space | Tab | LineTerminator ;
Space = " " ;
Tab = ? U+0009 ? ;
LineTerminator = LineFeed | CarriageReturn | CarriageReturnLineFeed ;
LineFeed = ? U+000A ? ;
CarriageReturn = ? U+000D ? ;
CarriageReturnLineFeed = CarriageReturn , LineFeed ;
ByteOrderMark = ? U+FEFF ? ;
```

## Full EBNF Grammar — Surface Syntax

Source: `star-ga/mind-spec/spec/v1.0/grammar-syntax.ebnf` (Apache 2.0, STARGA Inc.)

```ebnf
(* Module structure *)
Module = { ModuleItem } ;

ModuleItem = FunctionDeclaration
           | TypeDeclaration
           | StructDeclaration
           | TraitDeclaration
           | ImportDeclaration
           | ExportDeclaration ;

(* Function declarations *)
FunctionDeclaration = "fn" , Identifier , "(" , [ ParameterList ] , ")" ,
                      [ "->" , Type ] , Block ;

ParameterList = Parameter , { "," , Parameter } , [ "," ] ;
Parameter = Identifier , ":" , Type ;

(* Type declarations *)
TypeDeclaration = "type" , Identifier , [ TypeParameters ] , "=" , Type , ";" ;

TypeParameters = "<" , TypeParameter , { "," , TypeParameter } , [ "," ] , ">" ;
TypeParameter = Identifier , [ ":" , TraitBounds ] ;

TraitBounds = TraitBound , { "+" , TraitBound } ;
TraitBound = Identifier ;

(* Struct declarations *)
StructDeclaration = "struct" , Identifier , [ TypeParameters ] ,
                    "{" , [ FieldList ] , "}" ;

FieldList = Field , { "," , Field } , [ "," ] ;
Field = Identifier , ":" , Type ;

(* Trait declarations *)
TraitDeclaration = "trait" , Identifier , [ TypeParameters ] ,
                   [ ":" , TraitBounds ] , "{" , { TraitItem } , "}" ;

TraitItem = FunctionSignature ;
FunctionSignature = "fn" , Identifier , "(" , [ ParameterList ] , ")" ,
                    [ "->" , Type ] , ";" ;

(* Import/Export *)
ImportDeclaration = "import" , ImportPath , ";" ;
ExportDeclaration = "export" , Identifier , ";" ;
ImportPath = Identifier , { "::" , Identifier } ;

(* Types *)
Type = PrimitiveType
     | TensorType
     | TupleType
     | ArrayType
     | FunctionType
     | DifferentiableType
     | TraitObjectType
     | IdentifierType ;

PrimitiveType = "i32" | "i64" | "f32" | "f64" | "bool" | "unit" ;

TensorType = "Tensor" , "<" , DType , "," , Shape , ">" ;
DType = "i32" | "i64" | "f32" | "f64" ;
Shape = "[" , [ DimensionList ] , "]" ;
DimensionList = Dimension , { "," , Dimension } , [ "," ] ;
Dimension = IntegerLiteral | Identifier ;

TupleType = "(" , [ TypeList ] , ")" ;
TypeList = Type , { "," , Type } , [ "," ] ;

ArrayType = "[" , Type , ";" , IntegerLiteral , "]" ;

FunctionType = "(" , [ TypeList ] , ")" , "->" , Type ;

DifferentiableType = "diff" , Type ;

TraitObjectType = "dyn" , Identifier ;

IdentifierType = Identifier , [ "<" , TypeList , ">" ] ;

(* Statements and blocks *)
Block = "{" , { Statement } , [ Expression ] , "}" ;

Statement = LetStatement
          | ExpressionStatement
          | ReturnStatement
          | DeferStatement ;

LetStatement = "let" , Pattern , [ ":" , Type ] , "=" , Expression , ";" ;

Pattern = IdentifierPattern | TuplePattern | WildcardPattern ;
IdentifierPattern = Identifier ;
TuplePattern = "(" , [ PatternList ] , ")" ;
PatternList = Pattern , { "," , Pattern } , [ "," ] ;
WildcardPattern = "_" ;

ExpressionStatement = Expression , ";" ;
ReturnStatement = "return" , [ Expression ] , ";" ;
DeferStatement = "defer" , Block ;

(* Expressions — precedence encoded in production hierarchy *)
Expression = AssignmentExpression ;

AssignmentExpression = LogicalOrExpression ,
                       [ AssignmentOperator , AssignmentExpression ] ;
AssignmentOperator = "=" | "+=" | "-=" | "*=" | "/=" | ":=" ;

LogicalOrExpression = LogicalAndExpression , { "||" , LogicalAndExpression } ;

LogicalAndExpression = ComparisonExpression , { "&&" , ComparisonExpression } ;

ComparisonExpression = AdditiveExpression ,
                       { ComparisonOperator , AdditiveExpression } ;
ComparisonOperator = "==" | "!=" | "<" | ">" | "<=" | ">=" ;

AdditiveExpression = MultiplicativeExpression ,
                     { AdditiveOperator , MultiplicativeExpression } ;
AdditiveOperator = "+" | "-" ;

MultiplicativeExpression = UnaryExpression ,
                           { MultiplicativeOperator , UnaryExpression } ;
MultiplicativeOperator = "*" | "/" | "%" ;

UnaryExpression = [ UnaryOperator ] , PostfixExpression ;
UnaryOperator = "-" | "!" ;

PostfixExpression = PrimaryExpression , { PostfixOperator } ;

PostfixOperator = CallOperator
                | IndexOperator
                | FieldAccessOperator
                | MethodCallOperator ;

CallOperator = "(" , [ ArgumentList ] , ")" ;
ArgumentList = Expression , { "," , Expression } , [ "," ] ;

IndexOperator = "[" , IndexExpression , "]" ;
IndexExpression = Expression | SliceExpression ;
SliceExpression = [ Expression ] , ":" , [ Expression ] , [ ":" , [ Expression ] ] ;

FieldAccessOperator = "." , Identifier ;
MethodCallOperator = "." , Identifier , "(" , [ ArgumentList ] , ")" ;

(* Primary expressions *)
PrimaryExpression = Literal
                  | Identifier
                  | ParenthesizedExpression
                  | TupleExpression
                  | ArrayExpression
                  | TensorConstructor
                  | BlockExpression
                  | IfExpression
                  | MatchExpression
                  | WhileExpression
                  | ForExpression
                  | FunctionExpression ;

ParenthesizedExpression = "(" , Expression , ")" ;

TupleExpression = "(" , Expression , "," , [ ExpressionList ] , ")" ;
ExpressionList = Expression , { "," , Expression } , [ "," ] ;

ArrayExpression = "[" , [ ArrayElements ] , "]" ;
ArrayElements = Expression , { "," , Expression } , [ "," ] ;

TensorConstructor = "tensor" , "(" , Expression ,
                    [ "," , "dtype" , ":" , DType ] , ")" ;

BlockExpression = Block ;

IfExpression = "if" , Expression , Block , [ "else" , ( IfExpression | Block ) ] ;

MatchExpression = "match" , Expression , "{" , { MatchArm } , "}" ;
MatchArm = Pattern , "=>" , ( Expression , "," | Block ) ;

WhileExpression = "while" , Expression , Block ;

ForExpression = "for" , Pattern , "in" , Expression , Block ;

FunctionExpression = "fn" , "(" , [ ParameterList ] , ")" ,
                     [ "->" , Type ] , Block ;

(* Tensor operations — Core v1 intrinsics *)
(* Called as functions: sum(x, axes, keepdims) *)
(* Function call syntax is covered by CallOperator above *)
```

## Full EBNF Grammar — Core IR

Source: `star-ga/mind-spec/spec/v1.0/grammar-ir.ebnf` (Apache 2.0, STARGA Inc.)

The Core IR is the compiler's internal SSA representation. Agents write surface syntax, not IR directly. Included here for completeness.

```ebnf
(* IR Module *)
IRModule = { Instruction } , OutputDeclaration ;

(* Instructions *)
Instruction = ValueId , "=" , Operation , [ AttributeList ] , ":" , TensorType ;
ValueId = "%" , Identifier | Integer ;

(* Operations *)
Operation = InputOperation
          | ConstOperation
          | BinaryOperation
          | ReductionOperation
          | ShapeOperation
          | IndexOperation
          | LinearAlgebraOperation
          | ActivationOperation ;

(* Input operation *)
InputOperation = "Input" , "(" , ")" ;

(* Constant operations *)
ConstOperation = ConstI64Operation | ConstTensorOperation ;
ConstI64Operation = "ConstI64" , "(" , Integer , ")" ;
ConstTensorOperation = "ConstTensor" , "(" , TensorLiteral , ")" ;

TensorLiteral = "[" , [ TensorElements ] , "]" ;
TensorElements = TensorElement , { "," , TensorElement } , [ "," ] ;
TensorElement = Number | TensorLiteral ;

(* Binary operations *)
BinaryOperation = "BinOp" , "(" , BinaryOperator , "," , Operand , "," , Operand , ")" ;
BinaryOperator = "Add" | "Sub" | "Mul" ;
Operand = ValueId ;

(* Reduction operations *)
ReductionOperation = SumOperation | MeanOperation ;
SumOperation = "Sum" , "(" , Operand , "," , AxisList , "," , KeepDims , ")" ;
MeanOperation = "Mean" , "(" , Operand , "," , AxisList , "," , KeepDims , ")" ;

AxisList = "[" , [ Integers ] , "]" ;
Integers = Integer , { "," , Integer } , [ "," ] ;
KeepDims = "true" | "false" ;

(* Shape operations *)
ShapeOperation = ReshapeOperation | TransposeOperation
               | ExpandDimsOperation | SqueezeOperation ;

ReshapeOperation = "Reshape" , "(" , Operand , "," , Shape , ")" ;
TransposeOperation = "Transpose" , "(" , Operand , "," , Permutation , ")" ;
Permutation = "[" , [ Integers ] , "]" ;
ExpandDimsOperation = "ExpandDims" , "(" , Operand , "," , AxisList , ")" ;
SqueezeOperation = "Squeeze" , "(" , Operand , "," , AxisList , ")" ;

(* Indexing operations *)
IndexOperation = IndexOp | SliceOp | GatherOp ;
IndexOp = "Index" , "(" , Operand , "," , IndexList , ")" ;
IndexList = "[" , [ Integers ] , "]" ;
SliceOp = "Slice" , "(" , Operand , "," , SliceRanges , ")" ;
SliceRanges = "[" , [ SliceRange , { "," , SliceRange } , [ "," ] ] , "]" ;
SliceRange = Integer , ":" , Integer , [ ":" , Integer ] ;
GatherOp = "Gather" , "(" , Operand , "," , Operand , ")" ;

(* Linear algebra operations *)
LinearAlgebraOperation = DotOperation | MatMulOperation | Conv2dOperation ;
DotOperation = "Dot" , "(" , Operand , "," , Operand , ")" ;
MatMulOperation = "MatMul" , "(" , Operand , "," , Operand , ")" ;
Conv2dOperation = "Conv2d" , "(" , Operand , "," , Operand , "," ,
                  Strides , "," , Padding , ")" ;
Strides = "[" , Integer , "," , Integer , "]" ;
Padding = "Same" | "Valid" | CustomPadding ;
CustomPadding = "Custom" , "(" , PaddingValues , ")" ;
PaddingValues = "[" , [ PaddingPair , { "," , PaddingPair } , [ "," ] ] , "]" ;
PaddingPair = "[" , Integer , "," , Integer , "]" ;

(* Activation operations *)
ActivationOperation = ReluOperation ;
ReluOperation = "Relu" , "(" , Operand , ")" ;

(* Attributes *)
AttributeList = "{" , [ Attributes ] , "}" ;
Attributes = Attribute , { "," , Attribute } , [ "," ] ;
Attribute = Identifier , ":" , AttributeValue ;
AttributeValue = String | Integer | Number | Boolean | AxisList | Shape ;
Boolean = "true" | "false" ;

(* Types *)
TensorType = "Tensor" , "<" , DType , "," , Shape , ">" ;
DType = "i32" | "i64" | "f32" | "f64" ;
Shape = "[" , [ Dimensions ] , "]" ;
Dimensions = Dimension , { "," , Dimension } , [ "," ] ;
Dimension = Integer | "?" ;

(* Output declaration *)
OutputDeclaration = "outputs" , ":" , OutputList ;
OutputList = ValueId , { "," , ValueId } , [ "," ] ;

(* Primitives *)
Integer = [ "-" ] , Digit , { Digit } ;
Number = [ "-" ] , Digit , { Digit } , [ "." , { Digit } ] , [ Exponent ] ;
Exponent = ( "e" | "E" ) , [ "+" | "-" ] , Digit , { Digit } ;
Digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
String = '"' , { StringCharacter } , '"' ;
StringCharacter = ? any character except '"' and '\' ? | EscapeSequence ;
EscapeSequence = "\" , ( "n" | "r" | "t" | "\" | '"' ) ;
Identifier = IdentifierStart , { IdentifierContinue } ;
IdentifierStart = Letter | "_" ;
IdentifierContinue = Letter | Digit | "_" ;
Letter = ? Unicode XID_Start ? ;
```

## Example: Hello Tensor

```mind
import std.tensor;

fn main() {
    let x = tensor.zeros[f32, (2, 3)];
    let y = x + 1.0;
    on(gpu0) {
        print(y.sum());  // 6.0
    }
    let z = y.reshape((3, 2));
    let result = z * 2.0;
    print("Shape: ", result.shape());
    print("Mean: ", result.mean());
}
```

## Example: Policy Kernel

```mind
enum Action { Read, Write, Delete, Execute }
enum EffectTag { Allow, Deny, RequireConfirmation }
enum DenyCode { InvalidInput = 1, DefaultDeny = 255 }

struct Effect { tag: EffectTag, code: u32 }
struct Request { env: Env, action: Action, resource: Resource, target: Target }

fn evaluate(req: &Request) -> Effect {
    if !validate(req) {
        return Effect { tag: EffectTag::Deny, code: DenyCode::SuspiciousJustification as u32 }
    }
    match req.action {
        Action::Read => Effect { tag: EffectTag::Allow, code: 0 },
        Action::Write => check_write(req),
        _ => Effect { tag: EffectTag::Deny, code: DenyCode::DefaultDeny as u32 },
    }
}
```

## Example: Neural Network Layer with Autodiff

```mind
fn conv_layer(x: diff tensor<f32[batch, 1, 28, 28]>,
              w: diff tensor<f32[16, 1, 3, 3]>,
              b: diff tensor<f32[16]>) -> diff tensor<f32[batch, 16, 13, 13]> {
    let conv = conv2d(x, w, stride=[1,1], padding="valid");
    let biased = conv + b;
    let activated = relu(biased);
    return maxpool2d(activated, kernel=[2,2], stride=[2,2]);
}

fn train_step(x: diff tensor<f32[batch, 1, 28, 28]>,
              labels: tensor<i32[batch]>,
              w: diff tensor<f32[16, 1, 3, 3]>,
              b: diff tensor<f32[16]>,
              lr: f32) -> (diff tensor<f32[16, 1, 3, 3]>, diff tensor<f32[16]>) {
    let logits = forward(x, w, b);
    let loss = cross_entropy(logits, labels);
    let grad_w = backward(loss, w);
    let grad_b = backward(loss, b);
    return (w - lr * grad_w, b - lr * grad_b);
}
```

## Example: ODE Solver (Scientific Computing)

```mind
import std.math;
import std.tensor;

fn linspace(start: f64, end: f64, n: i32) -> tensor<f64[N]> {
    let step = (end - start) / (n - 1) as f64;
    let grid: tensor<f64[N]> = tensor.zeros[f64, (n,)];
    for i in 0..n {
        grid[i] = start + (i as f64) * step;
    }
    return grid;
}

fn interp_linear(x_grid: tensor<f64[N]>, y_grid: tensor<f64[N]>,
                 n: i32, x_query: f64) -> f64 {
    if x_query <= x_grid[0] { return y_grid[0]; }
    if x_query >= x_grid[n - 1] { return y_grid[n - 1]; }
    let step = (x_grid[n-1] - x_grid[0]) / (n - 1) as f64;
    let idx = ((x_query - x_grid[0]) / step) as i32;
    let t = (x_query - x_grid[0]) / step - (idx as f64);
    return y_grid[idx] * (1.0 - t) + y_grid[idx + 1] * t;
}

fn solve(a: fn(f64) -> f64, g: fn(f64) -> f64,
         lambda: f64, x_min: f64, x_max: f64,
         n_grid: i32) -> (tensor<f64[N]>, tensor<f64[N]>) {
    let x = linspace(x_min, x_max, n_grid);
    // ... solver implementation
    return (x, solution);
}
```

## Common Patterns

### Byte-level string operations (no allocator needed)
```mind
fn starts_with(slice: &[u8], prefix: &[u8]) -> bool {
    if prefix.len() > slice.len() { return false }
    let mut i = 0;
    while i < prefix.len() {
        if slice[i] != prefix[i] { return false }
        i += 1;
    }
    true
}
```

### Quantization-aware inference
```mind
fn quantize_weights(w: tensor<f32>, scale: f32) -> tensor<f32> {
    return round(w / scale) * scale;
}
```

### Function pointers as parameters
```mind
fn solve(coeff: fn(f64) -> f64, source: fn(f64) -> f64, lambda: f64) -> tensor<f64[N]> {
    // Accepts coefficient functions as arguments
}
```

## What This Skill Does NOT Cover

- `[protection]` attributes and runtime transforms (private, not in the public compiler)
- Core IR authoring (compiler internal — agents write surface syntax, not IR)
- MLIR lowering details (handled by the compiler automatically)
