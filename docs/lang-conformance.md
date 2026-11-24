# Language conformance corpus

The language is defined by its specification (the `star-ga/mind-spec` repository, `spec/v1.0/`). This corpus measures
how far the compiler of this tree follows it. The expected verdicts follow the published specification at commit
`4139f55` (main, 2026-11-23). When the specification gets a new revision, the corpus changes with it: the rows are
checked against the new text, and the change that does this names the new commit here. No sentence about compiler
behaviour is written by hand: the numbers and the gap list in [lang-conformance-status.md](lang-conformance-status.md)
are generated from data files that a gate keeps true.

## What is in it

- `tests/lang_conformance/<family>/<name>.mind`: one small program per row. The families are the parts of the
  language: `items`, `control`, `expressions`, `types`, `lexical`.
- `tests/lang_conformance/ROWS.tsv`: one record per program. The columns:
  - `id`: `<family>/<name>`, the path of the program without `.mind`;
  - `check`: what `mindc check --no-fmt <file>` must do, `accept` or `reject`;
  - `build`: what `mindc build <file> --backend native --out <image>` must do, `accept` or `reject`;
  - `exit` and `stdout`: what the image must give when it runs. The exit status is the value that `main` returns; a
    program that the specification gives a value pins it (an empty `stdout` means the image prints nothing; `\n` is a
    new line). A row pins a value only when every step from the literals of the program to the value of `main` has a
    value rule in the specification or in one of the value readings below. `-` in `exit` is for every other row (for
    example one whose value goes through an `as` cast); then the run must end by itself with an exit status from 0 to
    255. `-` in `stdout` means that the specification does not give the output: then the run may print anything. An
    accept row pins `stdout` also when its exit is `-`: no row calls an output function, so the image prints nothing. A
    reject row has `-` in both;
  - `code`: for a reject row, the code of `spec/v1.0/errors.md` whose trigger is the fault of the program (`-`: the
    catalog has no entry for it, as for a syntax error, or two entries could apply). The catalog is the contract:
    when the compiler prints another code, the row is a `conformance-gap`;
  - `spec`: the file and production or section of the specification that decides the row;
  - `construct`: a short name of what the row tests.

  The expected verdict is the verdict that the specification demands, not what the compiler does today. A program that
  the specification defines is an `accept` row at all three stages (a construct counts as solved only when check
  accepts it and the native image runs right). A program that the specification forbids is a `reject` row.
- `tests/lang_conformance/KNOWN_GAPS.tsv`: every row where this tree differs from the expected verdict, with its class
  and the measured first diagnostic or refusal key. The classes:
  - `conformance-gap`: the specification defines the program and check rejects it, or a reject row gets another code;
  - `native-only-gap`: check accepts, the native backend refuses with a `lane.` key, or its image runs wrong;
  - `loose`: the compiler accepts what the specification forbids;
  - `unkeyed-refusal`: the native backend refuses with no key.

## The gate

`examples/mindc_mind/lang_conformance_gate.py` runs every row through the three stages and compares the result with the
expected verdict. It is green only when the set of rows that differ equals `KNOWN_GAPS.tsv` exactly:

- a new difference makes it red (a regression, or a new row that nobody listed);
- a listed gap that no longer differs makes it red too: the list must change in the same commit as the fix;
- the number of listed gaps may never grow. The gate reads every committed version of `KNOWN_GAPS.tsv` in the git
  history and is red when the list holds more gaps than any of them. An edit of the list can not raise the count.
- a stage that crashes differs from every expected verdict: a signal, a timeout, or an exit status that is not
  documented (check: 0 or 1; build: 0 with an image, 1 for a compilation error, 3 for a refusal of the native backend
  with no image). A crash, a panic or a hang is never counted as a rejection.

It prints one verdict line per row (the `asserted=` count of `scripts/run_gate.py` is the number of rows) and the count
per class. It also fails when `docs/lang-conformance-status.md` differs from a new generation. `--self-test` proves on
planted rows, with the real compiler, that a green list, a new difference, a closed gap that is still listed, a wrong
class or key, a wrong expected verdict at each stage (check, error code, native build, exit status, stdout), a crash, a
hang and an undocumented status of the compiler, a build that panics, that exits 0 with no image or that writes an image
with the refusal status, a native refusal with and without a `lane.` key, an image that a signal kills or that does not
end in the run limit, a stdout that is not UTF-8, an image that prints on a row whose `stdout` is `-`, an exit status of
255, the keys of the first error and of a refusal, a gap count that grows (also with `#` lines in a committed list,
after a commit that deleted the list, in a shallow clone, and when `git log` fails), a stale or missing status table, a
compiler path that is not an executable file, and each malformed line of the two data files are each judged right.
It also runs the gate script itself as a child on planted corpora in their own git history and checks its exit status: 0
when the differences equal the list, 1 for a new difference, a listed gap that names no row, a gap count above the
committed one, a stale table, a program with no row, no compiler and a failed core limit check; `--write-status` exits 0
and writes the table, and exits 1 on a corpus that it refuses.

Run it with the compiler of the tree:

```bash
MINDC_BIN=target/release/mindc python3 scripts/run_gate.py --min-asserted <rows> \
    examples/mindc_mind/lang_conformance_gate.py
MINDC_BIN=target/release/mindc python3 scripts/run_gate.py examples/mindc_mind/lang_conformance_gate.py --self-test
```

## How to move a row out of the gap list

1. Fix the compiler so that the row gives its expected verdict.
2. Run `python3 examples/mindc_mind/lang_conformance_gate.py --write-gaps`. It removes each gap that no longer differs,
   and updates the class and key of each listed gap that still differs. It never adds a row: a new difference stays red.
3. Run `python3 examples/mindc_mind/lang_conformance_gate.py --write-status` to regenerate the status table.
4. Commit the compiler change, `KNOWN_GAPS.tsv` and `docs/lang-conformance-status.md` together.

## How to add a row

Add the program under its family and one line to `ROWS.tsv` with the verdict that the specification demands and the
reference that decides it. The reference is a file of `spec/v1.0/` of the specification and its production or section.
A row whose expected verdict the specification does not decide does not belong here: the specification text comes
first. A new row that differs from its expected verdict needs a line in `KNOWN_GAPS.tsv`, and the count of that list
may never grow. So a new differing row goes in only in the same change that closes another gap.

## Readings of the specification that the rows use

Every row rests on the text of `spec/v1.0/`, never on what a compiler does. Where the text needs a reading, the corpus
uses one reading for all rows:

- A statement is derived by `grammar-syntax.ebnf`: an `if`, `while`, `for` or `match` expression before another
  statement is an `ExpressionStatement` and ends with `;`. The last expression of a block has no `;`.
- Forms that the prose of the specification defines and that `grammar-syntax.ebnf` does not connect to a module item
  or to the expression chain are forms of the language: enum, `const`, inline `module` and `use` declarations, the
  `as` cast, `&[T]` and `&mut [T]`, and the operators of the precedence table of `language.md`
  (`&`, `|`, `^`, `<<`, `>>`, `.*`, `./`, `.+`, `.-`) at the level that the table gives them. The corpus measures the
  language with the v1.1 candidate additions of `grammar-syntax.ebnf` (an implementation that claims v1.1 MUST parse
  them) and with the traits and type parameters that the grammar derives, so their rows are accept rows.
- A sign on an integer or a floating-point literal is a form of the language because `grammar-lexical.ebnf` gives
  `[ Sign ]` to both literals (lines 55 and 68); the prose of `lexical.md` gives a
  floating-point literal no sign, and the grammar decides.
- A pattern is an identifier, a tuple or `_` (`Pattern`). A literal pattern is refused. Enum variant values and
  patterns (`Shape::Circle(3)`, `E::A`) use the spellings that `types.md` gives for enum variant paths: a variant
  pattern is `E::V` for a variant with no payload and `E::V(p1, ..., pn)`, one pattern for each payload value, for a
  variant with a payload.
- A literal with no expected type has the type of its kind: an integer literal `i64` (`ir.md`: the integer constant
  `ConstI64` has the dtype `i64`), a floating-point literal `f64`, a string literal the string type of `stdlib.md`. A
  literal whose expected type is a type of its kind takes that type (`types.md`: expressions are checked "by propagating
  expected types from their context"), so `let x: i32 = 1;` and `let y: f32 = 1.0;` are accepted. The text writes the
  string type as `string` in one place and as `String` in another, and no sentence makes either name usable in a type
  position, so no row names the string type, and no row depends on the value of a string at run time.
- A type alias denotes its right side, with each type parameter replaced by its type argument (`errors.md`, `E2036`:
  "after resolving type aliases"); the same holds for the fields of a struct with type parameters. A function with
  no `->` has the result type `unit`, and a block with no final expression has the type `unit` (`stdlib.md`: the
  unit type, no value).
- An inline `module name { ... }` block is transparent: its items are items of the enclosing source and are used
  with no qualifier (`types.md`: transparent syntax containers whose declarations are flattened into the file).
- `types.md` names struct literals but gives no spelling for one, so no row constructs a struct value: a row that
  needs one takes it as a parameter.
- `types.md` gives an array literal the type `[T; n]` and gives no way to build an `array<T>` value. A row that needs an
  `array<T>` value takes it as a parameter. An array literal stands only in these places: as the initializer of a `let`
  or `const` whose type is `[T; n]` (`types.md`, fixed-array literal cardinality), directly or in parentheses (the same
  section: parentheses do not change the cardinality), through an alias, as the value of each branch of an `if`, or as
  an element for a nested array type; as the initializer of a `let` with no annotation, which then has the type of the
  literal; as an argument for a `&[T]` or `&mut [T]` parameter (`types.md`, slice call boundary: "A compatible
  `array<T>` value or array literal MAY be passed"); and as the `shape: [i64]` argument of the tensor constructors of
  `stdlib.md` (its example: `zeros([2, 3])`).
- A trait method signature ends with `;` (`FunctionSignature`); a parameter has a type (`Parameter`).
- A function type is a type form (`types.md`, type formation). A function name used as a value is `E2037`
  (`errors.md`). No accept row uses a function name as a value.
- `dyn Trait` is defined only for traits marked as object-safe, and the specification gives no way to mark a trait.
  The corpus has no row for it.
- `Implementations MAY extend the set of primitive types` (`types.md`). The catalog entry `E2036` names `i8`, `u8`,
  `i16` and `u16` as element types, so the corpus reads these four as types of the language. No normative text
  names `u32`, `u64` or a type wider than 64 bits, so the corpus has no row for them.
- A program that builds to an image has a `main` with no parameter and the result type `i64`; no row gives `main`
  another form. The exit status of the image is the value that `main` returns, as the host reports it: the low
  eight bits (a `main` that returns 1000000 exits with 64, -5 exits with 251).
- Error codes. The same kind of fault has the same `code` in every row. A run of characters that starts with a decimal
  digit and goes on with letters, digits, `_` or `.` is one numeric literal; when no literal production derives it, the
  code is `E1005` (the catalog examples `0b2` and `1.2.3` are such runs; `1..3` is one too). A keyword used as a name
  and a syntax error have `-` (no catalog entry names them). A character that starts no token is part of an identifier
  only when identifier characters stand on both sides of it (`a$b`): then the code is `E1002` (`errors.md`: an
  identifier with a character outside its class; the example is `$`). In every other place (`'a'`, `x?`) the code is
  `-`. `E2001` is the code of a value whose type differs from its expected type when no other entry names the fault;
  where the trigger of an entry names it (`E2004`, `E2026`, `E2032`, `E2033`, `E2036`, `E3001`, `E3002`, `E3006`), the
  row uses that entry. When two such entries apply, the code is `-`: `log` of an integer has `-` (`E2005` and `E2006`
  both apply).
- Values. A pinned exit value uses only these rules. A literal has the value of its sign and digits in its base
  (`lexical.md`, integer literals; `ir.md`, `ConstI64`). `true` and `false` are the two values of `bool`
  (`stdlib.md`). `+`, `-` and `*` on integers are addition, subtraction and multiplication, `/` and `%` are
  division and remainder, unary `-` is negation, `==` `!=` `<` `>` `<=` `>=` compare the values of their operands
  and give `true` when the relation holds, `&&` `||` `!` are logical AND, OR and NOT (`language.md`, operator table
  and the values of its examples). On non-negative integers, `&` `|` `^` are the bitwise AND, OR and exclusive OR of
  the binary digits, `x << c` is `x` times 2 to the power `c`, and `x >> c` is `x` divided by 2 to the power `c`,
  rounded down (`grammar-syntax.ebnf`: `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `ShiftLeft`, `ShiftRight`;
  `language.md`: `0 - 17 >> 2` evaluates to `-4`). Floating-point operations and comparisons follow IEEE 754
  (`security.md`, numeric overflow; `language.md`, floating-point comparisons). An `if` runs its block when the
  condition is `true` and its `else` part when it is `false`, and an `if` with an `else` has the value of the part
  that runs (`IfExpression`). A `match` takes the first arm whose pattern matches (`errors.md`, `E2301`:
  first-match semantics) and has the value of the expression of that arm; an identifier pattern matches every value
  and binds it, `_` matches every value, a tuple pattern binds by position, and a variant pattern matches a value of
  its variant and binds the payload values by position. A variant value `E::V(e1, ..., en)` is a value of the
  variant `V` that holds the values of `e1` to `en`. A tuple expression is the tuple of the values of its elements.
  A block has the value of its final expression. A function returns the value of `return` or of the final expression
  of its body, and a call has the value that the called function returns; a function expression is a function that
  a call of its value runs. A `let` binds its pattern to the value, and a `const` name denotes the value of its
  initializer. Arguments are evaluated in the caller, then bound to the parameters (`conformance.md`). `abs` is the
  absolute value and `assert` aborts only on `false` (`stdlib.md`).
- Values that the text does not give, so every row that needs one has `-` in `exit`: the value of an `as` cast; a `bool`
  as an integer; a literal with `_` in its digits (the grammar allows `_`, no sentence gives its value); the element
  that an index selects (no text says where indexing starts); the result of the `Option` and `Result` methods
  (`stdlib.md` gives only their signatures); integer overflow (`security.md` leaves it to the implementation); an
  integer `/` or `%` with a negative operand or a zero divisor (a floating-point `/` follows the IEEE 754 sentence
  above); `%` on floating-point values; a shift count of 64 or more; `&`, `|`, `^` or a shift on a negative value; a
  loop; a floating-point value in the exit status; an elementwise operator `.*`, `./`, `.+` or `.-` on scalar operands
  (the text gives them no operand rule, result type or value there, so no row uses them on scalars). The text gives no
  value to an omitted slice bound (`a[:2]`) and no type to a slice of a fixed array, so no row uses them.
- An element read from a read-only slice (`xs[i]`) is a value of the element type, not a borrowed value (`types.md`:
  read-only slices permit indexing), so a row may return it through a non-slice return.
- `/` and `%` are operators of `MultiplicativeOperator` (`grammar-syntax.ebnf`). The sentence of `language.md` that
  division is not part of Core v1 is about the lowering of tensor arithmetic to the Core IR, not about the surface
  syntax, so a scalar `/` or `%` row is an accept row.
- The elementwise operators `.+`, `.-` and `.*` on two tensors are the elementwise addition, subtraction and
  multiplication of the precedence table (`language.md`), with the type rule of tensor arithmetic: the operands share a
  dtype and the result has the broadcast shape (`language.md`, tensor operations). `./` is division, which is not part
  of Core v1 for tensor operations (`language.md`, tensor operations), so no row uses it. No row pins a value that goes
  through an elementwise operator.
- `tensor(e, dtype: D)` with a rank-0 literal `e` has the type `Tensor<D, []>` (`language.md`: tensor literals with a
  dtype annotation, rank-0 literals represent scalars, a literal becomes a `ConstTensor` with its dtype and shape;
  `types.md`: a rank-0 value keeps an empty shape). The text gives no result type to `tensor(e)` with no dtype argument,
  so no row uses that form.
- The surface form `a[s:e]` on a tensor is the slice operation with step 1: slicing lowers to `Slice` (`language.md`,
  tensor operations), and the default step of `slice` is 1 (`stdlib.md`). Its shape and dtype follow the shape and dtype
  rules of `Slice` (`ir.md`); `a[i]` follows the rules of `Index`.
- Where the text gives no spelling (for example a length query of an array), the row does not use one. Where it gives
  a name but no signature (for example the functions of `std.vec`, `std.string` and `std.map`), no row calls it.
- A standard library entity is used without an import (`stdlib.md`: available without explicit imports). An import
  names only a module that the compiler MUST ship (`stdlib.md`, four pure-MIND modules): `std.vec`, `std.string`,
  `std.map` or `std.io`.
- `print` and `println` take a value whose type implements `Display`, and the text says for no type that it
  implements `Display`. So no row calls them.
