# `mindc fmt` — Canonical Formatter Reference

> Mindcraft Phase 2A. Shipped with mindc v0.6.8.

---

## 1. Overview

`mindc fmt` rewrites `.mind` source files to a canonical, deterministic form.
For a given (source, Mindcraft version) pair the output is byte-identical on
every machine, every run, forever. `mindc fmt --check` is the CI gate: exit 1
if any file deviates from the canonical form, zero writes.

`mindc fmt` operates on `mindc`'s own Rust front-end (`src/parser/`, via
`parse_with_trivia`) — the same lexer/parser/AST that `mindc build`/`check`
use. It does not implement a second MIND parser. **Note:** this is *not* the
same thing as RFC 0007 §7's separately-tracked "self-hosted pure-MIND
front-end" claim (`examples/mindc_mind/main.mind`) — that is a distinct,
independently-gated bootstrap proof that `mindc fmt` does not route through;
see `docs/rfcs/0007-mindcraft.md` §0.

Phase 2A ships all Phase 2A formatting rules. Soft line-wrapping at
`max_line_length` is deferred to Phase 2B.

---

## 2. CLI Surface

```
mindc fmt [PATHS...] [--check] [--diff] [--stdin]
```

### Modes

| Invocation | Effect |
|---|---|
| `mindc fmt` | Format all `*.mind` files in the current directory tree, in-place. |
| `mindc fmt path/to/file.mind` | Format one file in-place. |
| `mindc fmt src/ tests/` | Format all `*.mind` files under two directories in-place. |
| `mindc fmt --check` | Exit 1 if any file would change; no writes. Prints each drifted filename to stderr. |
| `mindc fmt --diff` | Print a unified diff of each file's changes to stdout; no writes. Exit 1 if any file would change. |
| `mindc fmt --stdin` | Read source from stdin; write formatted output to stdout. Cannot be combined with positional paths. |

### Flag rules

- `--check` and `--diff` are mutually exclusive with in-place writes.
- `--stdin` is mutually exclusive with positional paths. Exit code 2 if combined.
- When no positional paths are given, the formatter walks the current directory
  recursively. Files are processed in deterministic (sorted) order.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All files are already canonical (or `--check`/`--diff` found no drift). |
| 1 | One or more files would change (`--check`/`--diff` mode), a file could not be read or written, or a parse error was encountered. |
| 2 | Invalid usage (e.g. `--stdin` combined with path arguments). |

---

## 3. Configuration

Configuration lives in `Mind.toml` under `[mindcraft.format]`. When no
`Mind.toml` is found (walking upward from the current directory), all options
fall back to their canonical defaults.

```toml
[mindcraft.format]
indent_width      = 4    # spaces per nesting level; 1-16; default 4
max_line_length   = 100  # soft column limit (Phase 2B only; no effect in 2A)
trailing_comma    = true # insert trailing comma on multi-line collections
```

### Options

#### `indent_width`

- Type: unsigned integer, 1–16.
- Default: `4`.
- Controls the number of spaces per indentation level. Tabs are never emitted.
- Values below 1 or above 16 are rejected at configuration-load time with a
  diagnostic.

#### `max_line_length`

- Type: unsigned integer, 40–65535.
- Default: `100`.
- **Phase 2A: has no effect.** The formatter does not wrap long lines in Phase
  2A. This setting is read and validated but not acted upon. Soft line-wrapping
  at this limit is implemented in Phase 2B.
- Values below 40 are rejected at configuration-load time (impractically narrow).

#### `trailing_comma`

- Type: boolean.
- Default: `true`.
- When `true`, the formatter inserts a trailing comma after the final element of
  every multi-line collection (struct literals, struct declaration fields,
  function parameters, function call arguments, match arm bodies, array
  literals). Single-line constructs never receive a trailing comma regardless of
  this setting.

---

## 4. Formatter Rules (Phase 2A)

These eight rules are normative for Phase 2A. Each is a fixed point: applying
the rule to already-formatted output produces identical output.

### Rule 1 — Indentation

Every nesting level adds `cfg.indent_width` spaces. The formatter never emits
tab characters. Indentation is computed from the AST depth, not from the
original source indentation.

Nesting increments at:
- Function bodies (`fn ... { ... }`)
- `if`/`else` bodies
- `while` bodies
- `struct` declaration bodies
- `match` arm bodies
- Struct literal bodies
- Array literal bodies (when multi-line)

### Rule 2 — Top-level item spacing

Exactly one blank line separates top-level items (function definitions, struct
definitions, `use` declarations). No leading blank line at the start of the
file. No trailing blank line before EOF. Exactly one newline terminates the
file.

### Rule 3 — Trailing comma

When `cfg.trailing_comma == true`, a trailing comma is inserted after the final
element of any multi-line collection:

- Struct declaration fields.
- Struct literal fields.
- Function parameter lists (when the closing `)` is on its own line).
- Function call argument lists (when the closing `)` is on its own line).
- Match arm bodies.
- Array literals.

Single-line constructs — where the opening and closing delimiters are on the
same line — never receive a trailing comma.

When `cfg.trailing_comma == false`, trailing commas are stripped from all
collection tails.

### Rule 4 — Whitespace normalisation

Inside expressions:

- Multiple consecutive spaces collapse to one space.
- One space after each `,`.
- One space around binary operators: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`,
  `>`, `<=`, `>=`, `&&`, `||`, `=`, `+=`, `-=`, `*=`, `/=`.
- `**` and `..`/`..=` range operators are written tight (no surrounding space).
- No space inside `()`, `[]`, `{}` for single-line constructs: `(x, y)` not
  `( x, y )`.
- No space between a function name and its argument list: `f(x)` not `f (x)`.
- One space before `{` in `if`/`else`/`while`/`fn`/`struct`/`match` keywords:
  `if cond {` not `if cond{`.

### Rule 5 — String literal passthrough

The contents of string literals are passed through bytewise. The formatter
never modifies the text inside `"..."`. This includes strings that contain
what look like MIND operators, comments, or extra whitespace.

### Rule 6 — Blank-line collapse inside bodies

Inside function bodies and other block constructs:

- A maximum of one consecutive blank line is preserved.
- Leading blank lines (immediately after the opening `{`) are stripped.
- Trailing blank lines (immediately before the closing `}`) are stripped.

### Rule 7 — Comment attachment

Three comment attachment rules apply:

1. **Own-line comment**: a comment on a line by itself that immediately precedes
   an item attaches to that item and is emitted at the same indentation level,
   immediately before the item.
2. **Trailing comment**: a comment on the same line as code is preserved at the
   end of that line, separated from the code by one space.
3. **Doc comment** (`///`): a doc-comment immediately preceding an item attaches
   to the item and is re-emitted immediately before it, indented identically to
   the item.
4. **Copyright/license header**: a block of `//` comments at the very top of
   the file (before any item) is preserved verbatim as the file header.

### Rule 8 — Idempotence

`format_source(format_source(src, cfg), cfg) == format_source(src, cfg)` for
all valid MIND source. Every rule above is designed as a fixed point. In
particular:

- No rule introduces constructs that another rule would then reformat.
- Trailing-comma insertion checks for an existing comma before inserting.
- Blank-line collapse is applied once in a single pass.

---

## 5. Known Limitations

### MINDCRAFT-001 — `pub` keyword gap in AST

The current AST does not carry a dedicated `pub` visibility node for items
other than functions. The formatter emits `pub` for `pub fn` declarations but
does not yet handle `pub struct` or top-level `pub use` declarations that are
represented differently in the AST. This is tracked as **MINDCRAFT-001**.
Workaround: `pub struct` and `pub use` pass through unmodified.

### Phase 2B — Soft line-wrap deferred

`max_line_length` is validated at configuration-load time but **has no effect**
in Phase 2A. Lines that exceed the configured limit are not wrapped. Soft
line-wrapping is implemented in Phase 2B.

### No per-target sections

`[mindcraft.cpu]`, `[mindcraft.gpu]`, and `[mindcraft.cerebras]` per-target
configuration sections are planned (RFC 0007 §5) but not yet implemented.
They will be parsed and ignored by Phase 2A; Phase 3+ will activate them.

### No `[mindcraft.overrides]` glob support

The `[[mindcraft.overrides]]` glob-scoped override layer (RFC 0007 §5) is not
yet implemented. The `overrides` key is accepted in `Mind.toml` to avoid parse
errors but its contents are ignored. Implementation is planned for Phase 5.

---

## 6. Usage Examples

Format all `.mind` files under `src/`:

```
mindc fmt src/
```

Dry-run in CI (exit 1 on any drift):

```
mindc fmt --check
```

Preview changes without writing:

```
mindc fmt --diff src/
```

Editor integration (format the buffer via stdin):

```
mindc fmt --stdin < buffer.mind
```
