//! `#[bimap]` single-source bijection derive — Phase 1 (ACHIEVE).
//!
//! An `#[bimap]` attribute on an `enum` declares a one-to-one correspondence
//! between the enum's dense ordinal keys (`0..n-1`, exactly the tags the
//! lowering assigns — [`crate::eval::lower`] `enumerate()` over the variant
//! list) and a set of string values (each variant's name by default, or the
//! literal in its `= "…"` slot). From that ONE table this pass synthesises
//! three ordinary functions —
//!
//! ```text
//! pub fn <enum>_count()          -> i64      // cardinality (kills the NUM sentinel)
//! pub fn <enum>_to_str(k: i64)   -> String   // forward:  ordinal -> string
//! pub fn <enum>_from_str(s: String) -> i64   // inverse:  string  -> ordinal (-1 on miss)
//! ```
//!
//! Both directions are emitted in the same pass from the same rows, so a
//! hand-written drifting inverse (the classic `X -> 125` / `125 -> Y` desync)
//! is structurally impossible, not merely discouraged. Phase 1 uses
//! statement-form `if … { return …; }` chains only — exactly the constructs the
//! shipped `examples/bimap_currency` already runs on both backends — avoiding
//! the native-ELF value-if defect family. The Phase-2 canonical seedless
//! perfect-hash inverse swaps only the `from_str` body.
//!
//! ## The pass that synthesises is the pass that refuses.
//! Whole-table bijectivity / niceness validation lives HERE, not in a separate
//! type-checker pass: an invalid table never expands (fail-closed ordering), so
//! lowering never sees it, and there is exactly one owner of the fact "this
//! table is a bijection over a nice set". Diagnostics (sorted by span start):
//!   - E2017 `bimap duplicate key`   — subsumed by the #165 parse-time
//!     dup-variant guard for the enum form; allocated + defensively checked
//!     here for the Phase-3 const-table form.
//!   - E2018 `bimap duplicate value` — two variants map to the same string.
//!     Byte-exact over the DECODED literal bytes (one shared unescape with
//!     codegen — the values are the parser's own `Literal::Str`), via a
//!     `BTreeMap<Vec<u8>, Span>` of first occurrences.
//!   - E2019 `bimap non-total table` — a variant that cannot pair to a string
//!     (a payload-carrying variant, or an empty enum): no total ordinal->string
//!     map exists.
//!   - E2020 `bimap non-nice pair value` — a non-string-literal in the `=` slot
//!     under `#[bimap]` (e.g. `= 5`); the degenerate discriminants
//!     (negative / INT_MAX / bitmask) are loudly inexpressible.
//!   - E2021 `bimap generated-name collision` — a generated name clashes with a
//!     user-defined function, OR with the derive of an earlier `#[bimap]` enum
//!     that snake_case-normalises to the same base; fail-loud (no silent-override
//!     path, no silent-skip path).
//!   - E2022 `reserved marker attribute` — the compiler-only `__bimap_generated`
//!     stamp was written in input source (a forgery, or an impossible
//!     re-expansion). Trusting it would let a one-line attribute skip the enum
//!     before validation and silently suppress the derive; reject fail-loud.
//!
//! ## No mic@3 witness — ever.
//! Correctness is transitively entailed: a wrong or platform-divergent table
//! produces different emitted IR, hence a different `trace_hash`, hence red
//! cross-substrate canaries. Adding an explicit "completeness" witness to the
//! wire format would be the category error the RFC-0023 split verdict rejected.
//! Do NOT introduce one.

use crate::ast::{Attribute, BinOp, Literal, Module, Node, Param, Span, TypeAnn};
use crate::diagnostics::{Diagnostic as PrettyDiagnostic, Span as DiagnosticSpan};

/// Attribute name written by the user to request the derive.
const BIMAP_ATTR: &str = "bimap";
/// Marker attribute stamped onto every synthesised function. It is a
/// COMPILER-ONLY stamp: `expand_bimap` appends the generated fns AFTER scanning
/// the input, so a marker present in input source is always illegitimate and is
/// rejected as E2022. The `__` prefix IS spellable in source (`is_ident_start`
/// accepts `_`), so the marker must be actively REFUSED, never assumed absent —
/// otherwise a one-line `#[__bimap_generated]` on a `<base>_count` fn would make
/// expansion skip the enum before validation, silently suppressing the derive.
const GENERATED_MARKER: &str = "__bimap_generated";

/// Expand every `#[bimap]` enum in `module` in place, appending the three
/// derived functions per enum. Returns the collected validation diagnostics
/// (empty ⇒ success). An enum with ANY violation does not expand (fail-closed).
///
/// Single-shot: `expand_bimap` runs exactly once per parse (from the two parse
/// adapters, each on fresh source). A module that already carries the marker in
/// its input — whether a re-expansion or a user forgery — is rejected (E2022),
/// so double-expansion into duplicate definitions is impossible by construction
/// rather than merely skipped.
pub(crate) fn expand_bimap(
    module: &mut Module,
    source: &str,
    file: Option<&str>,
) -> Vec<PrettyDiagnostic> {
    // Cheap fast-path: modules without a single `#[bimap]` enum pay only this
    // scan (the same cost class as the shipped per-fn `is_test` scan). No
    // allocation, no synthesis, byte-identical emission for every non-user.
    if !module
        .items
        .iter()
        .any(|it| matches!(it, Node::EnumDef { attrs, .. } if has_attr(attrs, BIMAP_ATTR)))
    {
        return Vec::new();
    }

    // User-defined function names (no marker). A user fn with a generated name is
    // an E2021 collision (checked per-enum below).
    let mut user_fn_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Names generated by THIS invocation, populated ONLY as each enum expands
    // below — never seeded from the input. A second `#[bimap]` enum whose base
    // collides with an already-generated one is an E2021, not a silent skip.
    let mut generated_fn_names: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    // E2022 — the marker is a compiler-only stamp; `expand_bimap` appends the
    // generated fns AFTER this scan, so a marker in the INPUT is never legitimate
    // (a user forgery, or an impossible re-expansion). Trusting it would let a
    // one-line `#[__bimap_generated]` on a `<base>_count` fn skip the enum before
    // E2017–E2021 validation, silently suppressing the whole derive. Reject it
    // fail-loud rather than route it into `generated_fn_names`.
    let mut forged: Vec<(String, Span)> = Vec::new();
    for it in &module.items {
        if let Node::FnDef {
            name, attrs, span, ..
        } = it
        {
            if has_attr(attrs, GENERATED_MARKER) {
                forged.push((name.clone(), *span));
            } else {
                user_fn_names.insert(name.clone());
            }
        }
    }
    if !forged.is_empty() {
        let mut diags: Vec<PrettyDiagnostic> = forged
            .into_iter()
            .map(|(name, sp)| {
                err(
                    "E2022",
                    format!(
                        "reserved marker attribute `{GENERATED_MARKER}` may not be written in source; it is stamped only onto compiler-synthesised `#[bimap]` functions (found on `{name}`)"
                    ),
                )
                .with_span(span_of(source, sp, file))
            })
            .collect();
        diags.sort_by_key(|d| {
            d.span
                .as_ref()
                .map(|s| (s.line, s.column))
                .unwrap_or((0, 0))
        });
        return diags;
    }

    let mut diags: Vec<PrettyDiagnostic> = Vec::new();
    let mut synthesized: Vec<Node> = Vec::new();

    for item in &module.items {
        let Node::EnumDef {
            name,
            variants,
            attrs,
            span,
            ..
        } = item
        else {
            continue;
        };
        if !has_attr(attrs, BIMAP_ATTR) {
            continue;
        }

        let base = snake_case(name);
        let count_fn = format!("{base}_count");
        let to_str_fn = format!("{base}_to_str");
        let from_str_fn = format!("{base}_from_str");

        // `generated_fn_names` holds only THIS run's output (input markers are
        // rejected as E2022 above), so a hit here means a second `#[bimap]` enum
        // snake_case-normalises to a base an earlier enum already claimed (e.g.
        // `enum E` and `enum e` both → `e`). That is a real generated-name
        // collision — fail-loud with E2021 rather than silently dropping the
        // second enum's derive.
        if generated_fn_names.contains(&count_fn) {
            diags.push(
                err(
                    "E2021",
                    format!(
                        "bimap generated function `{count_fn}` collides with the derive of an earlier `#[bimap]` enum that snake_case-normalises to the same base `{base}`"
                    ),
                )
                .with_span(span_of(source, *span, file)),
            );
            continue;
        }

        // Build the pair table in DECLARATION ORDER. The ordinal mirrors the
        // lowering's `enumerate()` (eval/lower.rs) exactly, so the derive and
        // the tag assignment cannot disagree about keys.
        //
        // `literal` is the DECODED byte string: `paired` carries the parser's
        // own `Literal::Str` (one shared unescape with codegen), so two source
        // spellings that decode to the same bytes collide at E2018.
        struct Row {
            ordinal: i64,
            literal: String,
            span: Span,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(variants.len());
        let mut enum_diags: Vec<PrettyDiagnostic> = Vec::new();

        // E2019 — a totally-empty enum has no bijection to derive.
        if variants.is_empty() {
            enum_diags.push(
                err("E2019", format!("bimap non-total table: enum `{name}` has no variants — there is no ordinal->string mapping to derive"))
                    .with_span(span_of(source, *span, file)),
            );
        }

        for (ordinal, v) in variants.iter().enumerate() {
            // E2020 — a non-string discriminant is not a nice pair value.
            if v.paired_raw.is_some() {
                enum_diags.push(
                    err(
                        "E2020",
                        format!(
                            "bimap non-nice pair value: variant `{}` has a non-string `= …` discriminant; a bijection enum's pair value must be a string literal",
                            v.name
                        ),
                    )
                    .with_span(span_of(source, v.span, file)),
                );
                continue;
            }
            // E2019 — a payload-carrying variant has no scalar string pairing.
            if !v.payload.is_empty() {
                enum_diags.push(
                    err(
                        "E2019",
                        format!(
                            "bimap non-total table: variant `{}` carries a payload and cannot pair to a string",
                            v.name
                        ),
                    )
                    .with_span(span_of(source, v.span, file)),
                );
                continue;
            }
            let literal = v.paired.clone().unwrap_or_else(|| v.name.clone());
            rows.push(Row {
                ordinal: ordinal as i64,
                literal,
                span: v.span,
            });
        }

        // E2017 — duplicate KEY. For the enum form the ordinals are `0..n-1` by
        // `enumerate()` (unique by construction) and the #165 parse guard already
        // rejects duplicate variant NAMES, so this never fires here; the check is
        // defensive for the Phase-3 const-table form (whose keys are authored).
        {
            let mut seen_keys: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for r in &rows {
                if !seen_keys.insert(r.ordinal) {
                    enum_diags.push(
                        err(
                            "E2017",
                            format!(
                                "bimap duplicate key: ordinal {} appears more than once",
                                r.ordinal
                            ),
                        )
                        .with_span(span_of(source, r.span, file)),
                    );
                }
            }
        }

        // E2018 — duplicate VALUE. Byte-exact over decoded literal bytes.
        {
            let mut first: std::collections::BTreeMap<Vec<u8>, Span> =
                std::collections::BTreeMap::new();
            for r in &rows {
                let key = r.literal.as_bytes().to_vec();
                if let Some(prev) = first.get(&key) {
                    enum_diags.push(
                        err(
                            "E2018",
                            format!(
                                "bimap duplicate value: {:?} is paired by more than one variant of `{name}`",
                                r.literal
                            ),
                        )
                        .with_span(span_of(source, r.span, file))
                        .with_note(format!(
                            "first bound at line {}",
                            span_of(source, *prev, file).line
                        )),
                    );
                } else {
                    first.insert(key, r.span);
                }
            }
        }

        // E2021 — generated-name collision with a user function. Fail-loud: there
        // is no silent-override path. The attr-arg rename escape hatch (Phase 1.x)
        // resolves a genuine legacy clash without touching legacy code.
        for gen_name in [&count_fn, &to_str_fn, &from_str_fn] {
            if user_fn_names.contains(gen_name) {
                enum_diags.push(
                    err(
                        "E2021",
                        format!(
                            "bimap generated function `{gen_name}` collides with a user-defined function of the same name"
                        ),
                    )
                    .with_span(span_of(source, *span, file)),
                );
            }
        }

        // Fail-closed: a table with ANY violation is never synthesised.
        if !enum_diags.is_empty() {
            diags.append(&mut enum_diags);
            continue;
        }

        let enum_span = *span;
        let pairs: Vec<(i64, String)> = rows
            .iter()
            .map(|r| (r.ordinal, r.literal.clone()))
            .collect();
        synthesized.push(make_count_fn(
            &count_fn,
            &base,
            pairs.len() as i64,
            enum_span,
        ));
        synthesized.push(make_to_str_fn(&to_str_fn, &base, &pairs, enum_span));
        synthesized.push(make_from_str_fn(&from_str_fn, &base, &pairs, enum_span));

        // Record the freshly-generated names so a SECOND `#[bimap]` enum in the
        // same module that happened to snake_case to the same base collides via
        // E2021 rather than emitting duplicate fns.
        generated_fn_names.insert(count_fn.clone());
        generated_fn_names.insert(to_str_fn.clone());
        generated_fn_names.insert(from_str_fn.clone());
    }

    module.items.append(&mut synthesized);

    // Deterministic emission: all violations across the module sorted by span
    // start offset (unique usize — no tie ambiguity), byte-identical x86/ARM.
    diags.sort_by_key(|d| {
        d.span
            .as_ref()
            .map(|s| (s.line, s.column))
            .unwrap_or((0, 0))
    });
    diags
}

// ── AST builders (statement-form only — no value-if) ──────────────────────

fn make_count_fn(name: &str, base: &str, n: i64, span: Span) -> Node {
    let body = vec![ret(Node::Lit(Literal::Int(n), span), span)];
    fn_def(name, base, Vec::new(), TypeAnn::ScalarI64, body, span)
}

/// `pub fn <enum>_to_str(k: i64) -> String {
///     if k == <ord> { return "<lit>"; } …
///     return "";
/// }`
fn make_to_str_fn(name: &str, base: &str, pairs: &[(i64, String)], span: Span) -> Node {
    let mut body: Vec<Node> = Vec::with_capacity(pairs.len() + 1);
    for (ord, lit) in pairs {
        let cond = binary(
            BinOp::Eq,
            ident("k", span),
            Node::Lit(Literal::Int(*ord), span),
            span,
        );
        let then = vec![ret(Node::Lit(Literal::Str(lit.clone()), span), span)];
        body.push(if_stmt(cond, then, span));
    }
    body.push(ret(Node::Lit(Literal::Str(String::new()), span), span));
    let params = vec![param("k", TypeAnn::ScalarI64, span)];
    fn_def(
        name,
        base,
        params,
        TypeAnn::Named("String".to_string()),
        body,
        span,
    )
}

/// Synthesise `<enum>_from_str`.
///
/// Phase 2: under `std-surface` the inverse is a canonical **seedless
/// minimal-perfect-hash** O(1) lookup ([`crate::phf`]) when the key set is
/// in-envelope, or a deterministic O(log n) sorted-binary-search fallback when
/// it is not. BOTH std-surface bodies decode key bytes through the unshadowable
/// `__mind_load_i8` intrinsic — never the stdlib `string_eq` — so the
/// user-fn-shadowing miscompile (#177) is closed in every path. The chosen mode
/// (`phf-v1` / `binsearch-v1`) is stamped in the marker attribute args for
/// inspectability. Without `std-surface` (no `while` / no String-record heap
/// primitives) the Phase-1 statement-`if` `string_eq` chain is retained
/// verbatim (mode `linear-v1`) so that low-level builds are byte-identical to
/// Phase 1.
fn make_from_str_fn(name: &str, base: &str, pairs: &[(i64, String)], span: Span) -> Node {
    #[cfg(feature = "std-surface")]
    {
        let keys: Vec<(i64, Vec<u8>)> = pairs
            .iter()
            .map(|(o, l)| (*o, l.as_bytes().to_vec()))
            .collect();
        match crate::phf::build(&keys) {
            crate::phf::PhfOutcome::Built(plan) => make_from_str_phf(name, base, &plan, span),
            crate::phf::PhfOutcome::Fallback => make_from_str_binsearch(name, base, pairs, span),
        }
    }
    #[cfg(not(feature = "std-surface"))]
    {
        make_from_str_string_eq(name, base, pairs, span)
    }
}

/// Phase-1 body — retained ONLY for non-`std-surface` builds (where `while` and
/// the String-record heap primitives are unavailable). Byte-identical to the
/// shipped Phase-1 emission.
#[cfg(not(feature = "std-surface"))]
fn make_from_str_string_eq(name: &str, base: &str, pairs: &[(i64, String)], span: Span) -> Node {
    let mut body: Vec<Node> = Vec::with_capacity(pairs.len() + 1);
    for (ord, lit) in pairs {
        let call = Node::Call {
            callee: "string_eq".to_string(),
            args: vec![ident("s", span), Node::Lit(Literal::Str(lit.clone()), span)],
            span,
        };
        let cond = binary(BinOp::Eq, call, Node::Lit(Literal::Int(1), span), span);
        let then = vec![ret(Node::Lit(Literal::Int(*ord), span), span)];
        body.push(if_stmt(cond, then, span));
    }
    body.push(ret(miss_neg1(span), span));
    let params = vec![param("s", TypeAnn::Named("String".to_string()), span)];
    fn_def_mode(
        name,
        base,
        params,
        TypeAnn::ScalarI64,
        body,
        span,
        "linear-v1",
    )
}

/// The canonical seedless perfect-hash inverse (Slice 1). The three PHF tables
/// (`disp` / `meta` / `pool`) are emitted as pure-ASCII string-literal constants
/// (build-time `.rodata`, decoded once from the [`crate::phf::PhfPlan`]); the
/// lookup reads them via `__mind_load_i8` and hashes with the same twin 31-bit
/// mixers the plan was built with. Miss returns `0 - 1`.
#[cfg(feature = "std-surface")]
fn make_from_str_phf(name: &str, base: &str, plan: &crate::phf::PhfPlan, span: Span) -> Node {
    use crate::ast::BinOp::{Add, Eq, Lt, Mod, Mul, Ne};
    let m = plan.m as i64;
    let np = plan.np as i64;
    let base65 = crate::phf::NIBBLE_BASE;

    let mut body: Vec<Node> = Vec::new();

    // Table constants + their byte-buffer base addresses (String record field 0)
    // + the query string's address / length (record fields 0 / 1).
    body.push(let_(
        "__bm_d",
        Node::Lit(Literal::Str(plan.disp_table()), span),
        span,
    ));
    body.push(let_(
        "__bm_m",
        Node::Lit(Literal::Str(plan.meta_table()), span),
        span,
    ));
    body.push(let_(
        "__bm_p",
        Node::Lit(Literal::Str(plan.pool_table()), span),
        span,
    ));
    // A String value IS its `{ addr, len, cap }` record address; `as i64` is the
    // identity handle-cast (unshadowable, needs no `struct String` in scope), so
    // `load_i64(rec)` reads the byte-buffer address (field 0) and `load_i64(rec +
    // 8)` reads the length (field 1). This avoids depending on the imported
    // `String` struct being registered in the user module's schema.
    body.push(let_(
        "__bm_da",
        load_i64(as_i64(ident("__bm_d", span), span), span),
        span,
    ));
    body.push(let_(
        "__bm_ma",
        load_i64(as_i64(ident("__bm_m", span), span), span),
        span,
    ));
    body.push(let_(
        "__bm_pa",
        load_i64(as_i64(ident("__bm_p", span), span), span),
        span,
    ));
    body.push(let_("__bm_sr", as_i64(ident("s", span), span), span));
    body.push(let_(
        "__bm_sa",
        load_i64(ident("__bm_sr", span), span),
        span,
    ));
    body.push(let_(
        "__bm_n",
        load_i64(
            bin(
                crate::ast::BinOp::Add,
                ident("__bm_sr", span),
                int(8, span),
                span,
            ),
            span,
        ),
        span,
    ));

    // Twin 31-bit mixers over the query bytes.
    body.push(let_mut("__bm_ha", int(0, span), span));
    body.push(let_mut("__bm_hb", int(0, span), span));
    body.push(let_mut("__bm_i", int(0, span), span));
    let hash_body = vec![
        let_(
            "__bm_b",
            call(
                "__mind_load_i8",
                vec![bin(
                    Add,
                    ident("__bm_sa", span),
                    ident("__bm_i", span),
                    span,
                )],
                span,
            ),
            span,
        ),
        assign(
            "__bm_ha",
            bin(
                Mod,
                bin(
                    Add,
                    bin(
                        Mul,
                        ident("__bm_ha", span),
                        int(crate::phf::MULT_A, span),
                        span,
                    ),
                    ident("__bm_b", span),
                    span,
                ),
                int(crate::phf::MOD31, span),
                span,
            ),
            span,
        ),
        assign(
            "__bm_hb",
            bin(
                Mod,
                bin(
                    Add,
                    bin(
                        Mul,
                        ident("__bm_hb", span),
                        int(crate::phf::MULT_B, span),
                        span,
                    ),
                    ident("__bm_b", span),
                    span,
                ),
                int(crate::phf::MOD31, span),
                span,
            ),
            span,
        ),
        assign(
            "__bm_i",
            bin(Add, ident("__bm_i", span), int(1, span), span),
            span,
        ),
    ];
    body.push(while_(
        bin(Lt, ident("__bm_i", span), ident("__bm_n", span), span),
        hash_body,
        span,
    ));

    // bucket = hA % m; disp = LE16 at disp[bucket]; slot = (hB + disp) % np.
    body.push(let_(
        "__bm_bucket",
        bin(Mod, ident("__bm_ha", span), int(m, span), span),
        span,
    ));
    let bx2 = bin(Mul, ident("__bm_bucket", span), int(2, span), span);
    body.push(let_(
        "__bm_dlo",
        dec("__bm_da", bx2.clone(), base65, span),
        span,
    ));
    body.push(let_(
        "__bm_dhi",
        dec("__bm_da", bin(Add, bx2, int(1, span), span), base65, span),
        span,
    ));
    body.push(let_(
        "__bm_disp",
        bin(
            Add,
            ident("__bm_dlo", span),
            bin(Mul, ident("__bm_dhi", span), int(256, span), span),
            span,
        ),
        span,
    ));
    body.push(let_(
        "__bm_slot",
        bin(
            Mod,
            bin(Add, ident("__bm_hb", span), ident("__bm_disp", span), span),
            int(np, span),
            span,
        ),
        span,
    ));

    // meta[slot] = { ord, len, off_lo, off_hi } at logical base slot*4.
    let sx4 = bin(Mul, ident("__bm_slot", span), int(4, span), span);
    body.push(let_(
        "__bm_ord",
        dec("__bm_ma", sx4.clone(), base65, span),
        span,
    ));
    body.push(if_stmt(
        bin(
            Eq,
            ident("__bm_ord", span),
            int(crate::phf::ORD_EMPTY, span),
            span,
        ),
        vec![ret(miss_neg1(span), span)],
        span,
    ));
    body.push(let_(
        "__bm_len",
        dec(
            "__bm_ma",
            bin(Add, sx4.clone(), int(1, span), span),
            base65,
            span,
        ),
        span,
    ));
    body.push(if_stmt(
        bin(Ne, ident("__bm_len", span), ident("__bm_n", span), span),
        vec![ret(miss_neg1(span), span)],
        span,
    ));
    body.push(let_(
        "__bm_olo",
        dec(
            "__bm_ma",
            bin(Add, sx4.clone(), int(2, span), span),
            base65,
            span,
        ),
        span,
    ));
    body.push(let_(
        "__bm_ohi",
        dec("__bm_ma", bin(Add, sx4, int(3, span), span), base65, span),
        span,
    ));
    body.push(let_(
        "__bm_off",
        bin(
            Add,
            ident("__bm_olo", span),
            bin(Mul, ident("__bm_ohi", span), int(256, span), span),
            span,
        ),
        span,
    ));

    // Byte-compare the query against the pooled key at the resolved slot.
    body.push(let_mut("__bm_j", int(0, span), span));
    let cmp_body = vec![
        let_(
            "__bm_pb",
            dec(
                "__bm_pa",
                bin(Add, ident("__bm_off", span), ident("__bm_j", span), span),
                base65,
                span,
            ),
            span,
        ),
        let_(
            "__bm_qb",
            call(
                "__mind_load_i8",
                vec![bin(
                    Add,
                    ident("__bm_sa", span),
                    ident("__bm_j", span),
                    span,
                )],
                span,
            ),
            span,
        ),
        if_stmt(
            bin(Ne, ident("__bm_qb", span), ident("__bm_pb", span), span),
            vec![ret(miss_neg1(span), span)],
            span,
        ),
        assign(
            "__bm_j",
            bin(Add, ident("__bm_j", span), int(1, span), span),
            span,
        ),
    ];
    body.push(while_(
        bin(Lt, ident("__bm_j", span), ident("__bm_n", span), span),
        cmp_body,
        span,
    ));

    body.push(ret(ident("__bm_ord", span), span));
    let params = vec![param("s", TypeAnn::Named("String".to_string()), span)];
    fn_def_mode(name, base, params, TypeAnn::ScalarI64, body, span, "phf-v1")
}

/// Deterministic O(log n) sorted-binary-search fallback for out-of-envelope /
/// construction-resistant key sets (e.g. more than [`crate::phf::MAX_KEYS`]
/// keys). At COMPILE time the rows are sorted by lexicographic byte order of
/// the key — a stable total-order sort with no ties possible (E2018 already
/// rejects duplicate values) — and emitted as two pure-ASCII string-literal
/// tables in the shared PHF nibble encoding ([`crate::phf::enc_byte`]): a meta
/// table of 12 logical bytes per sorted row (`[ord, len, off]`, each a 4-byte
/// little-endian field — wide enough for ordinals / lengths / offsets past the
/// PHF one-byte envelope) and the concatenated sorted-key byte pool. The
/// generated body runs a `while lo <= hi` binary search whose midpoint compare
/// is a lexicographic byte compare through the unshadowable `__mind_load_i8`
/// intrinsic (no `string_eq`, so #177 stays closed here too). Pure integer
/// arithmetic over const tables — byte-identical x86/ARM. Miss returns `0 - 1`.
#[cfg(feature = "std-surface")]
fn make_from_str_binsearch(name: &str, base: &str, pairs: &[(i64, String)], span: Span) -> Node {
    use crate::ast::BinOp::{Add, Div, Eq, Le, Lt, Mul, Sub};
    let base65 = crate::phf::NIBBLE_BASE;

    // Compile-time sort: lexicographic byte order of the decoded key bytes.
    // `<[u8]>::cmp` is a pure byte-wise total order (no locale, no platform
    // dependence), so the emitted tables are identical on every host.
    let mut sorted: Vec<(&[u8], i64)> = pairs.iter().map(|(o, l)| (l.as_bytes(), *o)).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    // Meta table (`[ord, len, off]` × 4-byte LE each) + sorted-key byte pool.
    let mut meta = String::with_capacity(sorted.len() * 24);
    let mut pool = String::new();
    let mut off: usize = 0;
    for (k, ord) in &sorted {
        enc_le4(*ord as u64, &mut meta);
        enc_le4(k.len() as u64, &mut meta);
        enc_le4(off as u64, &mut meta);
        for &b in *k {
            crate::phf::enc_byte(b, &mut pool);
        }
        off += k.len();
    }

    // Table constants, their byte-buffer base addresses (String record field 0),
    // and the query string's address / length — same prelude shape as the PHF
    // body (identity handle-cast + `__mind_load_i64`, no `struct String` needed).
    let mut body: Vec<Node> = vec![
        let_("__bm_t", Node::Lit(Literal::Str(meta), span), span),
        let_("__bm_p", Node::Lit(Literal::Str(pool), span), span),
        let_(
            "__bm_ta",
            load_i64(as_i64(ident("__bm_t", span), span), span),
            span,
        ),
        let_(
            "__bm_pa",
            load_i64(as_i64(ident("__bm_p", span), span), span),
            span,
        ),
        let_("__bm_sr", as_i64(ident("s", span), span), span),
        let_("__bm_sa", load_i64(ident("__bm_sr", span), span), span),
        let_(
            "__bm_n",
            load_i64(bin(Add, ident("__bm_sr", span), int(8, span), span), span),
            span,
        ),
        let_mut("__bm_lo", int(0, span), span),
        let_mut("__bm_hi", int(sorted.len() as i64 - 1, span), span),
    ];

    // One binary-search step over the sorted rows: midpoint row, its key
    // length / pool offset from the meta table, then the byte compare.
    let mut step: Vec<Node> = vec![
        let_(
            "__bm_mid",
            bin(
                Div,
                bin(Add, ident("__bm_lo", span), ident("__bm_hi", span), span),
                int(2, span),
                span,
            ),
            span,
        ),
        // Logical entry base of row `mid` in the meta table.
        let_(
            "__bm_eb",
            bin(Mul, ident("__bm_mid", span), int(12, span), span),
            span,
        ),
        let_(
            "__bm_kl",
            dec4(
                "__bm_ta",
                bin(Add, ident("__bm_eb", span), int(4, span), span),
                base65,
                span,
            ),
            span,
        ),
        let_(
            "__bm_ko",
            dec4(
                "__bm_ta",
                bin(Add, ident("__bm_eb", span), int(8, span), span),
                base65,
                span,
            ),
            span,
        ),
        // min(query len, key len) — the byte-compare window.
        let_mut("__bm_ml", ident("__bm_kl", span), span),
        if_stmt(
            bin(Lt, ident("__bm_n", span), ident("__bm_kl", span), span),
            vec![assign("__bm_ml", ident("__bm_n", span), span)],
            span,
        ),
        // cmp = sign(query - key): first differing byte wins, then length.
        let_mut("__bm_cmp", int(0, span), span),
        let_mut("__bm_i", int(0, span), span),
    ];
    let cmp_body = vec![
        if_stmt(
            bin(Eq, ident("__bm_cmp", span), int(0, span), span),
            vec![
                let_(
                    "__bm_kb",
                    dec(
                        "__bm_pa",
                        bin(Add, ident("__bm_ko", span), ident("__bm_i", span), span),
                        base65,
                        span,
                    ),
                    span,
                ),
                let_(
                    "__bm_qb",
                    call(
                        "__mind_load_i8",
                        vec![bin(
                            Add,
                            ident("__bm_sa", span),
                            ident("__bm_i", span),
                            span,
                        )],
                        span,
                    ),
                    span,
                ),
                if_stmt(
                    bin(Lt, ident("__bm_qb", span), ident("__bm_kb", span), span),
                    vec![assign("__bm_cmp", miss_neg1(span), span)],
                    span,
                ),
                if_stmt(
                    bin(Lt, ident("__bm_kb", span), ident("__bm_qb", span), span),
                    vec![assign("__bm_cmp", int(1, span), span)],
                    span,
                ),
            ],
            span,
        ),
        assign(
            "__bm_i",
            bin(Add, ident("__bm_i", span), int(1, span), span),
            span,
        ),
    ];
    step.push(while_(
        bin(Lt, ident("__bm_i", span), ident("__bm_ml", span), span),
        cmp_body,
        span,
    ));
    // Equal over the compare window: the shorter string sorts first.
    step.push(if_stmt(
        bin(Eq, ident("__bm_cmp", span), int(0, span), span),
        vec![
            if_stmt(
                bin(Lt, ident("__bm_n", span), ident("__bm_kl", span), span),
                vec![assign("__bm_cmp", miss_neg1(span), span)],
                span,
            ),
            if_stmt(
                bin(Lt, ident("__bm_kl", span), ident("__bm_n", span), span),
                vec![assign("__bm_cmp", int(1, span), span)],
                span,
            ),
        ],
        span,
    ));
    // Hit: return the row's ordinal (meta field 0). Otherwise halve the range.
    step.push(if_stmt(
        bin(Eq, ident("__bm_cmp", span), int(0, span), span),
        vec![ret(
            dec4("__bm_ta", ident("__bm_eb", span), base65, span),
            span,
        )],
        span,
    ));
    step.push(if_stmt(
        bin(Lt, ident("__bm_cmp", span), int(0, span), span),
        vec![assign(
            "__bm_hi",
            bin(Sub, ident("__bm_mid", span), int(1, span), span),
            span,
        )],
        span,
    ));
    step.push(if_stmt(
        bin(Lt, int(0, span), ident("__bm_cmp", span), span),
        vec![assign(
            "__bm_lo",
            bin(Add, ident("__bm_mid", span), int(1, span), span),
            span,
        )],
        span,
    ));

    body.push(while_(
        bin(Le, ident("__bm_lo", span), ident("__bm_hi", span), span),
        step,
        span,
    ));
    body.push(ret(miss_neg1(span), span));

    let params = vec![param("s", TypeAnn::Named("String".to_string()), span)];
    fn_def_mode(
        name,
        base,
        params,
        TypeAnn::ScalarI64,
        body,
        span,
        "binsearch-v1",
    )
}

fn fn_def(
    name: &str,
    base: &str,
    params: Vec<Param>,
    ret_type: TypeAnn,
    body: Vec<Node>,
    span: Span,
) -> Node {
    Node::FnDef {
        is_pub: true,
        is_test: false,
        name: name.to_string(),
        type_params: Vec::new(),
        params,
        ret_type: Some(ret_type),
        body,
        reap_threshold: None,
        attrs: vec![Attribute {
            name: GENERATED_MARKER.to_string(),
            args: vec![base.to_string()],
            span,
        }],
        span,
    }
}

/// Like [`fn_def`] but stamps a second marker arg naming the inverse strategy
/// (`phf-v1` / `binsearch-v1` / `linear-v1`) for inspectability. `has_attr` only checks marker
/// presence, so the extra arg is inert to E2022 and never reaches the formatter
/// (fmt opts out of expansion).
fn fn_def_mode(
    name: &str,
    base: &str,
    params: Vec<Param>,
    ret_type: TypeAnn,
    body: Vec<Node>,
    span: Span,
    mode: &str,
) -> Node {
    Node::FnDef {
        is_pub: true,
        is_test: false,
        name: name.to_string(),
        type_params: Vec::new(),
        params,
        ret_type: Some(ret_type),
        body,
        reap_threshold: None,
        attrs: vec![Attribute {
            name: GENERATED_MARKER.to_string(),
            args: vec![base.to_string(), mode.to_string()],
            span,
        }],
        span,
    }
}

/// `0 - 1` — the documented miss sentinel, emitted as a plain subtraction to
/// keep the native-ELF path on the plain-binary form (mirrors the shipped
/// example).
fn miss_neg1(span: Span) -> Node {
    binary(
        BinOp::Sub,
        Node::Lit(Literal::Int(0), span),
        Node::Lit(Literal::Int(1), span),
        span,
    )
}

// ── Phase-2 statement/expression builders (std-surface only) ───────────────

#[cfg(feature = "std-surface")]
fn int(v: i64, span: Span) -> Node {
    Node::Lit(Literal::Int(v), span)
}

#[cfg(feature = "std-surface")]
fn bin(op: BinOp, left: Node, right: Node, span: Span) -> Node {
    binary(op, left, right, span)
}

#[cfg(feature = "std-surface")]
fn call(callee: &str, args: Vec<Node>, span: Span) -> Node {
    Node::Call {
        callee: callee.to_string(),
        args,
        span,
    }
}

/// `expr as i64` — the identity handle-cast used to read a String value as its
/// record address without a `struct String` in scope.
#[cfg(feature = "std-surface")]
fn as_i64(expr: Node, span: Span) -> Node {
    Node::As {
        expr: Box::new(expr),
        ty: TypeAnn::ScalarI64,
        span,
    }
}

/// `__mind_load_i64(addr)` — an 8-byte load via the unshadowable intrinsic.
#[cfg(feature = "std-surface")]
fn load_i64(addr: Node, span: Span) -> Node {
    call("__mind_load_i64", vec![addr], span)
}

#[cfg(feature = "std-surface")]
fn let_(name: &str, value: Node, span: Span) -> Node {
    Node::Let {
        name: name.to_string(),
        mutable: false,
        ann: None,
        value: Box::new(value),
        span,
    }
}

#[cfg(feature = "std-surface")]
fn let_mut(name: &str, value: Node, span: Span) -> Node {
    Node::Let {
        name: name.to_string(),
        mutable: true,
        ann: None,
        value: Box::new(value),
        span,
    }
}

#[cfg(feature = "std-surface")]
fn assign(name: &str, value: Node, span: Span) -> Node {
    Node::Assign {
        name: name.to_string(),
        value: Box::new(value),
        span,
    }
}

#[cfg(feature = "std-surface")]
fn while_(cond: Node, body: Vec<Node>, span: Span) -> Node {
    Node::While {
        cond: Box::new(cond),
        body,
        span,
    }
}

/// Decode one logical table byte at logical index `off` from the ASCII buffer at
/// address variable `addr_var`: `(load_i8(addr + 2*off) - 'A') * 16 +
/// (load_i8(addr + 2*off + 1) - 'A')`. Mirror of [`crate::phf::enc_byte`].
#[cfg(feature = "std-surface")]
fn dec(addr_var: &str, off: Node, base65: i64, span: Span) -> Node {
    use crate::ast::BinOp::{Add, Mul, Sub};
    let two_off = bin(Mul, int(2, span), off, span);
    let hi = bin(
        Sub,
        call(
            "__mind_load_i8",
            vec![bin(Add, ident(addr_var, span), two_off.clone(), span)],
            span,
        ),
        int(base65, span),
        span,
    );
    let lo = bin(
        Sub,
        call(
            "__mind_load_i8",
            vec![bin(
                Add,
                ident(addr_var, span),
                bin(Add, two_off, int(1, span), span),
                span,
            )],
            span,
        ),
        int(base65, span),
        span,
    );
    bin(Add, bin(Mul, hi, int(16, span), span), lo, span)
}

/// Decode a 4-byte little-endian logical field starting at logical index `off`
/// from the ASCII-nibble buffer at address variable `addr_var`:
/// `dec(off) + dec(off+1)*256 + dec(off+2)*65536 + dec(off+3)*16777216`.
/// Mirror of [`enc_le4`].
#[cfg(feature = "std-surface")]
fn dec4(addr_var: &str, off: Node, base65: i64, span: Span) -> Node {
    use crate::ast::BinOp::{Add, Mul};
    let mut out = dec(addr_var, off.clone(), base65, span);
    let mut scale: i64 = 1;
    for i in 1..4i64 {
        scale *= 256;
        out = bin(
            Add,
            out,
            bin(
                Mul,
                dec(
                    addr_var,
                    bin(Add, off.clone(), int(i, span), span),
                    base65,
                    span,
                ),
                int(scale, span),
                span,
            ),
            span,
        );
    }
    out
}

/// Append `v` as 4 little-endian logical bytes (8 ASCII nibble chars, via the
/// shared [`crate::phf::enc_byte`]) — the compile-time mirror of [`dec4`].
/// Ordinals / key lengths / pool offsets are all source-bounded far below
/// `2^32`, so the 4-byte field never truncates.
#[cfg(feature = "std-surface")]
fn enc_le4(v: u64, out: &mut String) {
    for i in 0..4 {
        crate::phf::enc_byte(((v >> (8 * i)) & 0xFF) as u8, out);
    }
}

fn param(name: &str, ty: TypeAnn, span: Span) -> Param {
    Param {
        name: name.to_string(),
        ty,
        span,
    }
}

fn ident(name: &str, span: Span) -> Node {
    Node::Lit(Literal::Ident(name.to_string()), span)
}

fn binary(op: BinOp, left: Node, right: Node, span: Span) -> Node {
    Node::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
        span,
    }
}

fn ret(value: Node, span: Span) -> Node {
    Node::Return {
        value: Some(Box::new(value)),
        span,
    }
}

fn if_stmt(cond: Node, then_branch: Vec<Node>, span: Span) -> Node {
    Node::If {
        cond: Box::new(cond),
        then_branch,
        else_branch: None,
        span,
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|a| a.name == name)
}

fn err(code: &'static str, message: String) -> PrettyDiagnostic {
    PrettyDiagnostic::error("bimap", code, message)
}

fn span_of(source: &str, span: Span, file: Option<&str>) -> DiagnosticSpan {
    DiagnosticSpan::from_offsets(source, span.start(), span.end(), file)
}

/// `Currency` -> `currency`, `HttpStatus` -> `http_status`. Underscore before an
/// interior uppercase, then lowercase throughout. Deterministic, ASCII-only.
fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn count_named(m: &Module, name: &str) -> usize {
        m.items
            .iter()
            .filter(|it| matches!(it, Node::FnDef { name: n, .. } if n == name))
            .count()
    }

    #[test]
    fn second_expansion_is_rejected_as_forgery() {
        // First expansion (inside `parse`) stamps the marker onto the generated
        // fns. A SECOND `expand_bimap` on that already-expanded module now sees
        // marker-bearing fns in its INPUT — indistinguishable from a user forgery
        // — and rejects them as E2022 rather than silently skipping. Double
        // expansion is thus impossible by construction; the compiler calls the
        // pass exactly once per parse, so this path is never hit in production.
        let src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
        let mut m = parse(src).expect("parse+expand");
        assert_eq!(count_named(&m, "currency_to_str"), 1);
        let before = m.items.len();

        let diags = expand_bimap(&mut m, src, None);
        assert!(
            diags.iter().any(|d| d.code == "E2022"),
            "re-expansion of a marked module must be E2022 forgery, got: {diags:?}"
        );
        // Fail-closed: nothing is synthesised on the rejected pass.
        assert_eq!(m.items.len(), before, "rejected re-expansion adds no items");
    }

    #[test]
    fn snake_case_examples() {
        assert_eq!(snake_case("Currency"), "currency");
        assert_eq!(snake_case("HttpStatus"), "http_status");
        assert_eq!(snake_case("A"), "a");
    }
}
