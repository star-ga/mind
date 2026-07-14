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
//!     user-defined function; fail-loud (there is no silent-override path).
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
/// Marker attribute stamped onto every synthesised function. Distinguishes a
/// previously-synthesised fn (skip — idempotent) from a coincidentally
/// same-named USER fn (E2021 collision). Never written by users (the `__`
/// prefix is not producible through the attribute grammar's normal use).
const GENERATED_MARKER: &str = "__bimap_generated";

/// Expand every `#[bimap]` enum in `module` in place, appending the three
/// derived functions per enum. Returns the collected validation diagnostics
/// (empty ⇒ success). An enum with ANY violation does not expand (fail-closed).
///
/// Idempotent: an enum whose derived functions are already present as
/// marker-stamped synthesised fns is skipped, so a module that flows through
/// two module-producing front-ends (e.g. checked, then evaluated) — or a
/// re-entrant / composed seam — is never double-expanded into duplicate
/// definitions.
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

    // Names already defined, split by whether WE stamped them. A user fn (no
    // marker) with a generated name is an E2021 collision; a marked fn means
    // this enum was already expanded (idempotency skip).
    let mut user_fn_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut generated_fn_names: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    for it in &module.items {
        if let Node::FnDef { name, attrs, .. } = it {
            if has_attr(attrs, GENERATED_MARKER) {
                generated_fn_names.insert(name.clone());
            } else {
                user_fn_names.insert(name.clone());
            }
        }
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

        // Idempotency: already expanded (our marker present) ⇒ skip silently.
        if generated_fn_names.contains(&count_fn) {
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
            if v.paired_is_nonstring {
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

/// `pub fn <enum>_from_str(s: String) -> i64 {
///     if string_eq(s, "<lit>") == 1 { return <ord>; } …   // declaration order
///     return -1;
/// }`
fn make_from_str_fn(name: &str, base: &str, pairs: &[(i64, String)], span: Span) -> Node {
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
    // `-1` on miss (documented policy). Emitted as `0 - 1` to mirror the shipped
    // example and keep the native-ELF path on the plain-binary form.
    let miss = binary(
        BinOp::Sub,
        Node::Lit(Literal::Int(0), span),
        Node::Lit(Literal::Int(1), span),
        span,
    );
    body.push(ret(miss, span));
    let params = vec![param("s", TypeAnn::Named("String".to_string()), span)];
    fn_def(name, base, params, TypeAnn::ScalarI64, body, span)
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
    fn idempotent_second_expansion_adds_nothing() {
        // First expansion happens inside `parse`. A SECOND `expand_bimap` on the
        // already-expanded module (the re-entrant / composed-seam case) must
        // synthesise NOTHING and emit ZERO diagnostics — the marker-stamped
        // generated fns are recognised and skipped.
        let src = "#[bimap]\nenum Currency { AUD, JPY, USD }\n";
        let mut m = parse(src).expect("parse+expand");
        let before = m.items.len();
        assert_eq!(count_named(&m, "currency_to_str"), 1);

        let diags = expand_bimap(&mut m, src, None);
        assert!(
            diags.is_empty(),
            "second expansion must not diagnose: {diags:?}"
        );
        assert_eq!(m.items.len(), before, "second expansion must add no items");
        assert_eq!(
            count_named(&m, "currency_to_str"),
            1,
            "no duplicate generated fn after re-expansion"
        );
    }

    #[test]
    fn snake_case_examples() {
        assert_eq!(snake_case("Currency"), "currency");
        assert_eq!(snake_case("HttpStatus"), "http_status");
        assert_eq!(snake_case("A"), "a");
    }
}
