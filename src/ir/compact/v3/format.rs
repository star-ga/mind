// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Version contract for the established MIC@3 layouts.

/// Magic header bytes for MIC@3 binary format.
pub const MIC3_MAGIC: [u8; 4] = *b"MIC3";

/// MIC@3 format version byte (the version this build *emits*).
///
/// # Version history
///
/// * `0x01` — original layout. `While.exit_ids` and `If.merges` (the
///   control-flow region-exit / merge metadata) were *not* serialised, so a
///   parsed artifact lost them (`exit_ids: Vec::new()` / `merges: Vec::new()`).
///   Consumer-side `mindc verify` therefore reported false define-before-use
///   errors on control-flow programs whose post-region instructions referenced
///   those exit ids (#24).
/// * `0x02` — appends `While.exit_ids` (a ValueId list) after `init_ids`, and
///   `If.merges` (a `(merge_id, then_val, else_val)` triple list) after
///   `branch_bindings`, so control-flow artifacts are independently
///   SSA-verifiable. This changes the mic@3 bytes for any program containing a
///   `While` with non-empty `exit_ids` or an `If` with non-empty `merges`, and
///   therefore changes `trace_hash` for those programs — the intended effect of
///   making the canonical content *complete* (RFC 0021 step-5).
/// * `0x03` — Step D. Appends the structurally-scoped aggregate `value_types`
///   tables (RFC-canonical array typing): a per-`FnDef` sub-list tail-appended
///   after each function's body, and a module-level table appended after the
///   `repr_c_structs` registry (the last `0x02` section). The version is
///   *content-derived*: `IRModule::has_scoped_value_types` chooses
///   `0x03` iff any scope carries a non-empty table, else the byte-for-byte
///   `0x02` layout ([`MIC3_VERSION_BASE`]). Every table is empty in a
///   pipeline-produced module today (populated only by the later semantic
///   slice), so real programs keep emitting `0x02` unchanged. A `0x03` module
///   with all tables empty is not canonical (it would have derived `0x02`); the
///   parser therefore normalizes an all-empty `0x03` stream back to `0x02` on
///   re-emit (WIRE_NON_CANONICAL_VERSION_NORMALIZES).
///
/// The parser ([`super::parse_mic3`]) accepts `0x01`, `0x02`, and `0x03` in a
/// `std-surface` build: a `0x01` artifact is read with empty `exit_ids` /
/// `merges` (the historical behaviour); a `0x02` artifact reads those but no
/// type tables; a `0x03` artifact reads the type tables too. A bare build
/// explicitly refuses `0x03` before decoding the body, because it cannot
/// preserve the aggregate tables on load/re-emit. The emitter writes
/// [`MIC3_VERSION_BASE`] or [`MIC3_VERSION`] per the content predicate above.
pub const MIC3_VERSION: u8 = 0x03;

/// Draft canonical semantic-metadata layout. Emission is selected only by the
/// checked boundary; legacy content continues to derive v0x02/v0x03.
pub const MIC3_VERSION_V04: u8 = 0x04;

/// The additive-append BASE version — the layout emitted when a module carries
/// no scoped aggregate `value_types` table (the overwhelmingly common case).
/// Its bytes are exactly the historical `0x02` layout, so any real program
/// (which never populates a type table before the semantic slice) is
/// byte-for-byte unchanged from before Step D. See [`MIC3_VERSION`] for the
/// version-derivation contract.
pub const MIC3_VERSION_BASE: u8 = 0x02;

/// Lowest MIC@3 format version this build can *read*. The parser is
/// backward-compatible down to this version; the emitter writes
/// [`MIC3_VERSION_BASE`] or [`MIC3_VERSION`] per the content predicate.
pub const MIC3_MIN_READ_VERSION: u8 = 0x01;

/// Opcode of the extern binding record (spec "Extern bindings"). It comes
/// directly after the `ExternFnDecl` record it qualifies and counts as one
/// record of its stream: string index of the declaration name, a kind byte
/// (0x01 library, 0x02 address), then the string index of the library name
/// for kind 0x01. A base binding writes no record, so a module with no `bind`
/// clause keeps its bytes and its `trace_hash`. The version byte is unchanged;
/// an older reader refuses the unknown opcode.
#[cfg_attr(not(feature = "std-surface"), allow(dead_code))]
pub(super) const OP_EXTERN_BIND: u8 = 0x2C;

/// The record count of an instruction stream: one per instruction, plus one
/// for each `ExternFnDecl` that carries a non-base binding.
pub(super) fn stream_records(instrs: &[crate::ir::Instr]) -> usize {
    #[cfg(feature = "std-surface")]
    let bound = instrs
        .iter()
        .filter(|i| matches!(i, crate::ir::Instr::ExternFnDecl { binding, .. } if *binding != Default::default()))
        .count();
    #[cfg(not(feature = "std-surface"))]
    let bound = 0;
    instrs.len() + bound
}

/// Write the `OP_EXTERN_BIND` record of declaration `name`; nothing for a base
/// binding. `idx` gives the string-table index of a string.
#[cfg(feature = "std-surface")]
pub(super) fn write_bind_record<W: std::io::Write>(
    w: &mut W,
    name: &str,
    binding: &crate::ast::ExternBinding,
    idx: impl Fn(&str) -> usize,
) {
    use crate::ir::compact::v2::uleb128_write;
    let (kind, lib) = match binding {
        crate::ast::ExternBinding::Base => return,
        crate::ast::ExternBinding::Library(lib) => (0x01, Some(lib)),
        crate::ast::ExternBinding::Address => (0x02, None),
    };
    w.write_all(&[OP_EXTERN_BIND]).unwrap();
    uleb128_write(w, idx(name) as u64).unwrap();
    w.write_all(&[kind]).unwrap();
    if let Some(lib) = lib {
        uleb128_write(w, idx(lib) as u64).unwrap();
    }
}

/// Put the binding of one `OP_EXTERN_BIND` record (declaration `name`, `kind`
/// byte, `lib` reads the library name) on the base declaration of the same name
/// directly before it. Kind 0x00, an unknown kind, a record with no such
/// declaration directly before it and a second record are refused.
#[cfg(feature = "std-surface")]
pub(super) fn bind_last_decl(
    out: &mut [crate::ir::Instr],
    name: String,
    kind: u8,
    lib: impl FnOnce() -> Result<String, super::Mic3Error>,
) -> Result<(), super::Mic3Error> {
    use crate::ast::ExternBinding;
    let refuse = |what: String| super::Mic3Error { message: what };
    let binding = match kind {
        0x01 => ExternBinding::Library(lib()?),
        0x02 => ExternBinding::Address,
        _ => {
            return Err(refuse(format!(
                "unknown extern binding kind 0x{kind:02X} for `{name}`"
            )));
        }
    };
    match out.last_mut() {
        Some(crate::ir::Instr::ExternFnDecl {
            name: decl,
            binding: slot,
            ..
        }) if *decl == name && *slot == ExternBinding::Base => {
            *slot = binding;
            Ok(())
        }
        _ => Err(refuse(format!(
            "extern binding record for `{name}` is not directly after its base declaration"
        ))),
    }
}

#[cfg(all(test, feature = "std-surface"))]
mod bind_record_tests {
    use crate::ast::{CallConv, ExternBinding};
    use crate::ir::compact::v3::{emit_mic3, parse_mic3};
    use crate::ir::{IRModule, Instr};

    fn decl(name: &str, binding: ExternBinding) -> Instr {
        Instr::ExternFnDecl {
            name: name.into(),
            param_types: vec!["i64".into()],
            ret_type: Some("i64".into()),
            is_varargs: false,
            vararg_hints: vec![],
            callconv: CallConv::C,
            binding,
        }
    }

    fn module(instrs: Vec<Instr>) -> IRModule {
        let mut m = IRModule::new();
        m.instrs = instrs;
        m
    }

    /// The `OP_EXTERN_BIND` record of a one-declaration address module: the
    /// bytes between the base module's decl and its common suffix.
    fn address_record(bound: &[u8], base: &[u8]) -> std::ops::Range<usize> {
        let suffix = (0..base.len())
            .take_while(|&i| bound[bound.len() - 1 - i] == base[base.len() - 1 - i])
            .count();
        bound.len() - suffix - 3..bound.len() - suffix
    }

    #[test]
    fn both_kinds_round_trip_to_a_byte_fixed_point() {
        let m = module(vec![
            decl("crc32", ExternBinding::Library("libz.so.1".into())),
            decl("cuInit", ExternBinding::Address),
            decl("strlen", ExternBinding::Base),
        ]);
        let bytes = emit_mic3(&m);
        let parsed = parse_mic3(&bytes).expect("a bound module parses");
        assert_eq!(emit_mic3(&parsed), bytes, "emit(parse(emit(m))) == emit(m)");
        assert_eq!(format!("{:?}", parsed.instrs), format!("{:?}", m.instrs));
    }

    #[test]
    fn a_base_binding_writes_no_record() {
        let base = emit_mic3(&module(vec![decl("f", ExternBinding::Base)]));
        let bound = emit_mic3(&module(vec![decl("f", ExternBinding::Address)]));
        let rec = address_record(&bound, &base);
        assert_eq!(bound[rec.start], super::OP_EXTERN_BIND);
        assert_eq!(bound[rec.end - 1], 0x02, "address kind byte");
        assert_eq!(
            bound.len(),
            base.len() + 3,
            "one 3-byte record, nothing else grows"
        );
    }

    #[test]
    fn the_parser_refuses_each_bad_record() {
        let base = emit_mic3(&module(vec![decl("f", ExternBinding::Base)]));
        let bound = emit_mic3(&module(vec![decl("f", ExternBinding::Address)]));
        let rec = address_record(&bound, &base);
        for kind in [0x00u8, 0x03, 0xFF] {
            let mut bad = bound.clone();
            bad[rec.end - 1] = kind;
            let err = parse_mic3(&bad).expect_err("a bad kind byte is refused");
            assert!(
                err.message.contains("unknown extern binding kind"),
                "{}",
                err.message
            );
        }
        // A second record for one declaration: duplicate the record (count + 1).
        let count_at = (0..base.len())
            .find(|&i| base[i] != bound[i])
            .expect("count byte");
        let mut twice = bound.clone();
        twice.splice(rec.end..rec.end, bound[rec.clone()].to_vec());
        twice[count_at] += 1;
        let err = parse_mic3(&twice).expect_err("a second record is refused");
        assert!(
            err.message.contains("not directly after"),
            "{}",
            err.message
        );
        // A record that is first in its stream has no declaration before it.
        let lone = module(vec![
            Instr::ConstI64(crate::ir::ValueId(0), 1),
            decl("f", ExternBinding::Address),
        ]);
        let mut moved = emit_mic3(&lone);
        let rec2 = address_record(
            &moved,
            &emit_mic3(&module(vec![
                Instr::ConstI64(crate::ir::ValueId(0), 1),
                decl("f", ExternBinding::Base),
            ])),
        );
        let record: Vec<u8> = moved.drain(rec2.clone()).collect();
        let first = count_at + 1;
        moved.splice(first..first, record);
        let err = parse_mic3(&moved).expect_err("a record before its declaration is refused");
        assert!(
            err.message.contains("not directly after"),
            "{}",
            err.message
        );
    }
}
