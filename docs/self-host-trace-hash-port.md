# #17 — Self-compute the native PT_NOTE (pure-MIND trace-hash port)

**Status (2026-07-01): CLOSED.** All 6 sub-milestones landed (commits `0711164`
through `457eff5` on `feat/cross-module-enums`). The full-scale no-feed mic@3
self-compute is byte-identical to the Rust oracle (360243 B), and the resulting
native ELF (incl. self-computed PT_NOTE) is byte-identical with zero oracle
bytes fed anywhere in the loop (1156476 B) — verified by a new no-feed
full-scale rung in `self_host_native_elf_smoke.py`. This was the last oracle
tie in the native-ELF self-host (#14). **Phase 1.4 (delete `src/native`) is
now unblocked** — see the roadmap doc before starting that.

## Goal

The native ELF's 32-byte `PT_NOTE` = `ir_trace_hash(ir) = mini_sha256(emit_mic3(ir))`
(`src/ir/evidence.rs:72`), where `ir` is the **seeded + pruned combined IRModule**.
Today the pure-MIND `nb_trace_hash` fails closed and the smoke feeds the oracle's hash.
Porting `emit_mic3(combined-pruned IR)` to pure MIND makes the note **self-computed** →
the no-feed native rung flips OPEN→PASS → then Phase 1.4 (delete `src/native`).

The SHA-256 leg is already ported + proven (`nb_sha256` / `nb_sha256_note`). The open
leg is reproducing `emit_mic3(combined-pruned IR)` byte-exact.

## Authoritative target (gate against this)

Dump the exact bytes the note hashes with the debug hook:

```bash
MIND_DUMP_MIC3=/tmp/add_combined.mic3 mind-native /tmp/add_fix.mind /tmp/add.elf
# sha256(/tmp/add_combined.mic3) == the ELF PT_NOTE  (verified)
```

Decoded `add` fixture target — **5447 B**, sha `e912443707c23b57…`:

| field | value |
|-------|-------|
| header | `MIC3` + version `0x02` |
| string table | **84 entries** (uleb count, then `uleb(len)+bytes` each), combined first-seen interning order — extern names (`open`/`close`/`read`/`write`/`lseek`/`unlink`/…), type strings (`i64`, `!llvm.ptr`), fn names |
| next_id | **618** |
| exports | 0 (uleb count, then sorted strtab-idxs) |
| instr_count | **1305** (then `emit_instr` each) |
| registries | struct_defs, then const_array_defs, then repr_c_structs (each `uleb(count)` + entries) |

Wire format: `src/ir/compact/v3/emit.rs:716` (`emit_mic3`). Opcodes there
(`OP_CONST_I64=0x01`, `OP_OUTPUT=0x13`, `OP_FN_DEF=0x15`, …).

## Mechanism (from `src/eval/lower.rs:780-925` + `src/bin/mind-native.rs:158`)

1. **Seed** — concat all 21 `parsed_stdlib_modules()` items + the user module items into
   `combined` (`mind-native.rs:77-84`).
2. **`lower_to_ir(&combined)`** — walk EVERY top-level item in order; each allocates a
   fresh id (`ir.fresh()`) and pushes `{value-instr, Output(id)}`:
   - `StructDef` / `EnumDef` → `ConstI64(id,0)` + `Output(id)` (metadata → `struct_defs` /
     `enum_variant_tags` / `boxed_enums` / … registries).
   - `Const`(array) → `ConstArray` + `Output`.
   - `FnDef` → the real `FnDef` body instr + `Output`.
   - extern → `ExternFnDecl`.
   - `Export` → extends `exports` (no instr).
   ⇒ `next_id ≈ item-count` (618); instr stream ≈ one `{value,Output}` pair per item (1305).
3. **`prune_unreachable_fns`** — BFS reachable-set from the **user file's** fn names over
   `Instr::Call` edges, then `ir.instrs.retain(|i| match i { FnDef{name,..} =>
   reachable.contains(name), _ => true })`. So it **keeps every non-FnDef instr** (all
   placeholders, externs, Outputs) and **drops only unreachable FnDef instrs entirely**
   (no placeholder left in their place). `next_id` is NOT renumbered (stays 618).
4. **`emit_mic3(pruned ir)`** — collect strings FROM the retained instrs (+ exports +
   registries), then header / strtab / next_id / exports / instrs / registries.

## Coverage map (measured via `/tmp/nfn_cmp.py`, 2026-06-30)

The nfn (`selftest_mic3_module_nfn`) already emits these top-level items byte-identical:
- ✅ `struct` def · ✅ `enum` def (bare and `= N` explicit-value) · ✅ `fn` bodies (incl
  `while`, since 468bd4e).
- ❌ **gap:** `E::Variant as i64` value-access fails closed.
- ⚠ extern uses block syntax `extern "C" { fn …; }` (not `extern "C" fn …;`) — confirm the
  stdlib form + that the nfn emits `ExternFnDecl`.
- `main.mind` has **zero** top-level enums, so the flip never exercised enum emission —
  the enum coverage above is newly measured.

## The port (architectural crux)

The nfn **streams** mic@3 (lower + emit in one pass), so it can't prune mid-stream. The
port must **materialise** the combined instr list + registries first, **prune** it, then
run the existing `emit_mic3` header/strtab/instr machinery over the pruned list. Steps:

1. Parse the combined (already fed by `selftest_native_elf_hb`); the kept-set machinery
   (`nb_collect_fns`/`nb_mark_roots`/`nb_reach_fixpoint`/`nb_build_kept`) already computes
   the reachable fn-name set — reuse it as the prune predicate.
2. Materialise the top-level instr stream + registries for the combined items with fresh-id
   numbering (mirror `lower.rs:780-925` per-item arms) → `next_id`, `struct_defs`, etc.
3. Emit `struct/enum/const/fn` per the nfn (present) + close the `E::Variant as i64` and
   `ExternFnDecl` gaps.
4. Drive `emit_mic3_header`/`emit_mic3_strtab`/instr-loop/registries over the pruned stream.
5. Feed `.buf` addr/len to `nb_sha256_note` in `nb_trace_hash`; reroute the `_h`/`_hb`
   shims to the no-hash `selftest_native_elf(_u)`; drop the `hash_addr` arg.

Gate every step byte-for-byte against `/tmp/add_combined.mic3` (5447 B), then `main.mind`'s
own target (≈313 KB, ~4358 instrs). Done ⇒ the no-feed rung self-computes the note; all
three self-host gates stay byte-identical.
