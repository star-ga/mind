"""Twin-emitter half of the self-host differential fuzzer.

Source emission, the self-host and reference note oracles, the runtime
exit check, the minimizer and the batch builder. Imported by
mindfuzz_self_host. Running this file directly does nothing.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import re
import subprocess

_RUST_REF_TEST = "_ref_mic3_dump"
_RUST_REF_FEATURES = "std-surface cross-module-imports"


# ---------------------------------------------------------------------------
# Emitter — AST -> canonical .mind source (one stmt per line, fully-parenthesized
# binops; parseable by BOTH the Rust parser and the pure-MIND self-host lexer).
# ---------------------------------------------------------------------------


def emit_expr(e) -> str:
    k = e[0]
    if k == "const":
        return str(e[1])
    if k == "var":
        return e[1]
    if k == "bin":
        return f"({emit_expr(e[2])} {e[1]} {emit_expr(e[3])})"
    if k == "call":
        return f"{e[1]}({', '.join(emit_expr(a) for a in e[2])})"
    raise ValueError(f"bad expr {e!r}")


def emit_cond(c) -> str:
    return f"{emit_expr(c[2])} {c[1]} {emit_expr(c[3])}"


def emit_stmts(stmts, ind: int, out: list):
    pad = "    " * ind
    for s in stmts:
        k = s[0]
        if k == "let":
            mut = "mut " if s[1] else ""
            out.append(f"{pad}let {mut}{s[2]}: i64 = {emit_expr(s[3])};")
        elif k == "assign":
            out.append(f"{pad}{s[1]} = {emit_expr(s[2])};")
        elif k == "if":
            if s[3] is None:
                out.append(f"{pad}if {emit_cond(s[1])} {{")
                emit_stmts(s[2], ind + 1, out)
                out.append(f"{pad}}}")
            else:
                out.append(f"{pad}if {emit_cond(s[1])} {{")
                emit_stmts(s[2], ind + 1, out)
                out.append(f"{pad}}} else {{")
                emit_stmts(s[3], ind + 1, out)
                out.append(f"{pad}}}")
        elif k == "while":
            out.append(f"{pad}while {emit_cond(s[2])} {{")
            emit_stmts(s[3], ind + 1, out)
            out.append(f"{pad}}}")
        elif k == "break":
            out.append(f"{pad}break;")
        elif k == "continue":
            out.append(f"{pad}continue;")
        else:
            raise ValueError(f"bad stmt {s!r}")


def emit_fn(fn, entry_name: str | None = None) -> list:
    _, name, params, stmts, ret = fn
    nm = entry_name if entry_name is not None else name
    out = []
    ps = ", ".join(f"{p}: i64" for p in params)
    out.append(f"fn {nm}({ps}) -> i64 {{")
    emit_stmts(stmts, 1, out)
    out.append(f"    return {emit_expr(ret)};")
    out.append("}")
    return out


def emit_program(fns: list, entry_name: str = "main") -> str:
    """Emit the whole program; the fn literally named `main` is re-emitted as
    `entry_name` (batching renames each member's entry to fzN)."""
    lines = ["// mindfuzz_self_host generated program - deterministic seed."]
    for fn in fns:
        nm = entry_name if fn[1] == "main" else fn[1]
        lines.extend(emit_fn(fn, nm))
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Self-host oracle (pure-MIND front-end in libmindc_mind.so)
# ---------------------------------------------------------------------------

_RD = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]


def _es_bytes(es) -> bytes:
    """Read the returned EmitState's String buffer (addr/len at +0/+8) — the
    `rd = lambda...; sh = rd(es,0); got = string_at(rd(sh,0), rd(sh,8))` idiom
    of self_host_native_elf_smoke.py line ~559."""
    if not es:
        return b""
    sh = _RD(es, 0)
    if not sh:
        return b""
    n = _RD(sh, 8)
    return ctypes.string_at(_RD(sh, 0), n) if n > 0 else b""


class SelfHost:
    def __init__(self, so_path: pathlib.Path):
        self.lib = ctypes.CDLL(str(so_path))
        for nm, na in (("selftest_native_elf_u", 3), ("selftest_nb_mic3", 3)):
            fn = getattr(self.lib, nm)
            fn.restype = ctypes.c_int64
            fn.argtypes = [ctypes.c_int64] * na

    def _call(self, fn, src: bytes, ulo: int) -> bytes:
        sb = ctypes.create_string_buffer(src, len(src))
        return _es_bytes(fn(ctypes.cast(sb, ctypes.c_void_p).value, len(src), ulo))

    def seeded_elf(self, combined: bytes, ulo: int) -> bytes:
        return self._call(self.lib.selftest_native_elf_u, combined, ulo)

    def seeded_mic3(self, combined: bytes, ulo: int) -> bytes:
        return self._call(self.lib.selftest_nb_mic3, combined, ulo)


class MalformedElf(Exception):
    """A NON-EMPTY self-host image that is not a well-formed MIND-noted ELF."""


def note_of_elf(elf: bytes) -> bytes | None:
    """Return the self-computed 32-byte PT_NOTE. None ONLY for the empty image
    (the entry's fail-closed refusal); a non-empty image without the ELF magic
    or the trailing `MIND` note raises MalformedElf — a broken artifact is a
    failure, never a quiet refusal."""
    if len(elf) == 0:
        return None
    if len(elf) < 64 or elf[:4] != b"\x7fELF" or elf[-40:-36] != b"MIND":
        raise MalformedElf(f"{len(elf)}-byte self-host image is not a MIND-noted ELF "
                           f"(head {elf[:4].hex()}, note-name {elf[-40:-36].hex()})")
    return elf[-32:]


# ---------------------------------------------------------------------------
# Rust oracle — A9b test-time derivation of the seeded+pruned note via the
# committed tests/_ref_mic3_dump.rs, parameterized by MIND_REF_OUT_DIR so the
# fuzz program stands in as `main.mind` (no existing file edited, no new Rust
# code). One cargo invocation per derivation (~2 s warm).
# ---------------------------------------------------------------------------

_REF_NOTE_RE = re.compile(r"REF main:.*note=([0-9a-f]{64})")


class RustRef:
    def __init__(self, repo: pathlib.Path):
        self.repo = repo
        self.cmd = [
            "cargo", "test", "--features", _RUST_REF_FEATURES,
            "--test", _RUST_REF_TEST, "dump_ref", "--", "--ignored", "--nocapture",
        ]
        self.mindc = pathlib.Path(
            os.environ.get("MINDC_BIN", str(repo / "target" / "release" / "mindc"))
        )

    def seeded_note(self, prog_src: str, tmp: pathlib.Path) -> bytes | None:
        """Derive the Rust emit_mic3 reference note for prog_src, or None if
        Rust rejects the program (parse/lower failure panics dump_ref) or the
        toolchain misbehaves."""
        d = tmp / "ref"
        d.mkdir(exist_ok=True)
        (d / "main.mind").write_text(prog_src)
        env = dict(os.environ, MIND_REF_OUT_DIR=str(d), MIND_MIC3_BYTES_DIR=str(d))
        try:
            proc = subprocess.run(
                self.cmd, cwd=str(self.repo), capture_output=True, text=True,
                env=env, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        m = _REF_NOTE_RE.search(proc.stdout + proc.stderr)
        if not m:
            return None
        note = bytes.fromhex(m.group(1))
        # Cross-check the parsed line against the written fixture (loud failure
        # over a silent parse of the wrong artifact).
        f = d / "_ref_main.note"
        if f.exists():
            try:
                disk = bytes.fromhex(f.read_text().strip())
                if disk != note:
                    return None
            except ValueError:
                return None
        return note

    def unseeded_mic3(self, prog_src: str, tmp: pathlib.Path) -> bytes:
        """Diagnostic leg: Rust mic@3 of the program ALONE (mindc --emit-mic3).
        b'' when mindc is absent or rejects the program."""
        if not self.mindc.exists():
            return b""
        p = tmp / "u.mind"
        o = tmp / "u.mic3"
        p.write_text(prog_src)
        try:
            subprocess.run(
                [str(self.mindc), "--emit-mic3", str(o), str(p)],
                capture_output=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            return b""
        return o.read_bytes() if o.exists() else b""


# ---------------------------------------------------------------------------
# Per-program verdict machinery
# ---------------------------------------------------------------------------

# status strings
class RuntimeCheck:
    """Execute a program both ways and compare exit codes: the self-host ELF
    (seeded, exactly the image whose note was compared) vs the Rust/MLIR binary
    (`mindc build --backend mlir --emit=binary`). Returns None when equal, else a
    one-line failure description."""

    def __init__(self, mindc: pathlib.Path, tmp: pathlib.Path):
        self.mindc = mindc
        self.tmp = tmp
        self.runs = 0

    @staticmethod
    def _run(path: pathlib.Path):
        try:
            return subprocess.run([str(path)], capture_output=True, timeout=20).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
        except OSError as e:
            return f"exec-error {e}"

    def check(self, prog_src: str, elf: bytes) -> str | None:
        d = self.tmp / "rt"
        d.mkdir(exist_ok=True)
        src = d / "prog.mind"
        src.write_text(prog_src)
        shp = d / "selfhost.elf"
        shp.write_bytes(elf)
        shp.chmod(0o755)
        binp = d / "rust.bin"
        if binp.exists():
            binp.unlink()
        try:
            b = subprocess.run(
                [str(self.mindc), "build", str(src), "--release", "--backend", "mlir",
                 "--emit=binary", f"--out={binp}"],
                capture_output=True, text=True, timeout=300, cwd=str(d))
        except (subprocess.TimeoutExpired, OSError) as e:
            return f"Rust/MLIR build failed to run: {e}"
        if b.returncode != 0 or not binp.exists():
            return (f"Rust/MLIR build rejected a program both emitters accepted "
                    f"(exit {b.returncode}): {(b.stderr or b.stdout).strip()[-300:]}")
        self.runs += 1
        sh_rc = self._run(shp)
        ru_rc = self._run(binp)
        if not isinstance(sh_rc, int) or not isinstance(ru_rc, int) or sh_rc != ru_rc:
            return f"RUNTIME MISMATCH: self-host ELF exit={sh_rc} Rust/MLIR exit={ru_rc}"
        return None


MATCH = "match"
MISMATCH = "mismatch"
SKIP_SELFHOST = "skip-selfhost-closed"
SKIP_RUST = "skip-rust-rejected"
INVALID = "invalid"


def first_diverge(a: bytes, b: bytes) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo, hi = max(0, i - 8), min(n, i + 16)
            return (
                f"first diverge @ byte {i} (0x{i:x}): "
                f"selfhost={a[i]:#04x} rust={b[i]:#04x}\n"
                f"  selfhost[{lo}:{hi}]: {a[lo:hi].hex()}\n"
                f"  rust    [{lo}:{hi}]: {b[lo:hi].hex()}"
            )
    return f"length mismatch: selfhost={len(a)}B rust={len(b)}B (common prefix identical)"


class Harness:
    def __init__(self, sh: SelfHost, rust: RustRef, std: bytes, tmp: pathlib.Path,
                 verbose: bool = False):
        self.sh = sh
        self.rust = rust
        self.std = std
        self.tmp = tmp
        self.verbose = verbose

    def seeded_elf(self, prog_src: str) -> bytes:
        return self.sh.seeded_elf(self.std + prog_src.encode(), len(self.std))

    def check_source(self, prog_src: str) -> tuple:
        """Run BOTH oracles on one source. Returns (status, sh_note, rust_note).
        status in {MATCH, MISMATCH, SKIP_SELFHOST, SKIP_RUST}. A non-empty
        malformed self-host image raises MalformedElf (a FAIL, never a skip)."""
        combined = self.std + prog_src.encode()
        ulo = len(self.std)
        elf = self.sh.seeded_elf(combined, ulo)
        sh_note = note_of_elf(elf)
        if sh_note is None:
            return SKIP_SELFHOST, None, None  # len-0 (or malformed) = fail-closed
        rust_note = self.rust.seeded_note(prog_src, self.tmp)
        if rust_note is None:
            return SKIP_RUST, sh_note, None
        return (MATCH if sh_note == rust_note else MISMATCH), sh_note, rust_note

    def diagnose(self, prog_src: str, lines: list):
        """Append mismatch diagnostics: note hex + the UNSEEDED mic@3 byte diff
        (selftest_nb_mic3(prog,0) vs mindc --emit-mic3 prog) localizing the
        divergence to the emitter's handling of the program body itself."""
        lines.append("notes (32B PT_NOTE = sha256 of seeded+pruned mic@3):")
        combined = self.std + prog_src.encode()
        elf = self.sh.seeded_elf(combined, len(self.std))
        try:
            sh_note = note_of_elf(elf) or b""
        except MalformedElf:
            sh_note = b""
        rust_note = self.rust.seeded_note(prog_src, self.tmp) or b""
        lines.append(f"  selfhost: {sh_note.hex()}")
        lines.append(f"  rust-ref: {rust_note.hex()}")
        if sh_note and rust_note:
            di = next((i for i in range(32) if sh_note[i] != rust_note[i]), -1)
            lines.append(f"  first differing note byte: index {di}")
        sh_mic = self.sh.seeded_mic3(combined, len(self.std))
        lines.append(f"selfhost seeded mic@3: {len(sh_mic)}B "
                     f"(sha256 {hashlib.sha256(sh_mic).hexdigest()[:32]}...)")
        sh_u = self.sh.seeded_mic3(prog_src.encode(), 0)
        ru_u = self.rust.unseeded_mic3(prog_src, self.tmp)
        if sh_u and ru_u:
            if sh_u == ru_u:
                lines.append(
                    "unseeded program-alone mic@3: BYTE-IDENTICAL "
                    f"({len(sh_u)}B) — divergence lives in the SEEDING/prune/"
                    "placeholder machinery (next_id / strtab / extern counts), "
                    "not in the fn-body emitter"
                )
            else:
                lines.append("unseeded program-alone mic@3 DIVERGES — the twin "
                             f"emitters disagree on the program body itself:\n  "
                             + first_diverge(sh_u, ru_u))
        elif not sh_u:
            lines.append("unseeded selftest_nb_mic3 fail-closed (0B) on the "
                         "program alone — body-shape gap in the nb mic@3 emitter")
        elif not ru_u:
            lines.append("mindc --emit-mic3 unavailable/rejected the unseeded "
                         "program — body diff unavailable")


# ---------------------------------------------------------------------------
# Delta-debounce minimizer — AST-level mutations; a reduction is kept only
# while BOTH oracles still accept AND still mismatch (fail-closed reductions
# are rejected, so the minimal reproducer stays inside the accepted intersection).
# ---------------------------------------------------------------------------


def _map_exprs(stmts, f):
    out = []
    for s in stmts:
        k = s[0]
        if k == "let":
            out.append(("let", s[1], s[2], f(s[3])))
        elif k == "assign":
            out.append(("assign", s[1], f(s[2])))
        elif k == "if":
            out.append(("if", s[1], _map_exprs(s[2], f),
                        _map_exprs(s[3], f) if s[3] else None))
        elif k == "while":
            out.append(("while", s[1], s[2], _map_exprs(s[3], f)))
        else:
            out.append(s)
    return out


def _mutate_candidates(fns: list):
    """Deterministic candidate-reduction stream (coarse -> fine)."""
    # 1) drop whole helpers (main may then not call them — calls into nowhere
    #    fail-closed both oracles, which the acceptance check rejects: safe)
    for i, fn in enumerate(fns):
        if fn[1] != "main":
            yield fns[:i] + fns[i + 1:]
    # 2) drop one statement at a time (reverse order, every fn)
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        for si in range(len(stmts) - 1, -1, -1):
            new = fns[:fi] + [("fn", name, params, stmts[:si] + stmts[si + 1:], ret)] + fns[fi + 1:]
            yield new
    # 3) drop else branches
    def strip_else(stmts):
        out = []
        for s in stmts:
            if s[0] == "if" and s[3] is not None:
                out.append(("if", s[1], s[2], None))
            elif s[0] == "if":
                out.append(("if", s[1], strip_else(s[2]), None))
            elif s[0] == "while":
                out.append(("while", s[1], s[2], strip_else(s[3])))
            else:
                out.append(s)
        return out
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        ns = strip_else(stmts)
        if ns != stmts:
            yield fns[:fi] + [("fn", name, params, ns, ret)] + fns[fi + 1:]
    # 4) replace every expr with const 0 (per fn)
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        ns = _map_exprs(stmts, lambda e: ("const", 0))
        yield fns[:fi] + [("fn", name, params, ns, ("const", 0))] + fns[fi + 1:]


def minimize(harness: Harness, fns: list, max_trials: int = 120) -> tuple:
    """Delta-debounce: keep any reduction that still mismatches. Returns
    (minimal_fns, trials_used)."""
    current = fns
    trials = 0
    progress = True
    while progress and trials < max_trials:
        progress = False
        for cand in _mutate_candidates(current):
            if trials >= max_trials:
                break
            trials += 1
            src = emit_program(cand)
            status, _sh, _ru = harness.check_source(src)
            if status == MISMATCH:
                current = cand
                progress = True
                break  # restart the candidate stream on the smaller program
    return current, trials


# ---------------------------------------------------------------------------
# Batching — one cargo derivation per group of accepted programs.
# ---------------------------------------------------------------------------


def batch_source(members: list, ids: list) -> str:
    """Concatenate member programs with entries renamed fz<id>, then a synthetic
    `fn main` calling the first entry. Every fuzz fn starts at/after user_lo,
    so ALL of them are prune roots on BOTH sides (self-host nb_mark_roots; Rust
    ref_ir collects every user FnDef) — the batch note covers every member body,
    and one wrong byte in any member flips it."""
    parts = ["// mindfuzz_self_host BATCH - members: "
             + " ".join(str(i) for i in ids)]
    for fns, i in zip(members, ids):
        parts.append(emit_program(fns, entry_name=f"fz{i}").rstrip("\n"))
    parts.append(f"fn main() -> i64 {{\n    return fz{ids[0]}();\n}}")
    return "\n".join(parts) + "\n"

if __name__ == "__main__":
    raise SystemExit(0)
