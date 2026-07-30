"""Self-host NATIVE-ELF ABI-alignment smoke (FIX #169).

The SysV AMD64 ABI requires rsp ≡ 0 (mod 16) at the point a `call` instruction
executes, so the callee — after the return-address push — sees rsp ≡ 8 (mod 16)
and its own `push rbp` restores 16-alignment for aligned SSE (`movaps`).

The pure-MIND native-ELF entry (`main`, nb_lower_fn in main.mind SECTION 4c)
reaches its body via a hard `movabs rsp, STACK_TOP` switch rather than a `call`,
so it lacks the 8-byte return-address displacement a normally-called function has
on entry. Before FIX #169 that made the entry — and therefore its ENTIRE call
graph — run at rsp ≡ 8 (mod 16), so every emitted `call` fired 8 bytes off the
ABI boundary (latent: the self-contained integer subset tolerates it, but it
breaks any callee that uses aligned SSE moves). FIX #169 switches the entry to
`STACK_TOP - 8`, emulating the missing return address so the shared
`push rbp ; sub rsp,frame(16-mult)` prologue lands the body at rsp ≡ 0.

This smoke is the non-fakeable regression: it emits each call-bearing fixture via
the pure-MIND `selftest_native_elf_h` entry, disassembles the emitted entry's
instruction stream, tracks rsp (mod 16) from the `movabs rsp,imm` switch through
push/pop/sub-rsp/add-rsp, and ASSERTS rsp ≡ 0 (mod 16) at every `call`. It also
runs each emitted ELF and checks its exit code, proving the alignment fix does
not perturb program semantics.

Run:  python3 examples/mindc_mind/self_host_native_elf_alignment_smoke.py
"""

import ctypes
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# Call-bearing fixtures (a subset of self_host_native_elf_smoke.FIXTURES that
# actually emit a `call` in the entry's call graph) + their expected exit codes.
FIXTURES = [
    ("add", "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"
            "fn main() -> i64 {\n    return add(2, 3);\n}\n", 5),
    ("if_ret", "fn f(c: i64) -> i64 {\n    if c == 0 {\n        return 1;\n    }\n"
               "    return 2;\n}\nfn main() -> i64 {\n    return f(0);\n}\n", 1),
    ("value_if", "fn f(a: i64, b: i64) -> i64 {\n    let m: i64 = if a > b { a } else { b };\n"
                 "    return m;\n}\nfn main() -> i64 {\n    return f(3, 7);\n}\n", 7),
    ("recursion", "fn fib(n: i64) -> i64 {\n    if n < 2 {\n        return n;\n    }\n"
                  "    return fib(n - 1) + fib(n - 2);\n}\n"
                  "fn main() -> i64 {\n    return fib(7);\n}\n", 13),
]

_ELF_CODE_START = 0x120  # 64-byte ehdr + 4 * 56-byte phdrs.


def mind_elf(lib, src: bytes) -> bytes:
    """Emit the native ELF for `src` via selftest_native_elf_h. The 32-byte
    PT_NOTE hash is irrelevant to the CODE region's alignment, so we feed zeros."""
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(b"\x00" * 32, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def _imm(op: str) -> int | None:
    """Parse an objdump immediate operand like `0x20` or `0x141000ff8`."""
    m = re.search(r"0x([0-9a-f]+)", op)
    return int(m.group(1), 16) if m else None


def disasm(code: bytes) -> list[tuple[str, str]]:
    """Disassemble a raw x86-64 byte region via objdump; return [(mnemonic, ops)]."""
    objdump = shutil.which("objdump")
    if objdump is None:
        raise RuntimeError("objdump not on PATH — cannot verify call-site alignment")
    with tempfile.NamedTemporaryFile(suffix=".code", delete=False) as f:
        f.write(code)
        path = f.name
    out = subprocess.run(
        [objdump, "-b", "binary", "-m", "i386", "-Mx86-64,intel", "-D", path],
        capture_output=True, text=True,
    ).stdout
    pathlib.Path(path).unlink(missing_ok=True)
    insns: list[tuple[str, str]] = []
    for line in out.splitlines():
        # "   45:\te8 1a 00 00 00       \tcall   0x64"
        m = re.match(r"\s+[0-9a-f]+:\t[0-9a-f ]+\t(\S+)\s*(.*)", line)
        if m:
            insns.append((m.group(1), m.group(2).strip()))
    return insns


def check_call_alignment(name: str, elf: bytes) -> bool:
    """Walk the entry from its `movabs rsp,imm` switch, tracking rsp (mod 16), and
    assert every `call` before the entry's terminating `syscall` fires at rsp ≡ 0."""
    insns = disasm(elf[_ELF_CODE_START:])
    rsp = None  # None until the movabs rsp,imm switch establishes the base.
    saw_call = False
    for mnem, ops in insns:
        if mnem == "movabs" and ops.startswith("rsp,"):
            rsp = (_imm(ops) or 0) & 15
            continue
        if rsp is None:
            continue
        if mnem == "push":
            rsp = (rsp - 8) & 15
        elif mnem == "pop":
            rsp = (rsp + 8) & 15
        elif mnem == "sub" and ops.startswith("rsp,"):
            rsp = (rsp - (_imm(ops) or 0)) & 15
        elif mnem == "add" and ops.startswith("rsp,"):
            rsp = (rsp + (_imm(ops) or 0)) & 15
        elif mnem == "call":
            saw_call = True
            if rsp != 0:
                print(f"  FAIL  {name}: `call {ops}` at rsp ≡ {rsp} (mod 16) — "
                      f"SysV requires 0 (entry is 8B off ABI alignment)")
                return False
        elif mnem == "syscall":
            break  # entry exit — stop before the (bogus) post-code bytes.
    if not saw_call:
        print(f"  FAIL  {name}: no `call` found in entry — fixture did not exercise the bug")
        return False
    print(f"  PASS  {name}: every entry-graph `call` fires at rsp ≡ 0 (mod 16) — SysV-aligned")
    return True


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    if not SO.exists():
        import os
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3

    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, src, expected_exit in FIXTURES:
            elf = mind_elf(lib, src.encode())
            if len(elf) < _ELF_CODE_START:
                print(f"  FAIL  {name}: emitter returned {len(elf)} bytes (no ELF)")
                ok = False
                continue
            ok = check_call_alignment(name, elf) and ok
            code = run_elf(elf, tmp)
            if code != expected_exit:
                print(f"  FAIL  {name}: emitted ELF exited {code} (expected {expected_exit}) "
                      f"— alignment fix perturbed semantics")
                ok = False
            else:
                print(f"  PASS  {name}: emitted ELF runs + exits {code}")

    print("\nALL PASS  (#169: every native-ELF entry-graph call is SysV 16-aligned)"
          if ok else "\nFAIL  (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
