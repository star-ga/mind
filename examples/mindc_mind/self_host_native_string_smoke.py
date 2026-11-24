"""
Self-host NATIVE-ELF STRING-LITERAL smoke (Rust-independence #14) — proves the
pure-MIND front-end (examples/mindc_mind/main.mind) lowers a source string literal
to a read-only rodata PT_LOAD segment, entirely in pure MIND with ZERO Rust codegen
in the emit path. Each literal site puts its bytes there, then (8-aligned by
address) its borrowed String handle { addr = 0x800000 + off, len, cap = 0 }, ONCE
at compile time. The literal expression is that handle's address (one movabs):
nothing is allocated at run time, so a literal in a loop costs no arena.

There is NO frozen byte-oracle for this construct (it is NEW — the deleted Rust
`src/native` backend never emitted rodata/strings), so the gate is
EXECUTION-CORRECTNESS + BYTE-DETERMINISM + LAYOUT, not byte-identity vs a Rust
reference. Two sections:

  A. the `.so` section (MINDC_SO, selftest_native_elf_h; skipped when MINDC_SO is not set):
     each fixture runs, exits with the expected value and compiles to identical
     bytes twice; the rodata PT_LOAD is R-only at 0x800000.
  B. the stage1 section (always runs): a compiler ELF built from THIS tree's sources by
     the frozen seed (--compiler ELF names one, else one hop is minted) compiles the
     programs of testdata/native_string_handle/ through the stdin protocol, on the
     default and on the bounded layout. Every image runs; its exit status, stdout and
     peak resident size are checked against values computed here from the program
     text (never from a compiler). A literal in a 50-million-iteration loop (before the
     handle was built once: 32 arena bytes per evaluation, SIGILL at the arena guard
     after 2^25) ends with the right value in a few MiB. The rodata of each image is
     parsed: every site must have its bytes, a zero gap, a 24-byte handle at an 8-aligned
     address with {0x800000 + off, len, 0}, and the code must load that address.
     testdata/string_handle_limits/ holds eight programs: six whose run the change alters (five
     loops of 2^25 evaluations or more, which ended at the arena guard before on the default
     layout and ran to the same end on the bounded layout, where the old run used about 1.5 GB;
     and a store through the handle, which did not fault before) and two control rows whose run it does
     not alter (a loop of 2^25 - 1 evaluations, a store through the literal's bytes). They sit
     outside testdata/native_*/, which the two-compiler identity run compares run for run.
     A program whose only literal is a `const` String that no code reads gets no image: the string
     rows of the compiler refuse every `const` String (lane.string.const, the unused_const row).
     handle_driver.mind (the minted unit with a driver in place of the compiler's main)
     feeds literals to nb_expr with the rodata header preset: every offset next to the
     8 MiB window edge x every length 0..17 must be accepted or refused exactly when
     align8(off + len) + 24 <= window, a bad escape is refused, and the handle words,
     the gap and the code bytes read back as expected.
  D. the window filled by many literals (same compiler, both layouts): chains of 40, 50 and 129
     fns with a 65,000-byte literal each, and 128 of them plus 2048 literals of 1 to 7 bytes
     (32 window bytes each), which fill the 8 MiB window to the byte, build, run to the sum of
     the lengths and have the rodata layout above; one site more is refused by name
     (lane.string.literal_capacity) at the last literal. The minted stage1 runs on the bounded
     layout (384 MiB stack): a rodata copy with a call frame per byte faulted there from 50 fns.

Run:  python3 examples/mindc_mind/self_host_native_string_smoke.py [--compiler ELF]
      MINDC_SO=<fresh .so> adds section A.
"""

import argparse
import ctypes
import os
import pathlib
import re
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.resolve().parents[1]
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
from _selfhost_sources import (start_child, BOUNDED_LAYOUT, COMPILER_COMPONENTS, FROZEN_SEED, LAYOUT_MAGIC,  # noqa: E402
                               _run_compiler, capture_bundle, driver_compile, self_mode_image)
from _selfhost_sources import core_limit_argv  # noqa: E402

# A dummy 32-byte PT_NOTE trace hash. This smoke gates runtime behavior +
# determinism of the CODE + rodata regions, not the note (there is no oracle to
# match), so any fixed 32 bytes is fine — both compiles feed the SAME hash, so the
# note stays constant and cannot mask a real code-region divergence.
_DUMMY_HASH = bytes(range(32))

# (name, source, expected_exit, expected_phdrs). Each binds a string literal and
# returns an observable i64 (exit code = low 8 bits of the returned value).
# expected_phdrs: every literal site puts a 24-byte handle in the rodata segment, so
# any program with a literal (an empty one too) has the 5th, rodata phdr.
FIXTURES = [
    # len field of the borrowed handle: s.len via __mind_load_i64(s + 8). "hello"
    # -> 5. Proves the ast_str_lit arm ran, the accumulator bumped, and the handle
    # carries the correct length.
    (
        "strlen",
        (
            "fn main() -> i64 {\n"
            '    let s = "hello";\n'
            "    return __mind_load_i64(s + 8);\n"
            "}\n"
        ),
        5,
        5,
    ),
    # first literal byte via a DOUBLE dereference: addr field (s+0) -> rodata
    # vaddr, then the first 8 bytes at that vaddr. "hi" -> 0x6968; exit = 0x68 =
    # 'h' = 104. Proves the rodata PT_LOAD segment maps at 0x800000 and holds the
    # copied bytes at runtime (the whole point of this slice).
    (
        "strbyte",
        (
            "fn main() -> i64 {\n"
            '    let s = "hi";\n'
            "    return __mind_load_i64(__mind_load_i64(s));\n"
            "}\n"
        ),
        104,
        5,
    ),
    # EDGE: an all-EMPTY-literal program has a handle {0x800000, 0, 0} in the rodata
    # segment, so the segment is emitted; s.len (s + 8) == 0, exit 0.
    (
        "empty_only",
        (
            "fn main() -> i64 {\n"
            '    let s = "";\n'
            "    return __mind_load_i64(s + 8);\n"
            "}\n"
        ),
        0,
        5,
    ),
    # EDGE: a "" beside a non-empty literal. Each has its own handle.
    # exit = a.len + b.len = 0 + 2 = 2.
    (
        "mixed_empty",
        (
            "fn main() -> i64 {\n"
            '    let a = "";\n'
            '    let b = "hi";\n'
            "    return __mind_load_i64(a + 8) + __mind_load_i64(b + 8);\n"
            "}\n"
        ),
        2,
        5,
    ),
]


def _phdrs(elf: bytes):
    """Parse the ELF64 program headers as
    [(p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align), ...]."""
    e_phoff = struct.unpack_from("<Q", elf, 0x20)[0]
    e_phnum = struct.unpack_from("<H", elf, 0x38)[0]
    return [struct.unpack_from("<IIQQQQQQ", elf, e_phoff + i * 56) for i in range(e_phnum)]


def mind_string_elf(lib, src: bytes) -> bytes:
    """Lower `src` through the pure-MIND native-ELF front-end (selftest_native_elf_h,
    dummy note) and return the emitted ELF bytes (empty on a fail-closed refusal)."""
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(_DUMMY_HASH, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    length = rd(sh, 8)
    if length <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), length)


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run(core_limit_argv([str(p)])).returncode


def so_section() -> tuple[int, int] | None:
    """Section A: (ran, failed), or None when MINDC_SO is not set (a skip, said so). The resolver
    refuses a stale oracle, so it is called only when the caller asks for the .so."""
    if not os.environ.get("MINDC_SO"):
        print("SKIP section A: MINDC_SO is not set (section B needs no .so)")
        return None
    so = resolve_so()
    lib = ctypes.CDLL(str(so))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3
    ran = fail = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, src, expected_exit, expected_phdrs in FIXTURES:
            elf = mind_string_elf(lib, src.encode())
            ran += 1
            if elf[:4] != b"\x7fELF":
                print(f"  FAIL  A {name}: pure-MIND emit failed closed or is not an ELF ({elf[:4]!r})")
                fail += 1
                continue
            ph = _phdrs(elf)
            ok = len(ph) == expected_phdrs
            if len(ph) == 5:
                p_type, p_flags, p_off, p_vaddr, _pp, _fsz, _msz, p_align = ph[4]
                # R only (never RW/RX), at 0x800000, file offset congruent to the vaddr
                ok = ok and p_type == 1 and p_flags == 4 and p_vaddr == 0x800000 and p_off % p_align == p_vaddr % p_align
            code = run_elf(elf, tmp)
            same = mind_string_elf(lib, src.encode()) == elf
            ok = ok and code == expected_exit and same
            seen = (f"phdrs={len(ph)} (expected {expected_phdrs}) exit={code} (expected {expected_exit}) "
                    f"recompile {'identical' if same else 'DIVERGED'} [{len(elf)} bytes]")
            if ok:
                print(f"  PASS  A {name} {seen}")
            else:
                print(f"  FAIL  A {name} {seen}")
                fail += 1
    return ran, fail


# ---- section B: programs through a compiler ELF -----------------------------------------

DATA = _HERE / "testdata" / "native_string_handle"
LIMITS = _HERE / "testdata" / "string_handle_limits"
CAP = 0x1000000 - 0x800000  # the rodata window: nb_arena_addr - nb_rodata_addr
RSS_MAX_KIB = 32 * 1024  # the old arm held 32 bytes of arena per evaluation: 1 GiB after 2^25
ESCAPES = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34}


def literal_texts(source: str) -> list[str]:
    """Content of each string literal of a source, in text order (a `//` comment ends a line)."""
    out = []
    for line in source.splitlines():
        i = 0
        while i < len(line) and not line.startswith("//", i):
            if line[i] == '"':
                j = i + 1
                while j < len(line) and line[j] != '"':
                    j += 2 if line[j] == "\\" else 1
                out.append(line[i + 1:j])
                i = j
            i += 1
    return out


def decode(content: str) -> bytes | None:
    """The bytes of a literal's content by the lexer's escape table; None for a bad escape."""
    out, i = bytearray(), 0
    raw = content.encode()
    while i < len(raw):
        if raw[i] == 92:
            if i + 1 >= len(raw) or chr(raw[i + 1]) not in ESCAPES:
                return None
            out.append(ESCAPES[chr(raw[i + 1])])
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return bytes(out)


def align8(n: int) -> int:
    return (n + 7) // 8 * 8


def feed(compiler: pathlib.Path, source: bytes, layout: bool) -> subprocess.CompletedProcess:
    """The compiler's stdin protocol: [layout words] + [user_lo][src_len][std blob + program]."""
    std = capture_bundle(_REPO).std_prefix
    data = std + source
    head = struct.pack("<qqqq", LAYOUT_MAGIC, *BOUNDED_LAYOUT) if layout else b""
    return start_child([str(compiler)], head + struct.pack("<qq", len(std), len(data)) + data,
                          capture_output=True, timeout=600)


def run_measured(image: bytes, tmp: pathlib.Path, timeout: int = 120) -> tuple[int, bytes, int]:
    """(exit status, or minus the signal number; stdout; peak resident KiB) of one run. GNU time starts
    the image: the peak of a process forked from this one would include this process's own size."""
    exe = tmp / "prog"
    exe.write_bytes(image)
    exe.chmod(0o755)
    p = subprocess.Popen(core_limit_argv(["/usr/bin/time", "-f", "RSS_KIB=%M", str(exe)]), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        out, err = p.communicate()
        return -9, out, 0
    signal_line = re.search(rb"terminated by signal (\d+)", err)
    rss = re.search(rb"RSS_KIB=(\d+)", err)
    return (-int(signal_line.group(1)) if signal_line else p.returncode), out, int(rss.group(1)) if rss else -1


def check_layout(image: bytes, texts: list[str]) -> list[str]:
    """Every site of `texts` (any order) in the rodata segment: bytes, zero gap, then an 8-aligned
    24-byte handle {0x800000 + off, len, 0}; the segment ends with the last handle; the code holds a
    movabs of each handle address."""
    seg = [p for p in _phdrs(image) if p[0] == 1 and p[3] == 0x800000]
    if len(seg) != 1 or seg[0][1] != 4:
        return [f"no read-only PT_LOAD at 0x800000 ({len(seg)} found)"]
    data = image[seg[0][2]:seg[0][2] + seg[0][5]]
    todo, off, errors = [decode(t) for t in texts], 0, []
    while todo and not errors:
        for k, lit in enumerate(todo):
            hoff = align8(off + len(lit))
            if data[off:off + len(lit)] == lit and data[off + len(lit):hoff] == bytes(hoff - off - len(lit)) \
                    and data[hoff:hoff + 24] == struct.pack("<qqq", 0x800000 + off, len(lit), 0):
                if b"\x48\xb8" + struct.pack("<q", 0x800000 + hoff) not in image:
                    errors.append(f"no movabs of the handle address {0x800000 + hoff:#x}")
                todo.pop(k)
                off = hoff + 24
                break
        else:
            errors.append(f"offset {off}: no remaining literal has its bytes, zero gap and handle here; rodata[{off}:{off + 32}]={data[off:off + 32].hex()}")
    if not errors and off != len(data):
        errors.append(f"rodata is {len(data)} bytes, the sites end at {off}")
    return errors


def rows() -> list[dict]:
    """The programs of section B. exit/stdout/signal are computed here from the program text.
    DATA holds programs whose run does not change with the handle (the two-compiler identity run
    compares them with the frozen seed's images); LIMITS holds six programs that the change gives
    another run (a loop of 2^25 evaluations or more: SIGILL at the arena guard before on the default
    layout, the same run on the bounded layout; a store through the handle: no fault before, the handle
    was in the arena) and two control rows that it gives the
    same run (loop_under_2p25: 2^25 - 1 evaluations; store_bytes_faults: a store through the bytes)."""
    text = lambda name, where=DATA: (where / f"{name}.mind").read_text()
    lens = lambda src: [len(decode(t)) for t in literal_texts(src)]
    sink = lambda k: k * 3
    show = [b"addr+len\n", b"", b'tab\tquote"slash\\\n', "été\n".encode()]
    loop = lambda name, **want: {"id": name, "src": text(name, LIMITS), "rss": True, **want}
    return [
        {"id": "one_literal", "src": text("one_literal"), "exit": 5, "layout": True},
        {"id": "empty_literal", "src": text("empty_literal"), "exit": 0, "layout": True},
        {"id": "loop_and_second", "src": text("loop_and_second"), "exit": 10 * 3 + 2, "layout": True},
        {"id": "len_edges", "src": text("len_edges"), "exit": sum(lens(text("len_edges"))), "layout": True},
        {"id": "escapes", "src": text("escapes"), "exit": sum(decode(r'\n\t\r\0\\\'\"')) % 256, "layout": True},
        {"id": "nul_inside", "src": text("nul_inside"), "exit": 98 * 2 + 0 + 3, "layout": True},
        {"id": "same_text_two_fns", "src": text("same_text_two_fns"), "exit": 6, "layout": True},
        {"id": "many_literals_300", "src": text("many_literals_300"), "exit": sum(lens(text("many_literals_300"))) % 256, "layout": True},
        {"id": "through_two_calls", "src": text("through_two_calls"), "exit": 2 * sum(b"ABCD") % 256, "layout": True},
        {"id": "many_args", "src": text("many_args"), "exit": 1 + 1 * 2 + 2 * 3 + 2 * 4 + 3 * 5 + 3 * 6 + 4 * 7 + 4 * 8, "layout": True},
        {"id": "forwarded", "src": text("forwarded"), "exit": sink(4) + sink(4 + 1) + 5, "layout": True},
        {"id": "literal_args_loop", "src": text("literal_args_loop"), "exit": 4 * 25 + 1 + 2 + 3 + 4 - 10, "layout": True},
        {"id": "addr_write", "src": text("addr_write"), "exit": (9 + 0 + 17 + sum(b"ABCD")) % 256, "stdout": b"".join(show), "layout": True},
        {"id": "print_arg", "src": text("print_arg"), "exit": 0, "stdout": b"hi", "layout": True},
        {"id": "loop_print_text", "src": text("loop_print_text"), "exit": 3, "stdout": b"ab" * 1000, "layout": True},
        # A `const` String that no code reads lowers no literal. The string rows refuse it all the same,
        # so no image exists. The text is inline, not in DATA: the identity run takes a literal in the
        # text of a native_*/ program for a changed image.
        {"id": "unused_const", "src": 'const GREETING: String = "hey";\nfn main() -> i64 {\n    return 7;\n}\n',
         "refused": "lane.string.const"},
        loop("loop_take_50m", exit=50_000_000 * 9 % 251),
        loop("loop_k_45m", exit=45_000_000 % 256),
        loop("loop_k_50m", exit=50_000_000 % 256),
        loop("loop_under_2p25", exit=(2 ** 25 - 1) // 1000 % 100),
        loop("loop_over_2p25", exit=2 ** 25 // 1000 % 100),
        loop("loop_print_empty_50m", exit=7, stdout=b""),
        # S4: the bytes and the handle of a literal are read-only; a store faults.
        loop("store_bytes_faults", exit=-11),
        loop("store_handle_faults", exit=-11),
    ]


def stage1_compiler(given: str | None, tmp: pathlib.Path) -> pathlib.Path:
    """--compiler ELF, else one hop: the frozen seed on THIS tree's compiler sources, bounded layout."""
    if given:
        return pathlib.Path(given).resolve()
    out = tmp / "stage1.elf"
    head = struct.pack("<qqqq", LAYOUT_MAGIC, *BOUNDED_LAYOUT)
    out.write_bytes(_run_compiler(_REPO / FROZEN_SEED, self_mode_image(head + capture_bundle(_REPO, COMPILER_COMPONENTS).stdin_image, _REPO / FROZEN_SEED)))
    out.chmod(0o755)
    print(f"minted stage1 from the frozen seed: {out.stat().st_size} B")
    return out


def row_section(compiler: pathlib.Path, tmp: pathlib.Path) -> tuple[int, int]:
    ran = fail = 0
    for row in rows():
        for layout in (False, True):
            tag = f"{row['id']}[{'bounded' if layout else 'default'}]"
            ran += 1
            done = feed(compiler, row["src"].encode(), layout)
            if row.get("refused"):
                ok = done.returncode == 3 and done.stdout[:4] != b"\x7fELF" and f"({row['refused']})" in done.stderr.decode(errors="replace")
                print(f"  {'PASS' if ok else 'FAIL'}  B {tag}: rc={done.returncode} want a refusal naming {row['refused']}")
                fail += 0 if ok else 1
                continue
            if done.returncode != 0 or done.stdout[:4] != b"\x7fELF":
                print(f"  FAIL  B {tag}: compile rc={done.returncode} {done.stderr[-200:]!r}")
                fail += 1
                continue
            code, out, rss = run_measured(done.stdout, tmp)
            errors = []
            if code != row["exit"]:
                errors.append(f"exit {code}, expected {row['exit']}")
            if "stdout" in row and out != row["stdout"]:
                errors.append(f"stdout {out[:40]!r}..., expected {len(row['stdout'])} B")
            if row.get("rss") and rss > RSS_MAX_KIB:
                errors.append(f"peak resident {rss} KiB > {RSS_MAX_KIB}")
            if row.get("layout"):
                errors += check_layout(done.stdout, literal_texts(row["src"]))
            seen = f"exit={code} stdout={len(out)}B rss={rss}KiB image={len(done.stdout)}B"
            if errors:
                print(f"  FAIL  B {tag}: {seen} " + "; ".join(errors))
                fail += 1
            else:
                print(f"  PASS  B {tag}: {seen}")
    return ran, fail


def drive(driver: pathlib.Path, off0: int, contents: list[str]) -> tuple[list[dict], bytes, int]:
    """Run the driver: per site {poisoned, last_id, next_id, header, count, code}, the rodata bytes, off0."""
    send = struct.pack("<qq", off0, len(contents))
    for c in contents:
        quoted = ('"' + c + '"').encode()
        send += struct.pack("<q", len(quoted)) + quoted
    r = start_child([str(driver)], send, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"driver exit {r.returncode} for off0={off0} {contents!r}")
    sites, pos = [], 0
    for _ in contents:
        poisoned, last_id, next_id, header, code_len, count = struct.unpack_from("<6q", r.stdout, pos)
        sites.append({"poisoned": poisoned, "last_id": last_id, "next_id": next_id, "header": header,
                      "count": count, "code": r.stdout[pos + 48:pos + 48 + code_len]})
        pos += 48 + code_len
    first, end = struct.unpack_from("<2q", r.stdout, pos)
    return sites, r.stdout[pos + 16:pos + 16 + end - first], first


def check_drive(driver: pathlib.Path, off0: int, contents: list[str]) -> list[str]:
    """One driver run against the layout rule computed here: a site is accepted iff
    align8(off + len) + 24 <= window; then bytes, zero gap, handle {0x800000 + off, len, 0}, header
    hoff + 24 and one movabs of the handle address; else the site is poisoned and writes nothing."""
    sites, dump, first = drive(driver, off0, contents)
    errors, off, next_id = [], off0, 0
    for n, (content, site) in enumerate(zip(contents, sites)):
        lit = decode(content)
        hoff = align8(off + len(lit)) if lit is not None else 0
        before = 17 * next_id  # the code String is threaded: each accepted site adds 17 bytes
        if site["count"] != next_id + 1:  # the frame-size pass: one id per site, accepted or refused
            errors.append(f"site {n} {content!r} at {off}: nb_count_expr gave {site['count']}, nb_expr takes 1 id from {next_id}")
        if lit is None or hoff + 24 > CAP:
            if site["poisoned"] != 1 or site["last_id"] != -1 or site["header"] != off or len(site["code"]) != before:
                errors.append(f"site {n} {content!r} at {off}: must be refused with nothing written, got {site}")
            continue
        code = b"\x48\xb8" + struct.pack("<q", 0x800000 + hoff) + b"\x48\x89\x85"  # movabs rax, handle ; mov [rbp+disp32], rax
        mine = site["code"][before:]
        if (site["poisoned"], site["last_id"], site["next_id"], site["header"]) != (0, next_id, next_id + 1, hoff + 24) \
                or mine[:len(code)] != code or len(mine) != len(code) + 4:
            errors.append(f"site {n} {content!r} at {off}: wrong state or code, got {site}")
        expect_dump = lit + bytes(hoff - off - len(lit)) + struct.pack("<qqq", 0x800000 + off, len(lit), 0)
        if dump[off - off0:hoff + 24 - off0] != expect_dump:
            errors.append(f"site {n} {content!r} at {off}: rodata {dump[off - off0:hoff + 24 - off0].hex()} != {expect_dump.hex()}")
        off, next_id = hoff + 24, next_id + 1
    return errors


def driver_section(compiler: pathlib.Path, tmp: pathlib.Path) -> tuple[int, int]:
    driver = tmp / "handle_driver.elf"
    done = driver_compile("examples/mindc_mind/testdata/native_string_handle/handle_driver.mind", compiler, driver)
    if not driver.exists():
        print(f"  FAIL  C driver did not build: rc={done.returncode} {done.stderr[-200:]!r}")
        return 1, 1
    cases = [(0, ["hello"]), (0, ["abc", "abc"]), (3, ["abc"]), (5, ["a", "", "bcdefghi"]),
             (0, [r"\n\t\r\0\\\'\""]), (0, [r"\u41"]), (0, [r"\x41"]), (0, ["abc\\"]), (0, [r"a\qb", "ok"])]
    cases += [(off, ["x" * n]) for off in range(CAP - 64, CAP + 1) for n in range(18)]
    fail = 0
    for off0, contents in cases:
        errors = check_drive(driver, off0, contents)
        fail += bool(errors)
        for e in errors[:3]:
            print("  FAIL  C " + e)
    accepted = sum(align8(off + n) + 24 <= CAP for off in range(CAP - 64, CAP + 1) for n in range(18))
    print(f"  {'FAIL' if fail else 'PASS'}  C driver: {len(cases)} runs, {fail} failed; window edge sweep "
          f"{accepted} accepted / {18 * 65 - accepted} refused by the rule align8(off + len) + 24 <= {CAP}")
    return len(cases), fail


# ---- section D: the window filled by many literals -------------------------------------

FILL_CHAINS = (40, 50, 129)  # 65,000-byte literals: 50 faulted, 129 is the last chain the window holds
FILL_TAIL = " No artifact written.\n"


def fill_program(lits: list[str]) -> str:
    """`import std.string;`, a chain of fns `fn g<k>() -> i64 { return string_len("<lits[k]>") + g<k+1>(); }`
    (the last fn has no call) and a main that calls g0."""
    lines = ["import std.string;"]
    for k, lit in enumerate(lits):
        nxt = f" + g{k + 1}()" if k + 1 < len(lits) else ""
        lines.append(f'fn g{k}() -> i64 {{ return string_len("{lit}"){nxt}; }}')
    return "\n".join(lines + ["fn main() -> i64 { return g0(); }"]) + "\n"


def fill_section(compiler: pathlib.Path, tmp: pathlib.Path) -> tuple[int, int]:
    big, small = "A" * 65000, ["B" * (k % 7 + 1) for k in range(2048)]
    assert 128 * (align8(65000) + 24) + 2048 * (8 + 24) == CAP  # the edge case fills the window to the byte
    cases = [(f"chain_{n}x65000", [big] * n) for n in FILL_CHAINS] + [("edge_128_big_2048_small", [big] * 128 + small)]
    over = [big] * 128 + small + ["C"]
    last = f"g{len(over) - 1}"
    want = (f"error[native-lane]: the program is outside candidate native-lane-v1 (lane.string.literal_capacity) "
            f"in fn `{last}` at line {len(over) + 1}, column {len(f'fn {last}() -> i64 {{ return string_len(') + 1} "
            f"of the composed user image.{FILL_TAIL}")
    ran = fail = 0
    for layout in (False, True):
        for name, lits in cases:
            tag, ran = f"{name}[{'bounded' if layout else 'default'}]", ran + 1
            done = feed(compiler, fill_program(lits).encode(), layout)
            if done.returncode != 0 or done.stdout[:4] != b"\x7fELF":
                print(f"  FAIL  D {tag}: compile rc={done.returncode} {done.stderr[-200:]!r}")
                fail += 1
                continue
            code, _out, _rss = run_measured(done.stdout, tmp)
            errors = ([f"exit {code}, expected {sum(map(len, lits)) % 256}"] if code != sum(map(len, lits)) % 256 else [])
            errors += check_layout(done.stdout, lits)
            fail += bool(errors)
            print(f"  {'FAIL' if errors else 'PASS'}  D {tag}: exit={code} image={len(done.stdout)}B " + "; ".join(errors))
        tag, ran = f"edge_one_site_more[{'bounded' if layout else 'default'}]", ran + 1
        done = feed(compiler, fill_program(over).encode(), layout)
        ok = done.returncode == 3 and done.stdout == b"" and done.stderr.decode(errors="replace") == want
        fail += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  D {tag}: rc={done.returncode} {done.stderr[-160:]!r}")
    return ran, fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--compiler")
    args = ap.parse_args()
    ran = fail = 0
    section_a = so_section()
    if section_a:
        ran, fail = section_a
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        compiler = stage1_compiler(args.compiler or os.environ.get("MINDC_NATIVE_LANE_ELF"), tmp)
        for section in (row_section, driver_section, fill_section):
            r, f = section(compiler, tmp)
            ran, fail = ran + r, fail + f
    print(f"SDLC-GATE native_string_handle ran={ran} fail={fail}")
    return 1 if fail or ran == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
