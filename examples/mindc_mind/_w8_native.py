"""Shared harness for the W8 float smokes (self_host_native_f64_*_smoke.py).

Builds a stage1 compiler ELF from a main.mind by running the FROZEN pure-MIND seed
(testdata/selfhost_loop/stage1.elf) on [std blob + main.mind + selfhost_driver.mind]
-- no Rust, no MLIR in that step -- then feeds user programs to that stage1 directly
(stdin protocol [user_lo][src_len][std blob + user source], see selfhost_driver.mind)
and runs the emitted ELF. The MLIR oracle builds the same program with
`mindc build --backend mlir` and runs it.

Env:
  W8_MAIN_MIND   main.mind to build stage1 from (default: this tree's main.mind).
                 Used to show RED on a base-tree copy.
  MINDC_BIN      mindc for the MLIR oracle (default: shared resolver).
"""
import hashlib
import math
import re
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _stdlib_manifest import seed_blob  # noqa: E402
from _selfhost_sources import compiler_library, self_mode_image, start_child  # noqa: E402
from self_host_native_diag_smoke import HEAD as FRAME_HEAD, TAIL as FRAME_TAIL  # noqa: E402

FROZEN_SEED = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
DRIVER = HERE / "selfhost_driver.mind"
CACHE = pathlib.Path(tempfile.gettempdir()) / "w8_native_stage1_cache"


def main_mind_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("W8_MAIN_MIND", str(HERE / "main.mind")))


def std_blob() -> bytes:
    return seed_blob()


def _write_exec(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    os.chmod(path, 0o755)
    return path


def build_stage1() -> pathlib.Path:
    """stage1 = frozen seed run on the chosen main.mind. Cached by input sha256."""
    blob = std_blob()
    combined = blob + compiler_library(main_mind_path().read_bytes()) + b"\n" + DRIVER.read_bytes()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    key = hashlib.sha256(FROZEN_SEED.read_bytes() + img).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"stage1_{key}.elf"
    if out.exists():
        return out
    r = start_child([str(FROZEN_SEED)], self_mode_image(img, FROZEN_SEED), capture_output=True, timeout=1200)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise RuntimeError(f"frozen seed failed to build stage1: rc={r.returncode} "
                           f"stderr={r.stderr[-400:]!r}")
    tmp = CACHE / f".{key}.{os.getpid()}"
    _write_exec(tmp, r.stdout)
    os.replace(tmp, out)
    return out


def mint_stage1(out: str) -> int:
    """Rehearsal compiler (never a seed move): copy the stage1 that the frozen seed builds from this
    tree to `out` and print its sha256. Returns the exit status of `--mint-stage1`."""
    path = _write_exec(pathlib.Path(out), build_stage1().read_bytes())
    print(f"MINT stage1 {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


def native_compile(stage1: pathlib.Path, src: str, blob: bytes | None = None) -> tuple[int, bytes, str]:
    """Feed one user program to stage1. Returns (rc, elf_bytes, stderr_tail).

    `blob` overrides the std blob (default: the pinned seed blob)."""
    if blob is None:
        blob = std_blob()
    combined = blob + src.encode()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    r = start_child([str(stage1)], img, capture_output=True, timeout=600)
    elf = r.stdout if r.returncode == 0 and r.stdout[:4] == b"\x7fELF" else b""
    return r.returncode, elf, r.stderr.decode(errors="replace")[-400:]


def run_elf_bytes(elf: bytes, workdir: pathlib.Path, name: str) -> tuple[int, bytes]:
    p = _write_exec(workdir / name, elf)
    r = subprocess.run([str(p)], capture_output=True, timeout=120)
    return r.returncode, r.stdout


def mindc_bin() -> str:
    if os.environ.get("MINDC_BIN"):
        return os.environ["MINDC_BIN"]
    from _selfhost_so import resolve_mindc
    return resolve_mindc()


def mlir_build_run(src: str, workdir: pathlib.Path, name: str) -> tuple[str, int, bytes]:
    """MLIR oracle: ('ok', rc, stdout) or ('build-failed', rc, stderr)."""
    f = workdir / f"{name}.mind"
    f.write_text(src)
    out = workdir / f"{name}.mlir.bin"
    b = subprocess.run([mindc_bin(), "build", "--backend", "mlir", str(f), "--release", "--out", str(out)],
                       capture_output=True, timeout=900, cwd=str(workdir))
    if b.returncode != 0 or not out.exists():
        return "build-failed", b.returncode, (b.stdout + b.stderr)[-600:]
    r = subprocess.run([str(out)], capture_output=True, timeout=120)
    return "ok", r.returncode, r.stdout


def bits_of(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def u64(v: int) -> int:
    return v & 0xFFFFFFFFFFFFFFFF


def words(stdout: bytes) -> list[int]:
    n = len(stdout) // 8
    return list(struct.unpack(f"<{n}q", stdout[: n * 8]))


def i64_lit(v: int) -> str:
    """A MIND i64 expression for v (INT64_MIN cannot be a positive literal)."""
    if v == -(1 << 63):
        return "((0 - 9223372036854775807) - 1)"
    if v < 0:
        return f"(0 - {-v})"
    return str(v)


# f64 -> i64 cast oracles for the casts_edges / casts_exprs rows of native_lane_value_shapes_smoke.py.
def _bits(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & (2**64 - 1)))[0]


def _sat(x: float) -> int:
    if math.isnan(x):
        return 0
    if x >= 2.0**63:
        return 2**63 - 1
    if x <= -(2.0**63):
        return -(2**63)
    return int(x)


def cast_edges_oracle() -> list[int]:
    nan, nnan = _bits(9221120237041090560), _bits(-2251799813685248)
    pinf, ninf = _bits(9218868437227405312), _bits(-4503599627370496)
    big, two63, below = _bits(5085969321953787526), _bits(4890909195324358656), _bits(4890909195324358655)
    floats = (nan, nnan, pinf, ninf, big, -big, 9.7, -9.7, 0.5, -0.5, two63, -two63, below, -below,
              _bits(-(2**63)), _bits(1))
    vals = [_sat(v) for v in floats] + [_sat(nan), _sat(big), _sat(-big)]
    vals += [bits_of(float(k)) for k in (9007199254740993, 2**63 - 1, -(2**63), -1, 0,
                                         9007199254740995, -9007199254740993)]
    vals += [_sat(float(-123456789)), bits_of(3.5), _sat(14.0)]
    return vals


def cast_exprs_oracle() -> list[int]:
    a, b, pv, k = 3.75, 2.5, 4617315517961601024, 1000001
    acc = 0.0
    for i in range(10):
        acc = acc + float(i) * 0.5
    return [_sat(a + b), _sat(a * b), _sat(a - b * 4.0), _sat(a / 2.0), _sat(a + b + 0.25),
            _sat(7.75), _sat(-7.75), _sat(_bits(pv)), _sat(-a), _sat(float(k) / 3.0),
            bits_of(float(k * 3 + 1)), bits_of(float(pv)), _sat(a), _sat(acc), bits_of(acc),
            _sat(float(_sat(a))), bits_of(1.0), bits_of(0.0)]


def cast_words(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", v) for v in values)


# ---- extern key rows (nbt_extern_keys.mind; family libc_rows of native_lane_value_shapes_smoke.py) ----
# One program per case, built through the stdin protocol. A refusal row: rc 3, no image, and stderr holds
# the whole report line of the frame (key name, line and column of the declaration name in the program) where the
# site is the byte offset of the declaration name from the start of the composed image (user_lo + the offset in
# the program). Each refusal has a twin: the same program with the
# fault repaired, which builds and exits 42, so the refusal comes from the declaration. An accept row:
# repeated identical declarations build, and the calls (close and dup2 on descriptor -1) give -1. The
# rows xk_seed_once and xk_seed_dup reach a std function (tcp_close of std/net.mind), so the seed
# declaration of close joins the image beside the accepted user declarations. A user table that is refused
# (xk_user_conflict_seed_call) is refused before any seed declaration is read. The row xk_seed_region_unknown
# reaches proc_pid of std/process.mind, whose getpid has no row, and xk_seed_other_types_reached reaches the
# first std function that calls a std declaration of getenv with the shape the control fixture
# refused_getenv_mut_result pins as key 66 (found by the rule in seed_other_types_reached, not by a fixed std
# text): the site of each refusal lies in the std prefix, so the frame names the region and no line.
# Every refusal row asserts the whole line the frame prints (lane_report): key name from testdata/lane_keys.txt,
# line and column of a user site, `in the std region` for a std site.
# The type, reserved-name, order and sig2 rows follow the rule family: a type outside the registry's set is
# judged like any other width (66, 65 first when the count differs), an earlier defect keeps its word, and
# the calling convention step precedes the name step, the name step the count step, the count step the types.
XK_C = 'extern "C"'
XK_CLOSE, XK_DUP2 = "unsafe fn close(fd: i32) -> i32;", "unsafe fn dup2(a: i32, b: i32) -> i32;"
XK_UMASK, XK_LSEEK = "unsafe fn umask(mode: i32) -> i32;", "unsafe fn lseek(fd: i32, off: i64, whence: i32) -> i64;"
XK_CLOSEDIR, XK_KILL = "unsafe fn closedir(dir: *mut u8) -> i32;", "unsafe fn kill(pid: i32, sig: i32) -> i32;"
XK_BIND = "unsafe fn bind(sockfd: i32, addr: *const u8, addrlen: i32) -> i32;"
XK_NAME = {"mark": "nblxk001", "family": "libc_rows", "direct_feed": True}
XK_FAULTS = (  # (id, key, faulty declaration, its repair; "" removes it, a header repairs the block)
    ("unknown_never_called", 64, "unsafe fn frobnicate(a: i32) -> i32;", ""),
    ("unknown_accept", 64, "unsafe fn accept(fd: i32, addr: *mut u8, len: *mut u8) -> i32;", ""),
    ("arity_one_more", 65, "unsafe fn kill(pid: i32, sig: i32, extra: i32) -> i32;", XK_KILL),
    ("arity_one_less", 65, "unsafe fn kill(pid: i32) -> i32;", XK_KILL),
    ("arity_twelve", 65, "unsafe fn umask(" + ", ".join(f"p{i}: i32" for i in range(12)) + ") -> i32;", XK_UMASK),
    ("sig_i64_for_i32", 66, "unsafe fn umask(mode: i64) -> i32;", XK_UMASK),
    ("sig_i32_for_i64", 66, "unsafe fn lseek(fd: i32, off: i32, whence: i32) -> i64;", XK_LSEEK),
    ("sig_pointer_for_int", 66, "unsafe fn umask(mode: *const u8) -> i32;", XK_UMASK),
    ("sig_int_for_pointer", 66, "unsafe fn closedir(dir: i64) -> i32;", XK_CLOSEDIR),
    ("sig_const_for_mut", 66, "unsafe fn closedir(dir: *const u8) -> i32;", XK_CLOSEDIR),
    ("sig_result_width", 66, "unsafe fn umask(mode: i32) -> i64;", XK_UMASK),
    ("sig_missing_result", 66, "unsafe fn umask(mode: i32);", XK_UMASK),
    ("sig_result_for_void", 66, "unsafe fn freeaddrinfo(res: *mut u8) -> i32;", "unsafe fn freeaddrinfo(res: *mut u8);"),
    ("sig_u32_for_i32", 66, "unsafe fn umask(mode: u32) -> i32;", XK_UMASK),
    ("sig_f32_param", 66, "unsafe fn umask(mode: f32) -> i32;", XK_UMASK),
    ("sig_variadic", 66, "unsafe fn umask(mode: i32, ...) -> i32;", XK_UMASK),
    ("conflict_second", 66, "unsafe fn close(fd: i64) -> i32;", XK_CLOSE),
    ("conflict_result_only", 66, "unsafe fn close(fd: i32) -> i64;", XK_CLOSE),
    ("conflict_pointer_const", 66, "unsafe fn closedir(dir: *const u8) -> i32;", XK_CLOSEDIR),
    ("conflict_arity", 65, "unsafe fn close(fd: i32, extra: i32) -> i32;", XK_CLOSE),
    ("seed_form_bind", 66, "unsafe fn bind(sockfd: i32, addr: i64, addrlen: i32) -> i32;", XK_BIND),
)
XK_CALLCONV = (("win64", 'extern "C" callconv(.win64)'), ("aapcs", 'extern "C" callconv(.aapcs)'),
               ("vectorcall", 'extern "C" callconv(.vectorcall)'), ("stdcall_abi", 'extern "stdcall"'))
XK_ACCEPT = (  # (id, blocks): repeated declarations of one name with equal types
    ("dup_one_block", [(XK_C, [XK_CLOSE, XK_CLOSE, XK_DUP2])]),
    ("dup_two_blocks", [(XK_C, [XK_CLOSE, XK_DUP2]), (XK_C, [XK_CLOSE, XK_DUP2])]),
    ("dup_three_times", [(XK_C, [XK_CLOSE, XK_DUP2, XK_CLOSE]), (XK_C, [XK_CLOSE, XK_DUP2]), (XK_C, [XK_DUP2])]),
    ("dup_param_names", [(XK_C, [XK_CLOSE, XK_DUP2]), (XK_C, ["unsafe fn close(handle: i32) -> i32;", XK_DUP2.replace("a:", "x:")])]),
    ("dup_sysv_and_c", [(XK_C, [XK_CLOSE, XK_DUP2]), ('extern "C" callconv(.sysv)', [XK_CLOSE, XK_DUP2])]),
    ("dup_between_others", [(XK_C, [XK_CLOSE, XK_UMASK, XK_DUP2]), (XK_C, [XK_KILL, XK_CLOSE, XK_UMASK, XK_DUP2])]),
    # std/iouring.mind (a seed module) declares bind as (i32, *const u8, i32) -> i32; the user declaration equals the row
    ("seed_form_other_accepted", [(XK_C, [XK_CLOSE, XK_DUP2, XK_BIND]), (XK_C, [XK_BIND, XK_CLOSE])]),
)
XK_SEED_MAIN = ("fn main() -> i64 {\n    let minus_one: i64 = 0 - 1;\n    let h: i64 = __mind_alloc(8);\n"
                "    __mind_store_i64(h, minus_one);\n    tcp_close(h);\n"
                "    if close(minus_one as i32) as i64 != minus_one {\n        return 1;\n    }\n    return 42;\n}\n")
XK_PID_MAIN = "fn main() -> i64 {\n    let p: i64 = proc_pid();\n    if p == 0 {\n        return 1;\n    }\n    return 42;\n}\n"
XK_SEED_ACCEPT = (  # std/net.mind tcp_close calls close: the user declarations of close sit beside the seed ones
    ("seed_once", [(XK_C, [XK_CLOSE])]),
    ("seed_dup", [(XK_C, [XK_CLOSE]), (XK_C, [XK_CLOSE]), (XK_C, [XK_CLOSE])]),
)


def _xk_text(blocks: list) -> str:
    return "\n".join(f"{h} {{\n" + "".join(f"    {d}\n" for d in decls) + "}\n" for h, decls in blocks)


def _xk_prog(blocks: list) -> str:
    text = _xk_text(blocks)
    body = ("fn main() -> i64 {\n    let minus_one: i64 = 0 - 1;\n"
            "    if close(minus_one as i32) as i64 != minus_one {\n        return 1;\n    }\n")
    if any("fn dup2(" in x for _, decls in blocks for x in decls):
        body += "    if dup2(minus_one as i32, minus_one as i32) as i64 != minus_one {\n        return 2;\n    }\n"
    return text + "\n" + body + "    return 42;\n}\n"


def lane_key_names() -> dict[int, str]:
    """{key id: key name} from the committed key table (the one the frame check compares the compiler's own table with)."""
    rows = (pathlib.Path(__file__).resolve().parent / "testdata" / "lane_keys.txt").read_text().splitlines()
    return {int(a): b for a, b in (r.split(" ", 1) for r in rows)}


def lane_report(key: int, site: int, prog: str, user_lo: int, sym: str = "", fn: str = "") -> str:
    """The whole refusal line the frame prints for `key` at byte `site` of the composed image (the std prefix, then
    `prog`; the head and tail text are the frame check's own). A site inside the std prefix is named by the region;
    a user site by its line and column (1-based, in characters) of the program. `fn` is the fn that holds the site.
    An extern key (64 to 67) whose site is a function name (a name, trivia, `(`) also names that symbol: `sym`, or
    for a user site the name the program has there."""
    if 64 <= key <= 67 and not sym and site >= user_lo:
        named = re.match(r"[A-Za-z0-9_]+(?=(?:[ \t\r\n]|//[^\n]*(?:\n|$))*\()", prog.encode()[site - user_lo:].decode())
        sym = named.group(0) if named else ""
    symbol = f" Symbol `{sym}`." if 64 <= key <= 67 and sym else ""
    held = f" in fn `{fn}`" if fn else ""
    if site < user_lo:
        where = " in the std region."
    else:
        head = prog.encode()[:site - user_lo].decode()
        where = f" at line {head.count(chr(10)) + 1}, column {len(head) - head.rfind(chr(10))} of the composed user image."
    return FRAME_HEAD + lane_key_names()[key] + ")" + held + where + symbol + FRAME_TAIL


XK_OTHER_DECL = "testdata/native_libc_rows/refused_getenv_mut_result.mind"  # carries the declaration text, key 66 as a user declaration


def seed_other_types_row(std: bytes, user_lo: int) -> dict:
    """xk_seed_other_types_reached: a kept std function reaches a std declaration whose types the row of its name
    does not have. The declaration is chosen by a rule, never by a fixed std text: the text of the declaration that
    the control fixture refused_getenv_mut_result.mind pins as key 66 when it stands in the user region (the first
    line of that fixture with `unsafe fn`); the wrapper is the first `pub fn` of the std prefix with i64 parameters
    that calls the name. The refusal is the same rule at a std site, so the frame names the region only. When std
    has no such declaration or no such wrapper, the row carries the facts as `row_error`: the runner reports it as
    a FAIL of this one row (the row stays, the count holds) and every mode of the smoke still imports."""
    decl = next(x.strip() for x in (pathlib.Path(__file__).resolve().parent / XK_OTHER_DECL).read_text().splitlines()
                if "unsafe fn " in x)
    name = decl[decl.index("fn ") + 3:decl.index("(")]
    text = std.decode()

    def broken(message: str) -> dict:
        # no leg runs and no mark is asked: the runner prints this text as the row's FAIL
        return {**XK_NAME, "id": "xk_seed_other_types_reached", "fixture": "", "mark": "", "direct_feed": False,
                "row_error": message}
    if decl not in text:
        return broken(f"xk_seed_other_types_reached: no std extern declaration equals `{decl}` (the shape of {XK_OTHER_DECL}); "
                      f"a std edit removed or retyped it: pick the new first std declaration of `{name}` with other types, or "
                      f"change the declaration text of that fixture and its twin together")
    for chunk in text.split("\npub fn ")[1:]:
        head = chunk.split("{", 1)[0]
        body = chunk.split("\n}\n", 1)[0]
        wrapper, params = head.split("(", 1)
        params = [q for q in params.rsplit(")", 1)[0].split(",") if q.strip()]
        if (f" {name}(" in body or f"= {name}(" in body) and "unsafe fn" not in body and head.rstrip().endswith("-> i64") \
                and all(q.split(":")[1].strip() == "i64" for q in params):
            main = f"fn main() -> i64 {{\n    let p: i64 = {wrapper}({', '.join('0' for _ in params)});\n    return 42;\n}}\n"
            return {**XK_NAME, "id": "xk_seed_other_types_reached", "fixture": main,
                    "refusal": lane_report(66, 0, "", user_lo, sym=name)}
    return broken(f"xk_seed_other_types_reached: no std `pub fn` with i64 parameters calls `{name}`; the declaration "
                  f"`{decl}` is declared but not reached")


def extern_key_rows(bundle) -> list[dict]:
    """The rows described above; `bundle` is the captured source bundle: the std prefix of the stdin protocol
    ends at user_lo, and the site of a user declaration is user_lo plus its offset in the program."""
    user_lo, std = bundle.user_lo, bundle.std_prefix

    def site(prog: str, decl: str) -> int:
        return user_lo + prog.index(decl) + decl.index("fn ") + 3

    def refuse(rid: str, prog: str, key: int, decl: str) -> dict:
        return {**XK_NAME, "id": f"xk_{rid}", "fixture": prog, "refusal": lane_report(key, site(prog, decl), prog, user_lo)}

    def accept(rid: str, prog: str) -> dict:
        return {**XK_NAME, "id": f"xk_{rid}", "fixture": prog, "expect_exit": 42}
    rows = []
    for rid, key, bad, good in XK_FAULTS:
        pair = rid.startswith("conflict_")
        if pair:  # the second declaration of a name: the first equals the row, the second differs
            second = _xk_prog([(XK_C, [XK_CLOSE, XK_DUP2] + ([XK_CLOSEDIR] if "closedir" in bad else [])), (XK_C, [bad])])
            rows += [refuse(rid, second, key, bad), accept(rid + "_twin", second.replace(bad, good))]
            continue
        prog = _xk_prog([(XK_C, [XK_CLOSE, bad, XK_DUP2])])
        rows += [refuse(rid, prog, key, bad), accept(rid + "_twin", prog.replace(f"    {bad}\n", f"    {good}\n" if good else ""))]
    first = refuse("conflict_first_wrong", _xk_prog([(XK_C, ["unsafe fn close(fd: i64) -> i32;"]), (XK_C, [XK_CLOSE, XK_DUP2])]), 66, "unsafe fn close(fd: i64) -> i32;")
    rows += [first,
             refuse("conflict_third", _xk_prog([(XK_C, [XK_CLOSE, XK_DUP2]), (XK_C, [XK_CLOSE]), (XK_C, ["unsafe fn close(fd: u32) -> i32;"])]), 66, "unsafe fn close(fd: u32) -> i32;")]
    for rid, header in XK_CALLCONV:
        prog = _xk_prog([(XK_C, [XK_CLOSE, XK_DUP2]), (header, [XK_UMASK])])
        rows += [refuse("callconv_" + rid, prog, 67, XK_UMASK), accept("callconv_" + rid + "_twin", prog.replace(header, XK_C))]
    rows += [accept(rid, _xk_prog(blocks)) for rid, blocks in XK_ACCEPT]
    rows += [accept(rid, _xk_text(blocks) + "\n" + XK_SEED_MAIN) for rid, blocks in XK_SEED_ACCEPT]
    wrong = "unsafe fn close(fd: i64) -> i32;"
    seed_conflict = _xk_text([(XK_C, [XK_CLOSE]), (XK_C, [wrong])]) + "\n" + XK_SEED_MAIN
    rows += [refuse("user_conflict_seed_call", seed_conflict, 66, wrong), accept("user_conflict_seed_call_twin", seed_conflict.replace(wrong, XK_CLOSE))]
    # the seed declaration of getpid (std/process.mind) has no row: key 64 at its name inside the std prefix
    assert std.count(b"unsafe fn getpid(") == 1
    rows += [{**XK_NAME, "id": "xk_seed_region_unknown", "fixture": XK_PID_MAIN,
              "refusal": lane_report(64, std.index(b'unsafe fn getpid(') + len(b'unsafe fn '), "", user_lo, sym="getpid")}]
    # two defective declarations: the first in source order is named, whatever its key
    unknown, wide = "unsafe fn frobnicate(a: i32) -> i32;", "unsafe fn umask(mode: i64) -> i32;"
    for rid, key, pair in (("first_defect_unknown", 64, (unknown, wide)), ("first_defect_signature", 66, (wide, unknown))):
        rows.append(refuse(rid, _xk_prog([(XK_C, [XK_CLOSE, *pair])]), key, pair[0]))
    rows.append(seed_other_types_row(std, user_lo))
    return rows + extern_key_more_rows(refuse, accept) + extern_key_form_rows(user_lo, refuse, accept)


# ---- rows of the type, reserved-name, order and sig2 rules (same family, same stage1) ----
XK_FOO = "struct Foo {\n    a: i64,\n}\n\n"
XK_REPR_FOO = "#[repr(C)]\nstruct Foo {\n    a: i32,\n}\n\n"
XK_ALIAS = "type Fd = i32;\n\n"
XK_CUL = ("unsafe fn cuLaunchKernel(f: *mut u8, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, sm: u32, "
          "s: *mut u8, p: *mut *mut u8, e: *mut *mut u8) -> i32;")
XK_CUL_WRONG = (  # (id, parameter index, the wrong spelling of that parameter); indexes 6..10 are in the second signature word
    ("sig2_param7", 6, "bz: u32", "bz: i64"), ("sig2_param8", 7, "sm: u32", "sm: i64"), ("sig2_param9", 8, "s: *mut u8", "s: i64"),
    ("sig2_param10", 9, "p: *mut *mut u8", "p: *mut u8"), ("sig2_param11", 10, "e: *mut *mut u8", "e: *mut u8"),
)
XK_TYPES = (  # (id, definitions before the blocks, faulty declaration, repair, key): a type outside the registry's set
    ("type_struct_param", XK_FOO, "unsafe fn dup2(a: Foo, b: i32) -> i32;", XK_DUP2, 66),
    ("type_repr_c_param", XK_REPR_FOO, "unsafe fn dup2(a: Foo, b: i32) -> i32;", XK_DUP2, 66),
    ("type_struct_second_param", XK_FOO, "unsafe fn dup2(a: i32, b: Foo) -> i32;", XK_DUP2, 66),
    ("type_struct_result", XK_FOO, "unsafe fn umask(mode: i32) -> Foo;", XK_UMASK, 66),
    ("type_repr_c_result", XK_REPR_FOO, "unsafe fn umask(mode: i32) -> Foo;", XK_UMASK, 66),
    ("type_struct_pointer", XK_FOO, "unsafe fn closedir(dir: *mut Foo) -> i32;", XK_CLOSEDIR, 66),
    ("type_alias_param", XK_ALIAS, "unsafe fn umask(mode: Fd) -> i32;", XK_UMASK, 66),
    ("type_struct_arity", XK_FOO, "unsafe fn kill(pid: Foo) -> i32;", XK_KILL, 65),
    ("reserved_name", "", "unsafe fn __mind_zzz(a: i32) -> i32;", "", 64),
    ("reserved_name_known_shape", "", "unsafe fn __mind_close(fd: i32) -> i32;", "", 64),
)


def extern_key_more_rows(refuse, accept) -> list[dict]:
    """Refusal rows with accepted twins for the rules the first rows leave open (see the comment above)."""
    rows = []
    for rid, defs, bad, good, key in XK_TYPES:
        prog = defs + _xk_prog([(XK_C, [XK_CLOSE, bad])])
        repaired = prog.replace(f"    {bad}\n", f"    {good}\n" if good else "")
        rows += [refuse(rid, prog, key, bad), accept(rid + "_twin", repaired)]
    # a repeated name whose later declaration has a type outside the set: refused at the later one
    second = "unsafe fn close(fd: Foo) -> i32;"
    prog = XK_FOO + _xk_prog([(XK_C, [XK_CLOSE]), (XK_C, [second])])
    rows += [refuse("type_struct_second_close", prog, 66, second), accept("type_struct_second_close_twin", prog.replace(second, XK_CLOSE))]
    # two defects, the second one a type outside the set: the earlier defect keeps its word, whatever its key
    unknown, bad = "unsafe fn frobnicate(a: i32) -> i32;", "unsafe fn dup2(a: Foo, b: i32) -> i32;"
    for rid, key, pair in (("type_first_defect_unknown", 64, (unknown, bad)), ("type_first_defect_type", 66, (bad, unknown))):
        rows.append(refuse(rid, XK_FOO + _xk_prog([(XK_C, [XK_CLOSE, *pair])]), key, pair[0]))
    # sig2: parameters 7..11 sit in the second signature word of the row compare
    for rid, _, good, wrong in XK_CUL_WRONG:
        bad = XK_CUL.replace(good, wrong)
        prog = _xk_prog([(XK_C, [XK_CLOSE, bad])])
        rows += [refuse(rid, prog, 66, bad), accept(rid + "_twin", prog.replace(bad, XK_CUL))]
    twice = _xk_prog([(XK_C, [XK_CLOSE, XK_CUL]), (XK_C, [XK_CUL.replace("sm: u32", "sm: i64")])])
    rows += [refuse("sig2_repeat_wide", twice, 66, XK_CUL.replace("sm: u32", "sm: i64")),
             accept("sig2_repeat_equal", twice.replace(XK_CUL.replace("sm: u32", "sm: i64"), XK_CUL))]
    # close and lseek declared three times, both called on descriptor -1: the calls give -1 (the native table has no
    # row for write, read, unlink or getpid)
    main = ("fn main() -> i64 {\n    let minus_one: i64 = 0 - 1;\n"
            "    if close(minus_one as i32) as i64 != minus_one {\n        return 1;\n    }\n"
            "    if lseek(minus_one as i32, 0, 0 as i32) != minus_one {\n        return 2;\n    }\n    return 42;\n}\n")
    three = _xk_text([(XK_C, [XK_CLOSE, XK_LSEEK]), (XK_C, [XK_LSEEK, XK_CLOSE]), (XK_C, [XK_CLOSE, XK_LSEEK])]) + "\n" + main
    wide = XK_LSEEK.replace("off: i64", "off: i32")
    rows += [accept("dup_close_lseek_three", three),
             refuse("conflict_lseek_offset", three.replace(f"    {XK_LSEEK}\n    {XK_CLOSE}\n", f"    {wide}\n    {XK_CLOSE}\n"), 66, wide)]
    # the order of the steps: callconv, then name, then count, then types
    win = 'extern "C" callconv(.win64)'
    for rid, key, header, decl in (
            ("order_callconv_unknown", 67, win, "unsafe fn frobnicate(a: i32) -> i32;"),
            ("order_callconv_signature", 67, win, "unsafe fn umask(mode: i64) -> i32;"),
            ("order_callconv_arity", 67, win, "unsafe fn kill(pid: i32) -> i32;"),
            ("order_arity_signature", 65, XK_C, "unsafe fn kill(pid: i64) -> i32;"),
            ("order_arity_result", 65, XK_C, "unsafe fn kill(pid: i32) -> i64;")):
        rows.append(refuse(rid, _xk_prog([(XK_C, [XK_CLOSE]), (header, [decl])]), key, decl))
    return rows


def extern_key_form_rows(user_lo, refuse, accept) -> list[dict]:
    """Forms that are not names, empty blocks, and calls the host used to judge.
    A refusal names the compiler key at the declaration. A row that declares a
    function is direct only. An empty block has no function, so it also runs at
    the front door. `**const` and `extern "stdcall"` stay direct: the host parser
    stops before the compiler. An accepted twin repairs the form."""
    def door(row: dict) -> dict:
        row["front_door"] = True
        if "refusal" in row:
            row["stderr_required"] = row["refusal"].strip()
        return row

    def at(rid: str, prog: str, key: int, site: int) -> dict:
        return {**XK_NAME, "id": f"xk_{rid}", "fixture": prog, "refusal": lane_report(key, site, prog, user_lo)}

    def run(rid: str, prog: str, code: int) -> dict:
        row = accept(rid, prog)
        row["expect_exit"] = code
        return row

    rows = []
    forms = (
        ("form_array", "unsafe fn close(fd: [u8; 16]) -> i32;", XK_CLOSE, 66),
        ("form_array_i32", "unsafe fn close(fd: [i32; 2]) -> i32;", XK_CLOSE, 66),
        ("form_ref", "unsafe fn close(fd: &i32) -> i32;", XK_CLOSE, 66),
        ("form_ref_mut", "unsafe fn close(fd: &mut i32) -> i32;", XK_CLOSE, 66),
        ("form_generic", "unsafe fn close(fd: Vec<i32>) -> i32;", XK_CLOSE, 66),
        ("form_generic_two", "unsafe fn close(fd: Vec<i32, i64>) -> i32;", XK_CLOSE, 66),
        ("form_unit", "unsafe fn close(fd: i32) -> ();", XK_CLOSE, 66),
        ("form_slice", "unsafe fn close(fd: *const [u8]) -> i32;", XK_CLOSE, 66),
        ("form_slice_len", "unsafe fn close(fd: *const [u8; 4]) -> i32;", XK_CLOSE, 66),
        ("form_fnptr", 'unsafe fn close(fd: extern "C" fn(i32) -> i32) -> i32;', XK_CLOSE, 66),
    )
    for rid, bad, good, key in forms:
        prog = _xk_prog([(XK_C, [bad, XK_DUP2])])
        rows += [refuse(rid, prog, key, bad), accept(rid + "_twin", prog.replace(bad, good))]
    tup = "unsafe fn getuid() -> (i32, i32);"
    prog = _xk_prog([(XK_C, [XK_CLOSE, tup, XK_DUP2])])
    rows += [refuse("form_tuple", prog, 66, tup),
             accept("form_tuple_twin", prog.replace(tup, "unsafe fn getuid() -> i32;"))]
    star = "unsafe fn close(fd: **const u8) -> i32;"
    prog = _xk_prog([(XK_C, [star, XK_DUP2])])
    rows += [refuse("form_ptrptr", prog, 66, star), accept("form_ptrptr_twin", prog.replace(star, XK_CLOSE))]
    unknown, bad = "unsafe fn frobnicate(a: i32) -> i32;", "unsafe fn close(fd: [u8; 16]) -> i32;"
    prog = _xk_prog([(XK_C, [unknown, bad])])
    rows.append(refuse("form_mask_unknown", prog, 64, unknown))
    called = (
        ("called_f64_param", 66, "unsafe fn close(fd: f64) -> i32;",
         'extern "C" {\n    unsafe fn close(fd: f64) -> i32;\n}\nfn main() -> i64 {\n    close(1.5);\n    return 7;\n}\n',
         _xk_prog([(XK_C, [XK_CLOSE, XK_DUP2])]), 42),
        ("called_f64_result", 66, "unsafe fn getuid() -> f64;",
         'extern "C" {\n    unsafe fn getuid() -> f64;\n}\nfn main() -> i64 {\n    let y: f64 = getuid();\n    return 7;\n}\n',
         'extern "C" {\n    unsafe fn getuid() -> i32;\n}\nfn main() -> i64 {\n    let y: i64 = getuid() as i64;\n    if y < 0 {\n        return 1;\n    }\n    return 42;\n}\n',
         42),
        ("called_cbrt", 64, "unsafe fn cbrt(x: f64) -> f64;",
         'extern "C" {\n    unsafe fn cbrt(x: f64) -> f64;\n}\nfn main() -> i64 {\n    let y: f64 = cbrt(8.0);\n    return 7;\n}\n',
         "fn main() -> i64 {\n    return 7;\n}\n", 7),
        ("called_f32", 66, "unsafe fn umask(mode: f32) -> i32;",
         'extern "C" {\n    unsafe fn umask(mode: f32) -> i32;\n}\nfn main() -> i64 {\n    umask(1.5);\n    return 7;\n}\n',
         'extern "C" {\n    unsafe fn umask(mode: i32) -> i32;\n}\nfn main() -> i64 {\n    umask(0);\n    return 7;\n}\n',
         7),
    )
    for rid, key, decl, bad_prog, good_prog, code in called:
        rows += [refuse(rid, bad_prog, key, decl), run(rid + "_twin", good_prog, code)]
    empty = 'extern "C" {\n}\nfn main() -> i64 {\n    return 7;\n}\n'
    unused = 'extern "C" {\n    unsafe fn close(fd: i32) -> i32;\n}\nfn main() -> i64 {\n    return 7;\n}\n'
    rows += [door(run("empty_block", empty, 7)), run("uncalled_close", unused, 7)]
    win = 'extern "C" callconv(.win64) {\n}\nfn main() -> i64 {\n    return 7;\n}\n'
    rows.append(door(at("empty_win64", win, 67, user_lo + win.index(".win64") + 1)))
    stdcall = 'extern "stdcall" {\n}\nfn main() -> i64 {\n    return 7;\n}\n'
    rows.append(at("empty_stdcall", stdcall, 67, user_lo + stdcall.index('"stdcall"')))
    head = 'extern "C" callconv(.win64) {\n}\n'
    beside = head + _xk_prog([(XK_C, [XK_CLOSE, XK_DUP2])])
    rows += [at("empty_win64_beside", beside, 67, user_lo + beside.index(".win64") + 1),
             run("empty_win64_beside_twin", beside.replace(head, 'extern "C" {\n}\n', 1), 42)]
    # a call of a table extern with another argument count or an argument of another class: the key at the call
    for rid, key, args in (("call_arity_none", 65, ""), ("call_arity_two", 65, "minus_one as i32, minus_one as i32"),
                           ("call_f64_arg", 66, "1.5")):
        prog = (_xk_text([(XK_C, [XK_CLOSE])]) + "\nfn main() -> i64 {\n    let minus_one: i64 = 0 - 1;\n"
                f"    close({args});\n    return 42;\n}}\n")
        site = user_lo + prog.index("    close(") + 4
        rows += [{**XK_NAME, "id": f"xk_{rid}", "fixture": prog, "refusal": lane_report(key, site, prog, user_lo, fn="main")},
                 run(rid + "_twin", prog.replace(f"close({args})", "close(minus_one as i32)"), 42)]
    return rows + extern_symbol_rows(user_lo) + extern_grammar_rows(refuse, accept)


# ---- every type form of the parser's type rule in an extern signature gets a key: the parser's path `a.b.c` alone,
# with `<..>` or `[N]` after it and inside each composite form, `Name[N]`, the slice and tensor forms, `diff tensor`,
# a variadic or void callback and the scalars no row has. The `libc::pid_t` spelling (the parser refuses it) gets
# the key too. A lone `:`, a third `:`, a path that ends in `::` and two names in a row are no type: generic refusal ----
XK_GRAMMAR_PARAM = (
    ("path", "libc::c_int"), ("path_std", "std::ffi::c_int"), ("path_deep", "a::b::c::d"), ("path_lead", "::libc::c_int"),
    ("path_ptr_const", "*const libc::c_char"), ("path_ptr_mut", "*mut std::ffi::c_void"),
    ("path_ptr_ptr", "*mut *const libc::c_char"), ("path_ref", "&libc::c_int"), ("path_ref_mut", "&mut libc::c_int"),
    ("path_array", "[libc::c_int; 4]"), ("path_slice", "&[libc::c_char]"), ("path_generic", "libc::Foo<i32>"),
    ("path_in_generic", "Vec<libc::Foo<i32>>"), ("path_tuple", "(libc::c_int, i32)"),
    ("path_fnptr_param", 'extern "C" fn(libc::c_int) -> i32'), ("path_fnptr_result", 'extern "C" fn(i32) -> libc::c_int'),
    ("slice_ref", "&[u8]"), ("slice_ref_mut", "&mut [u8]"), ("slice_bare", "[u8]"), ("tensor", "tensor<f32[3]>"),
    ("tensor_plain", "tensor<f32>"), ("tensor_angle", "Tensor<f32, [3]>"), ("tensor_bracket", "Tensor[f32,(3)]"),
    ("tensor_diff", "diff tensor<f32[3]>"), ("tensor_sparse", "tensor<sparse[csr], q16_16>"),
    ("fnptr_variadic", 'extern "C" fn(i32, ...) -> i32'), ("fnptr_void", 'extern "C" fn(i32)'),
    ("scalar_bool", "bool"), ("scalar_f64", "f64"), ("scalar_u8", "u8"), ("scalar_u64", "u64"), ("scalar_i8", "i8"),
    ("scalar_i16", "i16"), ("scalar_u16", "u16"), ("scalar_usize", "usize"), ("scalar_isize", "isize"),
    ("name_len", "bytes[32]"), ("dot_path", "libc.c_int"), ("dot_path_deep", "a.b.c"), ("dot_generic", "libc.Foo<i32>"),
    ("dot_name_len", "libc.bytes[32]"), ("dot_ptr_const", "*const libc.c_char"), ("dot_ptr_mut", "*mut std.ffi.c_void"),
    ("dot_ref", "&libc.c_int"), ("dot_array", "[libc.c_int; 4]"), ("dot_slice", "&[libc.c_char]"),
    ("dot_in_generic", "Vec<libc.Foo<i32>>"), ("dot_tuple", "(libc.c_int, i32)"),
    ("dot_fnptr_param", 'extern "C" fn(libc.c_int) -> i32'), ("dot_fnptr_result", 'extern "C" fn(i32) -> libc.c_int'),
)
XK_GRAMMAR_RESULT = (("path", "libc::pid_t"), ("path_std", "std::ffi::c_int"), ("path_ptr", "*mut libc::c_void"),
                     ("path_generic", "libc::Foo<i32>"), ("tensor_diff", "diff tensor<f32[3]>"),
                     ("name_len", "bytes[32]"), ("dot_path", "libc.pid_t"), ("dot_ptr", "*mut libc.c_void"),
                     ("dot_generic", "libc.Foo<i32>"), ("dot_name_len", "libc.bytes[32]"))
XK_NOT_A_TYPE = (("colon_single", "libc:c_int"), ("colon_triple", "libc:::c_int"), ("path_trailing", "libc::"),
                 ("path_lead_trailing", "::"), ("name_name", "Foo Bar"), ("diff_name_name", "Foo diff"))


def extern_grammar_rows(refuse, accept) -> list[dict]:
    """Key 66 for each form as the parameter or the result of close; the name and count steps keep their order (64 for
    a name with no row, 65 for a count that differs) when a path type stands in the declaration. One twin repairs every
    form row: it is the same program with close declared as its row says."""
    def form(decl: str) -> str:
        return _xk_prog([(XK_C, [decl, XK_DUP2])])
    rows = [refuse(f"grammar_param_{rid}", form(f"unsafe fn close(fd: {ty}) -> i32;"), 66, f"unsafe fn close(fd: {ty}) -> i32;")
            for rid, ty in XK_GRAMMAR_PARAM]
    rows += [refuse(f"grammar_result_{rid}", form(f"unsafe fn close(fd: i32) -> {ty};"), 66, f"unsafe fn close(fd: i32) -> {ty};")
             for rid, ty in XK_GRAMMAR_RESULT]
    rows.append(accept("grammar_twin", form(XK_CLOSE)))
    rows += [{**XK_NAME, "id": f"xk_grammar_not_a_type_{rid}", "fixture": form(f"unsafe fn close(fd: {ty}) -> i32;"),
              "refusal": "error[backend-native]: unsupported construct; use --backend=mlir"} for rid, ty in XK_NOT_A_TYPE]
    for rid, key, bad, good in (("path_arity", 65, "unsafe fn kill(pid: libc::pid_t) -> i32;", XK_KILL),
                                ("path_unknown", 64, "unsafe fn getpid() -> libc::pid_t;", ""),
                                ("path_getuid", 66, "unsafe fn getuid() -> libc::pid_t;", ""),
                                ("dot_path_arity", 65, "unsafe fn kill(pid: libc.pid_t) -> i32;", XK_KILL),
                                ("dot_path_unknown", 64, "unsafe fn getpid() -> libc.pid_t;", ""),
                                ("dot_path_getuid", 66, "unsafe fn getuid() -> libc.pid_t;", ""),
                                ("dot_path_second", 66, "unsafe fn dup2(a: i32, b: libc.c_int) -> i32;", XK_DUP2)):
        prog = _xk_prog([(XK_C, [XK_CLOSE, bad, XK_DUP2])])
        rows += [refuse(rid, prog, key, bad), accept(rid + "_twin", prog.replace(f"    {bad}\n", f"    {good}\n" if good else ""))]
    return rows


def extern_symbol_rows(user_lo) -> list[dict]:
    """The symbol of an extern key is the name before `(` whatever trivia stands between them (spaces, a line feed,
    a line comment) and whatever its length. Each row states its symbol by hand, not by the text rule of
    lane_report. (The front door of this smoke is the bounded profile, which refuses every extern: the host
    pass-through of extern results is tested in tests/native_lane_extern_parity.rs.)"""
    def call(text: str) -> str:
        return (_xk_text([(XK_C, [XK_CLOSE])]) + "\nfn main() -> i64 {\n    let minus_one: i64 = 0 - 1;\n"
                f"    {text};\n    return 42;\n}}\n")

    def decl(text: str) -> str:
        return f'extern "C" {{\n    {text}\n}}\nfn main() -> i64 {{\n    return 7;\n}}\n'

    long = "q" * 130
    rows = []
    for rid, key, prog, at, sym, fn in (
            ("sym_call_space", 66, call("close (1.5)"), "close (", "close", "main"),
            ("sym_call_line_feed", 66, call("close\n        (1.5)"), "close\n", "close", "main"),
            ("sym_call_comment", 66, call("close // the call\n        (1.5)"), "close //", "close", "main"),
            ("sym_decl_space", 64, decl("unsafe fn prctl (option: i32) -> i32;"), "prctl (", "prctl", ""),
            ("sym_decl_long", 64, decl(f"unsafe fn {long}(a: i32) -> i32;"), long + "(", long, "")):
        rows.append({**XK_NAME, "id": f"xk_{rid}", "fixture": prog,
                     "refusal": lane_report(key, user_lo + prog.index(at), prog, user_lo, sym=sym, fn=fn)})
    return rows
