"""Self-tests of the identity partition rules (`--identity-vs`): the merged partition and the allocator
partition. The rules and their helpers stay in _selfhost_identity.py; these plants prove that each guard
of a rule catches its break."""
import ast
import collections
import struct


def _prologue_elf() -> bytes:
    """A synthetic image whose entry starts with the allocator runtime prologue (has_runtime_prologue)."""
    prologue = bytearray(0x200)
    prologue[:4] = b"\x7fELF"
    struct.pack_into("<Q", prologue, 0x18, 0x400078)           # e_entry
    struct.pack_into("<Q", prologue, 0x20, 0x40)               # e_phoff
    struct.pack_into("<H", prologue, 0x38, 1)                  # e_phnum
    struct.pack_into("<IIQQQQ", prologue, 0x40, 1, 5, 0, 0x400000, 0, 0x200)
    prologue[0x78:0x82] = b"\x49\xb9" + struct.pack("<Q", 0x100_0000 + (1 << 30) - 4096)
    return bytes(prologue)


def _alloc_partition_selftest() -> bool:
    """The partition rule must catch a planted stray difference and a prologue on an identical image."""
    from _selfhost_identity import _alloc_partition_compare
    elf = b"\x7fELF" + bytes(0x100)
    prologue = _prologue_elf()
    ok = lambda b: (0, b, b"")
    stray = [("t/stray", ok(elf), ok(elf + b"\x01"))]                     # a plain image that differs
    allowed = [("t/allowed", ok(elf), ok(bytes(prologue)))]                 # a prologue image
    same_prologue = [("t/same", ok(bytes(prologue)), ok(bytes(prologue)))]  # identical but explicit-layout
    fake = lambda image: (0, b"")  # the synthetic images are never executed
    return (_alloc_partition_compare(stray, set(), 1, 0, True, fake)[0] == 1
            and _alloc_partition_compare(same_prologue, set(), 1, 0, True, fake)[0] == 1
            and _alloc_partition_compare(allowed, set(), 1, 0, True, fake)[0] == 0)


def _note_elf(note: bytes, mark: int) -> bytes:
    """Synthetic ELF whose only program header is a PT_NOTE. `mark` differs the image, not the note."""
    blob = bytearray(0x80 + len(note))
    blob[:4] = b"\x7fELF"
    struct.pack_into("<Q", blob, 0x20, 0x40)
    struct.pack_into("<H", blob, 0x38, 1)
    struct.pack_into("<IIQQQQ", blob, 0x40, 4, 4, 0x78, 0, 0, len(note))
    blob[0x78:0x78 + len(note)] = note
    blob[-1] = mark
    return bytes(blob)


def _row_shape_ok() -> bool:
    """expect_exit rows are admitted, refusal rows are not, and the identity row is the unknown-symbol site."""
    from _selfhost_identity import _REPO, _add_keys
    from _w8_native import _libc_admitted
    mind = _REPO / "examples/mindc_mind"
    table = ast.literal_eval((mind / "testdata/native_libc_rows/ROWS.txt").read_text())
    libc, _admit = _libc_admitted(table)
    refused = [f"examples/mindc_mind/{r['fixture']}" for r in table if r.get("refusal") and not r.get("front_door")]
    admitted = [f"examples/mindc_mind/{r['fixture']}" for r in table if "expect_exit" in r and not r.get("front_door")]
    if not refused or any(n in libc for n in refused) or any(n not in libc for n in admitted):
        return False
    name = "examples/mindc_mind/testdata/native_libc_io/generated_source.mind"
    keys: dict[str, set] = collections.defaultdict(set)
    pos: dict[str, dict] = collections.defaultdict(dict)
    _add_keys(keys, pos, name, (mind / "testdata/native_libc_io/identity_row.txt").read_text())
    return "lane.extern.unknown_symbol" in keys[name] and (1, 17) in pos[name].get("lane.extern.unknown_symbol", ())


def _merged_selftest() -> bool:
    """A clean set passes, and removing any class guard makes one planted case fail."""
    from _selfhost_identity import _merged_partition
    elf = lambda tag: b"\x7fELF" + bytes(0x100) + tag
    pro, ok = _prologue_elf(), lambda b: (0, b, b"")
    pro_eq, pro_b = bytearray(pro), bytearray(pro)
    pro_eq[0x100], pro_b[0x101] = 9, 3
    pro_eq, pro_b = bytes(pro_eq), bytes(pro_b)
    note_a, note_b = _note_elf(b"AAAA", 1), _note_elf(b"BBBB", 2)
    generic = (3, b"", b"error[backend-native]: unsupported construct; use --backend=mlir\n")
    code = (3, b"", b"error[E2039]: extern block: this calling convention has no ABI\n")
    lane = (3, b"", b"error[native-lane]: the program is outside candidate native-lane-v1 (lane.extern.arity) at line 2\n")
    other = (3, b"", b"error[E0001]: no\n")
    runs = {elf(b"a"): (0, b"x"), elf(b"b"): (0, b"x"), elf(b"c"): (0, b"y"), elf(b"d"): (-11, b""), pro: (33, b""),
            elf(b"e"): (33, b""), elf(b"L"): (33, b"x"), pro_eq: (0, b"x"), pro_b: (33, b""),
            note_a: (0, b"q"), note_b: (0, b"q"), elf(b"T"): None, elf(b"U"): None, elf(b"r1"): (1, b"\x01" + bytes(7)),
            elf(b"c0"): (0, bytes(8)), elf(b"c2"): (2, bytes(8))}
    msg = b"error[native-lane]: the program is outside candidate native-lane-v1 ("
    keyed = lambda key, line: (3, b"", msg + key + b") in fn `f` at line " + line + b", column 1 of the composed user image.\n")
    at = lambda key, line, col: (3, b"", msg + key + b") at line " + line + b", column " + col + b" of the composed user image.\n")
    sres, ext = "fn f() -> String {\n    1\n}\n", keyed(b"lane.extern.type", b"1")
    std_site = lambda fn: (3, b"", msg + b"lane.string.result) in fn `" + fn + b"` in the std region. No artifact written.\n")
    std_g = "pub fn g(s: String) -> String {\n    1\n}\n"
    std_sha = "pub fn sha(n: i64) -> i64 {\n    let z: i64 = __mind_free(n);\n    return z;\n}\n"

    def fake(image, args=(), env=None):
        if args:
            return (len(args), b"")
        if env is not None:
            return (int(env.get("EXIT", "0")), b"")
        return runs[image] if image in runs else (0, b"")

    lit, plain = 'fn main() { print("a"); }\n', "fn main() { 1 }\n"
    free_loop = "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 2 {\n        let f: i64 = __mind_free(i);\n        i = i + 1;\n    }\n    return i;\n}\n"
    directed = "// mind: native_alloc = bump\n" + free_loop
    for_bound = "fn bound(n: i64) -> i64 {\n    let z: i64 = __mind_free(n);\n    return n;\n}\nfn main() -> i64 {\n    let mut a: i64 = 0;\n    for i in 0..bound(2) {\n        a = a + 1;\n    }\n    return a;\n}\n"
    sha_loop = "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 1 {\n        let r: i64 = sha(i);\n        i = i + 1;\n    }\n    return i;\n}\n"
    lit_loop = 'fn main() -> i64 {\n    while i < 2 {\n        let f: i64 = __mind_free(i);\n    }\n    print("z");\n    return 0;\n}\n'
    cond_free = "fn main() -> i64 {\n    while __mind_free(i) < 2 {\n        i = i + 1;\n    }\n    return i;\n}\n"
    via_g = "fn h() {\n    g()\n}\nfn main() {\n    h()\n}\n"
    rows = lambda keys=None, libc=(), alloc=None, admit=None, pos=None, alloc_class=None, std="": {"libc": frozenset(libc), "keys": keys or {}, "alloc": alloc or {}, "admit": admit or {}, "pos": pos or {}, "alloc_class": alloc_class or {}, "std": std}
    arity = {"named": frozenset({"lane.extern.arity"})}
    exf = rows({"ex": frozenset({"lane.extern.unknown_symbol"})}, pos={"ex": {"lane.extern.unknown_symbol": frozenset({(1, 17)})}})
    admit_ok, kb = rows(libc={"adm"}, admit={"adm": [(42, {"EXIT": "42"}, (), None)]}), "examples/mindc_mind/testdata/native_avx2/k3_bank.mind"
    k87, ref1, cand0 = "fn main() { __mind_group_bank_i64(0, 0, 0, 0, 35) }\n", ok(elf(b"r1")), ok(elf(b"c0"))

    def v(name, text, ref, cand, facts):
        return _merged_partition({name: text}, [(name, ref, cand)], facts, 1, 0, True, fake)
    clean = [("lit", lit, ok(elf(b"a")), ok(elf(b"b"))), ("same", plain, ok(elf(b"a")), ok(elf(b"a"))),
             ("named", plain, generic, lane), ("kept", plain, lane, lane)]
    pairs = [(n, r, c) for n, _, r, c in clean]
    good = _merged_partition({n: t for n, t, _, _ in clean}, pairs, rows(arity), 1, 0, True, fake) == 0
    died = lambda alloc, text=free_loop: v("al", text, ok(elf(b"d")), ok(pro), rows(alloc=alloc))
    cases = [
        ("lit", lit, ok(elf(b"a")), ok(elf(b"c")), rows(), 1), ("stray", plain, ok(elf(b"a")), ok(elf(b"b")), rows(), 1),
        ("lit", lit, ok(elf(b"a")), ok(elf(b"a")), rows(), 1), ("nt", lit, ok(note_a), ok(note_b), rows(), 1),
        ("to", lit, ok(elf(b"T")), ok(elf(b"U")), rows(), 1), ("ip", plain, ok(pro), ok(pro), rows(), 1),
        ("lp", lit, ok(elf(b"a")), ok(pro_eq), rows(), 1), ("rp", free_loop, ok(pro), ok(pro_b), rows(), 1),
        ("adm", plain, generic, ok(elf(b"a")), rows(), 1), ("adm", plain, generic, ok(elf(b"a")), admit_ok, 2),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(7, {"EXIT": "42"}, (), None)]}), 1),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(1, {}, ("8",), b""), (0, {}, (), None)]}), 2),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(1, {}, ("8",), b"x")]}), 1),
        (kb, k87, ref1, cand0, rows(), 0), (kb[:-5] + "_x.mind", k87, ref1, cand0, rows(), 1),
        (kb, k87, ok(elf(b"e")), cand0, rows(), 1), (kb, k87, ref1, ok(elf(b"c2")), rows(), 1),
        (kb, k87, ref1, ok(elf(b"c")), rows(), 1), (kb, k87, cand0, ref1, rows(), 1),
        ("adm", plain, other, ok(elf(b"a")), admit_ok, 1), ("named", plain, generic, lane, rows(), 1),
        ("named", plain, code, lane, rows(arity), 2), ("named", plain, ok(elf(b"a")), lane, rows(), 1),
        ("named", plain, ok(elf(b"a")), lane, rows(arity), 2),
        ("ex", plain, generic, at(b"lane.extern.unknown_symbol", b"1", b"17"), exf, 2),
        ("ex", plain, generic, at(b"lane.extern.callconv", b"1", b"17"), exf, 1),
        ("ex", plain, generic, at(b"lane.extern.unknown_symbol", b"9", b"1"), exf, 1),
        ("ex", plain, ok(elf(b"a")), at(b"lane.extern.callconv", b"1", b"1"), exf, 1),
        ("lv", free_loop, ok(elf(b"L")), ok(pro), rows(alloc={"lv": (33, ())}), 1),
        ("al", lit, ok(elf(b"d")), ok(elf(b"e")), rows(alloc={"al": (33, ())}), 0),
        ("s", sres, generic, keyed(b"lane.string.result", b"1"), rows(), 2),
        ("s", sres, generic, keyed(b"lane.string.result", b"2"), rows(), 1),
        ("s", sres, generic, keyed(b"lane.string.binding", b"1"), rows(), 1),
        ("s", plain, generic, std_site(b"g"), {**rows(), "std": std_g}, 1),
        ("s", "fn main() {\n    g()\n}\n", generic, std_site(b"g"), {**rows(), "std": std_g}, 2),
        ("s", "fn other() {\n    g()\n}\nfn main() { 1 }\n", generic, std_site(b"g"), {**rows(), "std": std_g}, 1),
        ("s", plain, generic, std_site(b"g"), {**rows({"s": frozenset({"lane.string.result"})}), "std": std_g}, 2),
        ("s", plain, generic, std_site(b"h"), {**rows(), "std": std_g}, 1),
        ("corpus/w3_extern_x", plain, generic, ext, rows(), 1), ("w3_extern_x", plain, generic, ext, rows(), 1),
        ("nf", plain, ok(elf(b"a")), ok(pro_eq), rows(), 1),
        ("bd", directed, ok(elf(b"d")), ok(pro), rows(alloc={"bd": (33, ())}), 1),
        ("bp", free_loop, ok(elf(b"d")), ok(pro), rows(alloc={"bp": (33, ())}, alloc_class={"bp": "BUMP"}), 1),
        ("fb", for_bound, ok(elf(b"d")), ok(pro), rows(alloc={"fb": (33, ())}), 1),
        ("rk", plain, ok(elf(b"d")), ok(pro), rows(alloc={"rk": (33, ())}, alloc_class={"rk": "RECLAIM"}), 0),
        ("sh", sha_loop, ok(elf(b"d")), ok(pro), rows(alloc={"sh": (33, ())}, std=std_sha), 0),
        ("lf", lit_loop, ok(elf(b"a")), ok(pro_eq), rows(), 1),
        ("wc", cond_free, ok(elf(b"d")), ok(pro), rows(alloc={"wc": (33, ())}), 0),
        ("s", via_g, generic, std_site(b"g"), {**rows(), "std": std_g}, 2),
    ]
    return (good and _row_shape_ok() and all(v(*c[:5]) == c[5] for c in cases)
            and died({"al": (33, ())}) == 0 and died({"al": (2, ("a", "b"))}) == 0
            and died({"al": (34, ())}) == 1 and died({}) == 1)
