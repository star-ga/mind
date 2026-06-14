#!/usr/bin/env python3
"""
CUTOVER COVERAGE MEASUREMENT (analysis only — does NOT edit main.mind).

For each of main.mind's top-level fns, feed a self-contained module (imports +
struct defs + the single fn) to BOTH:
  (1) the pure-MIND N-fn driver `selftest_mic3_module_nfn` (via ctypes on the .so), and
  (2) `target/release/mindc --emit-mic3`,
then byte-diff. Reports X/N byte-exact = coverage%.

Because a fn that REFERENCES other fns/types fails-closed in the nfn driver
(undefined callee -> resolve fails) AND the oracle emits an E2003 stub artifact,
the strict byte-diff conflates construct-unsupported with callee-unresolved.
So we ALSO statically bucket every fn body by the dominant blocking construct
(field-read, struct-lit, for, while, match, method-call) for the ranked table.

Methodology is fully reproducible: run from repo root with the .so + release
mindc built.
"""
import ctypes, os, pathlib, re, subprocess, sys, tempfile, collections

HERE = pathlib.Path(__file__).resolve().parent
SO = HERE / "libmindc_mind.so"
MINDC = HERE.parents[1] / "target" / "release" / "mindc"
MAIN = HERE / "main.mind"
Int64Ptr = ctypes.POINTER(ctypes.c_int64)

# NOTE: prepending struct defs makes the oracle emit a populated struct registry
# (84B vs 34B for tk_eof) while the nfn driver always emits EMPTY registries ->
# guaranteed mismatch. So we feed each fn BARE. Fns referencing custom types in
# their signature/body cannot byte-match (oracle errors / nfn fails) and are
# bucketed by construct instead. This is the honest in-isolation coverage.
PREAMBLE = ""

def read_i64(addr, off=0):
    return int(ctypes.cast(addr + off, Int64Ptr)[0])

def read_handle(handle):
    if handle == 0: return b""
    addr = read_i64(handle, 0); length = read_i64(handle, 8)
    if addr == 0 or length == 0: return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))

def split_fns(text):
    """Return list of (name, header_line_idx, src_text) for each top-level fn."""
    lines = text.split("\n")
    fns = []
    i = 0
    hdr = re.compile(r'^(pub )?fn ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    while i < len(lines):
        m = hdr.match(lines[i])
        if not m:
            i += 1; continue
        name = m.group(2)
        start = i
        depth = 0; seen = False; j = i
        while j < len(lines):
            code = lines[j].split("//", 1)[0]   # strip line comments
            for ch in code:
                if ch == '{': depth += 1; seen = True
                elif ch == '}': depth -= 1
            if seen and depth == 0:
                break
            j += 1
        body = "\n".join(lines[start:j+1])
        fns.append((name, start+1, body))
        i = j + 1
    return fns

def _nfn_emit_raw(fn, src):
    srcb = src.encode()
    srcc = ctypes.create_string_buffer(srcb, len(srcb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 4096)()
    ccell = (ctypes.c_int64 * 1)()
    es = fn(ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
            ctypes.cast(strbuf, ctypes.c_void_p).value,
            ctypes.cast(offs, ctypes.c_void_p).value,
            ctypes.cast(ccell, ctypes.c_void_p).value)
    return read_handle(read_i64(es, 0))

def nfn_emit(fn, src):
    """Fork-isolated: a segfault in the .so kills only the child. Child writes the
    emitted bytes to a pipe; -1 sentinel byte means CRASH."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            out = _nfn_emit_raw(fn, src)
            os.write(w, b"\x01" + out)
        except Exception:
            os.write(w, b"\x00")
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b: break
        chunks.append(b)
    os.close(r)
    _, status = os.waitpid(pid, 0)
    data = b"".join(chunks)
    if os.WIFSIGNALED(status) or not data or data[0] != 1:
        return None   # crash / fail-closed-with-no-data sentinel
    return data[1:]

def oracle(src):
    if not MINDC.exists(): return None, ""
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "c.mind"; op = pathlib.Path(td) / "c.mic3"
        sp.write_text(src)
        r = subprocess.run([str(MINDC), "--emit-mic3", str(op), str(sp)],
                           capture_output=True, text=True)
        err = r.stderr + r.stdout
        if not op.exists(): return None, err
        return op.read_bytes(), err

# Static construct classification on a fn body's text (post-signature).
def body_text(src):
    b = src.find("{")
    return src[b:] if b >= 0 else src

def classify(name, src):
    bt = body_text(src)
    # strip the signature/param list to avoid types like `s: EmitState` matching method.
    tags = set()
    # struct-lit construction: Ident{...}  (Capitalized name immediately followed by {)
    if re.search(r'\b[A-Z][A-Za-z0-9_]*\s*\{', bt):
        tags.add("struct-lit")
    # field-read: receiver.field where field is not a method call (no '(' after)
    if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*(?!\s*\()', bt):
        tags.add("field-read")
    # method-call: receiver.method(
    if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\s*\(', bt):
        tags.add("method-call")
    if re.search(r'\bfor\b', bt): tags.add("for")
    if re.search(r'\bwhile\b', bt): tags.add("while")
    if re.search(r'\bmatch\b', bt): tags.add("match")
    return tags

def main():
    lib = ctypes.CDLL(str(SO))
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5

    text = MAIN.read_text()
    fns = split_fns(text)
    n = len(fns)

    byte_exact = 0
    nfn_empty = 0
    nfn_crash = 0
    oracle_fail = 0
    exact_names = []
    fail_records = []   # (name, reason, tags)
    construct_block = collections.Counter()   # construct -> #failing fns where it's present
    construct_sole = collections.Counter()    # construct -> #failing fns where it's the ONLY blocking construct
    pure_fail_no_construct = []  # failing fns with NO flagged construct (callee/other)

    CONSTRUCTS = ["field-read", "for", "struct-lit", "match", "method-call", "while"]

    for fi, (name, ln, src) in enumerate(fns):
        if fi % 50 == 0:
            print(f"  ...{fi}/{n}", file=sys.stderr, flush=True)
        mod = PREAMBLE + src + "\n"
        got = nfn_emit(fn, mod)
        crashed = got is None
        if crashed:
            nfn_crash += 1
            got = b""
        orc, err = oracle(mod)
        if orc is None:
            oracle_fail += 1
        exact = (len(got) > 0 and orc is not None and got == orc)
        if exact:
            byte_exact += 1
            exact_names.append(name)
            continue
        # failing
        tags = classify(name, src)
        # method-call implies field-read regex also fires; treat method-call as distinct,
        # remove field-read if it's solely the method receiver. Heuristic: if method-call
        # present and field-read only from same pattern, keep both but rank separately.
        reason = "nfn-empty" if len(got) == 0 else ("mismatch" if orc is not None else "oracle-fail")
        if len(got) == 0: nfn_empty += 1
        present = [c for c in CONSTRUCTS if c in tags]
        for c in present:
            construct_block[c] += 1
        if len(present) == 1:
            construct_sole[present[0]] += 1
        if not present:
            pure_fail_no_construct.append(name)
        fail_records.append((name, reason, sorted(present)))

    print(f"=== CUTOVER COVERAGE — {n} top-level fns ===")
    print(f"BYTE-EXACT (nfn == --emit-mic3): {byte_exact}/{n} = {100.0*byte_exact/n:.1f}%")
    print(f"  nfn fail-closed (empty buf): {nfn_empty}")
    print(f"  nfn CRASHED (.so segfault, fork-isolated): {nfn_crash}")
    print(f"  oracle wrote artifact: {n - oracle_fail}/{n}")
    print()
    print("=== FAILING FNs bucketed by blocking construct (a fn may have several) ===")
    print(f"{'construct':<14}{'#fns where present':<22}{'#fns SOLE blocker':<20}{'real sites(grep)':<10}")
    grep_sites = {}
    for c, pat in [("field-read", None), ("for", r'\bfor '), ("struct-lit", None),
                   ("match", r'\bmatch '), ("method-call", None), ("while", r'\bwhile ')]:
        pass
    for c in CONSTRUCTS:
        print(f"{c:<14}{construct_block[c]:<22}{construct_sole[c]:<20}")
    print()
    print(f"failing fns with NO flagged construct (callee-unresolved / other): {len(pure_fail_no_construct)}")
    print()
    print("=== RANKING: single construct that, if folded, unblocks the most SOLE-blocked fns ===")
    for c, cnt in construct_sole.most_common():
        print(f"  {c}: {cnt} fns are blocked SOLELY by {c}")
    print()
    # Also: of byte-exact fns, what do they look like
    print(f"byte-exact fn names (first 40): {exact_names[:40]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
