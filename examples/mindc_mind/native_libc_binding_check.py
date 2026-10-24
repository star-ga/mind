#!/usr/bin/env python3
"""Source-bound libc declaration diagnostics; no call/production authority."""

import argparse, csv, hashlib, json, os, re, resource, shutil, struct, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def typ(t):
    out = []
    while t.startswith("*"):
        x, t = t.split(" ", 1)
        out.append(32 if x == "*const" else 33)
    out.append({"void": 0, "i8": 1, "u8": 2, "i32": 5, "u32": 6, "i64": 7, "u64": 8}[t])
    return bytes(out)


def key(t):
    b = typ(t)
    return (len(b) << 32) + int.from_bytes(b, "little")


def catalog():
    out = list(
        csv.DictReader(
            (HERE / "native_libc_binding_contracts.tsv").read_text().splitlines(),
            delimiter="\t",
        )
    )
    for r in out:
        for k in ["id", "kind", "fixed_prefix", "noreturn"]:
            r[k] = int(r[k])
        r["params"] = [] if r["params"] == "-" else r["params"].split(",")
        for k in ["library_keys", "actions", "roles"]:
            r[k] = list(map(int, r[k].split(",")))
    return out


def source(r, safe=False, sysv=False, params=None, result=None):
    ps = r["params"] if params is None else params
    rt = r["result"] if result is None else result
    return (
        'extern "C"'
        + (" callconv(.sysv)" if sysv else "")
        + " { "
        + ("safe " if safe else "")
        + "fn "
        + r["name"]
        + "("
        + ",".join(f"p{i}:{t}" for i, t in enumerate(ps))
        + ")"
        + ("" if rt == "void" else " -> " + rt)
        + "; }\n"
    )


def mcabi(r, safe=False, sysv=False):
    name = r["name"].encode()
    return (
        b"MCABI001"
        + struct.pack("<II", 1, len(name))
        + name
        + bytes([int(sysv), 0 if safe else 1, 0])
        + struct.pack("<I", len(r["params"]))
        + typ(r["result"])
        + b"".join(typ(t) for t in r["params"])
    )


def expected(r, src, row=0, safe=False, sysv=False, wire=None):
    wire = mcabi(r, safe, sysv) if wire is None else wire
    types = [r["result"]] + r["params"]
    parts = []
    for i, t in enumerate(types):
        mask = (
            1
            | (2 if r["kind"] == 2 else 4 if r["kind"] == 3 else 0)
            | (8 if t.startswith("*") else 0)
            | (16 if r["actions"][i] in [1, 2] else 0)
            | (32 if r["noreturn"] else 0)
        )
        parts.append(
            struct.pack(
                "<5q",
                key(t),
                r["library_keys"][i],
                r["actions"][i],
                r["roles"][i],
                mask,
            )
        )
    texts = [
        r["name"].encode(),
        r["version"].encode(),
        b"libc.so.6",
        b"/lib64/ld-linux-x86-64.so.2",
    ]
    payload = b"".join(parts) + src + wire + b"".join(texts)
    head = [
        int.from_bytes(b"MCBND001", "little"),
        192 + len(payload),
        1,
        1,
        row,
        0,
        0,
        r["kind"],
        int(r["kind"] == 2),
        r["fixed_prefix"],
        r["noreturn"],
        int(sysv),
        0 if safe else 1,
        0,
        len(r["params"]),
        len(src),
        len(wire),
        *[len(t) for t in texts],
        len(types),
        r["id"],
        0,
    ]
    assert len(head) == 24
    return struct.pack("<24q", *head) + payload


def enumerate_cases():
    rows = []
    contracts = catalog()

    def add(name, src="", r=None, mode=0, error=0, offset=0, regerr=0, **kw):
        c = dict(
            name=name,
            source=src,
            contract=r["id"] if r else 0,
            mode=mode,
            error=error,
            offset=offset,
            registry_error=regerr,
            row=0,
            profile=1,
            edit=0,
            value=0,
            length=0,
            checkrow=0,
            checkprofile=1,
            scalar=0,
            safe=False,
            sysv=False,
        )
        c.update(kw)
        rows.append(c)

    for r in contracts:
        for label, safe, sysv in [
            ("default", False, False),
            ("safe", True, False),
            ("sysv", False, True),
        ]:
            add(r["name"] + "_" + label, source(r, safe, sysv), r, safe=safe, sysv=sysv)
        wrong = "i64" if r["result"] != "i64" else "i32"
        s = source(r, result=wrong)
        add(r["name"] + "_wrong_result", s, r, error=6, offset=s.index(r["name"]))
        s = source(r, params=r["params"] + ["i64"])
        add(r["name"] + "_wrong_count", s, r, error=6, offset=s.index(r["name"]))
        if r["params"]:
            p = r["params"][:]
            p[0] = "i64" if p[0] != "i64" else "i32"
            s = source(r, params=p)
            add(r["name"] + "_wrong_param", s, r, error=6, offset=s.index(r["name"]))
        add(r["name"] + "_roundtrip", source(r), r, mode=1)
        if any(t.startswith("*") for t in r["params"]):
            ps = r["params"][:]
            i = next(i for i, t in enumerate(ps) if t.startswith("*"))
            ps[i] = (
                ps[i].replace("*mut", "*const", 1)
                if ps[i].startswith("*mut")
                else ps[i].replace("*const", "*mut", 1)
            )
            s = source(r, params=ps)
            add(r["name"] + "_qualifier", s, r, error=6, offset=s.index(r["name"]))
    close = next(r for r in contracts if r["name"] == "close")
    s = source(close)
    for name in ["getpid", "puts", "__libc_start_main", "unknown_foreign"]:
        ss = s.replace("close", name)
        add("unknown_" + name, ss, error=5, offset=ss.index(name))
    add("duplicate", s + s, error=3, offset=len(s) + s.index("close"), regerr=5)
    ss = "type i32=u32; " + s
    add("primitive_shadow", ss, error=3, offset=ss.index("i32"), regerr=3)
    add("target", s, close, error=2, profile=2)
    add("negative_row", s, close, error=4, row=-1)
    add("past_row", s, close, error=4, row=1)
    for i in range(24):
        original = expected(close, s.encode())[i * 8]
        add(
            "forged_header_" + str(i),
            s,
            close,
            mode=2,
            error=9,
            edit=i * 8,
            value=original ^ 1,
        )
    for i in range(10):
        at = 192 + i * 8
        original = expected(close, s.encode())[at]
        add(
            "forged_slot_" + str(i),
            s,
            close,
            mode=2,
            error=9,
            edit=at,
            value=original ^ 1,
        )
    for label, at in [
        ("source", 272),
        ("wire", 272 + len(s)),
        ("version", len(expected(close, s.encode())) - 34 - len(close["version"])),
    ]:
        original = expected(close, s.encode())[at]
        add("forged_" + label, s, close, mode=2, error=9, edit=at, value=original ^ 1)
    for name in ["open", "openat", "fcntl", "prctl"]:
        r = next(x for x in contracts if x["name"] == name)
        for label, at, val in [
            ("callready", 48, 1),
            ("libraryvarargs", 64, 0),
            ("fixedclass", 56, 1),
        ]:
            add(
                name + "_bypass_" + label,
                source(r),
                r,
                mode=2,
                error=9,
                edit=at,
                value=val,
            )
    add(
        "source_changed_comment",
        s + "//x\n",
        close,
        mode=3,
        error=9,
        edit=len(s) + 2,
        value=121,
    )
    add("stale_row", s, close, mode=4, error=4, checkrow=1)
    add("stale_profile", s, close, mode=5, error=2, checkprofile=2)
    add("trailing_plan", s, close, mode=6, error=9)
    for n in [0, 191, 192, 16777217, 9223372036854775807]:
        add(
            "truncated_or_huge_" + str(n),
            s,
            close,
            mode=7,
            error=9 if n == 192 else 1,
            length=n,
        )
    add("owned_source_release", s, close, mode=8)
    for n in [-1, 4194305, 9223372036854775807]:
        add("source_length_" + str(n), s, close, mode=10, error=1, length=n)
    for action in [1, 2]:
        for value in [
            0,
            2147483647,
            2147483648,
            4294967295,
            4294967296,
            -1,
            -4294967295,
        ]:
            add(f"view_{action}_{value}", mode=9, value=action, scalar=value)
    for action in [0, 3, 4, -1]:
        add(f"unsupported_view_{action}", mode=9, value=action, error=1)
    assert len(set(c["name"] for c in rows)) == len(rows)
    for c in rows:
        c["expected_stdout_hex"] = expected_output(c).hex()
    return rows


def expected_output(c):
    contracts = {r["id"]: r for r in catalog()}
    src = c["source"].encode()
    diag = [c["error"], c["offset"], c["registry_error"], 0, 0]
    want = b""
    if c["mode"] == 9 and c["error"] == 0:
        value = c["scalar"] & 0xFFFFFFFF
        if c["value"] == 2 and value >= 0x80000000:
            value -= 0x100000000
        diag[4] = value
    if c["mode"] == 0 and c["error"] == 0:
        want = expected(contracts[c["contract"]], src, c["row"], c["safe"], c["sysv"])
        diag[4] = len(want)
    return struct.pack("<5q", *diag) + want


def limit():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (500, 550))


def mutation_plans(product):
    text = product.decode()
    edits = [
        (
            "parameter_comparison",
            "close_wrong_param",
            "ncb_param_same",
            "return 0;",
            "return 1;",
        ),
        (
            "whole_plan_comparison",
            "forged_header_6",
            "ncb_check",
            "same = ncb_same(plan, fresh, size);",
            "same = 1;",
        ),
        (
            "library_varargs",
            "open_default",
            "ncb_lib_varargs",
            "return 1;",
            "return 0;",
        ),
        (
            "callready",
            "close_default",
            "ncb_header",
            "__mind_store_i64(out + 48, 0);",
            "__mind_store_i64(out + 48, 1);",
        ),
        (
            "source_snapshot",
            "close_default",
            "ncb_payload",
            "ncb_copy(base, snapshot, n)",
            "ncb_zero(base, n)",
        ),
    ]
    plans = []
    for name, case, function, old, new in edits:
        begin = text.index("pub fn " + function + "(")
        end = text.index("\n}", begin) + 2
        body = text[begin:end]
        if name == "parameter_comparison":
            # The first return0 preserves a prior mismatch; mutate the actual
            # current type comparison only, after its explicit inequality.
            pos = body.index("if ncb_type_key(ty)")
            before, after = body[:pos], body[pos:]
            assert after.count(old) == 1
            body = before + after.replace(old, new, 1)
        else:
            assert body.count(old) == 1, (name, body.count(old))
            body = body.replace(old, new, 1)
        changed = (text[:begin] + body + text[end:]).encode()
        assert changed != product
        plans.append((name, case, changed))
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enumerate", action="store_true")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--seed", type=Path)
    p.add_argument("--seed-sha256")
    p.add_argument("--mindc", type=Path)
    p.add_argument("--mindc-sha256")
    p.add_argument("--source-evidence", type=Path)
    p.add_argument("--native", type=Path)
    p.add_argument("--native-sha256")
    p.add_argument("--oracle-sha256")
    p.add_argument("--mutation-source", type=Path)
    p.add_argument("--mutation-source-sha256")
    p.add_argument("--oracle", type=Path)
    a = p.parse_args()
    e = a.evidence.resolve()
    if a.enumerate:
        e.mkdir(parents=True, exist_ok=False)
        b = (json.dumps(enumerate_cases(), indent=2) + "\n").encode()
        (e / "cases.json").write_bytes(b)
        (e / "cases.sha256").write_text(sha(b) + "\n")
        os.chmod(e / "cases.json", 0o444)
        print("FROZEN", len(enumerate_cases()), sha(b))
        return
    data = (e / "cases.json").read_bytes()
    assert sha(data) == (e / "cases.sha256").read_text().strip()
    cases = json.loads(data)
    assert cases == enumerate_cases()
    (e / "execution.started").open("x").close()
    assert a.seed and a.mindc, "Both compiler identities are required"
    a.seed, a.mindc = a.seed.resolve(), a.mindc.resolve()
    assert sha(a.seed.read_bytes()) == a.seed_sha256
    assert sha(a.mindc.read_bytes()) == a.mindc_sha256
    paths = [HERE / "main.mind", HERE / "native_libc_binding.mind",
             HERE / "native_libc_binding_fixture.mind"]
    std_paths = sorted((ROOT / "std").rglob("*.mind"))
    dependencies = paths + std_paths + [Path(__file__).resolve(),
        HERE / "_stdlib_manifest.py", HERE / "testdata/stdlib_manifest.txt",
        HERE / "native_libc_binding_contracts.tsv",
        HERE / "native_libc_binding_mutations.tsv",
        HERE / "testdata/native_libc_binding/views.c"]
    # The two build routes consume the same captured sources and exact std tree.
    captured = {str(x.resolve()): x.read_bytes() for x in dependencies}
    inputs = {name: sha(b) for name, b in captured.items()}
    inputs.update({str(a.seed): a.seed_sha256, str(a.mindc): a.mindc_sha256})
    sys.path.insert(0, str(HERE))
    from _stdlib_manifest import seed_modules
    modules = seed_modules()
    prefix = b"\n".join(captured[str((ROOT / "std" / (n + ".mind")).resolve())]
                          for n in modules) + b"\n"
    source = b"\n".join(captured[str(x.resolve())] for x in paths)
    combined = prefix + source
    packet = struct.pack("<6q", int.from_bytes(b"MINDLAY\x01", "little"),
                         536870912, 134217728, 4096, len(prefix), len(combined)) + combined
    (e / "harness.snapshot.py").write_bytes(captured[str(Path(__file__).resolve())])
    (e / "SOURCE_HASHES.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (e / "STDLIB.json").write_text(json.dumps(dict(modules=modules,
        prefix_sha256=sha(prefix), tree=[str(x.resolve()) for x in std_paths]), indent=2) + "\n")
    (e / "assembled.mind").write_bytes(combined)
    (e / "build.packet").write_bytes(packet)
    receipts = []
    env = os.environ.copy()
    env.update(LC_ALL="C", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MIND_STDLIB_PATH=str(ROOT / "std"))
    for k in list(env):
        if k.startswith("LD_") or k in ("GLIBC_TUNABLES", "MINDC_NATIVE_ELF", "MINDC_SO"):
            env.pop(k)

    tools = {}
    for key, default in [("MLIR_OPT", "mlir-opt"), ("MLIR_TRANSLATE", "mlir-translate"),
                         ("CLANG", "clang"), ("CC", "cc"), ("GCC", "gcc")]:
        selected = env.get(key, default)
        resolved = shutil.which(selected, path=env.get("PATH"))
        assert resolved, ("Missing build instrument", key, selected)
        path = Path(resolved).resolve()
        inputs[str(path)] = sha(path.read_bytes())
        tools[key] = dict(selected=selected, path=str(path), sha256=inputs[str(path)])
    (e / "INPUTS.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (e / "BUILD_ENVIRONMENT.json").write_text(json.dumps(dict(
        stdlib=env["MIND_STDLIB_PATH"], tools=tools), indent=2) + "\n")

    def intact():
        assert (e / "cases.json").read_bytes() == data, "Case drift"
        assert sorted(str(x.resolve()) for x in (ROOT / "std").rglob("*.mind")) == [str(x.resolve()) for x in std_paths], "Std tree drift"
        for name, digest in inputs.items():
            assert sha(Path(name).read_bytes()) == digest, ("Input drift", name)

    intact()

    def run(name, args, data=None, cwd=None, timeout=600):
        q = subprocess.run(
            list(map(str, args)),
            input=data,
            capture_output=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
            preexec_fn=limit,
        )
        stem = f"{len(receipts):04d}-{name}"
        (e / (stem + ".stdout")).write_bytes(q.stdout)
        (e / (stem + ".stderr")).write_bytes(q.stderr)
        receipts.append(
            dict(
                name=name,
                argv=list(map(str, args)),
                exit=q.returncode,
                stdin_sha256=None if data is None else sha(data),
                cwd=str(cwd or Path.cwd()),
                stdout_sha256=sha(q.stdout),
                stderr_sha256=sha(q.stderr),
            )
        )
        (e / "executions.json").write_text(json.dumps(receipts, indent=2) + "\n")
        return q

    if a.native:
        assert a.source_evidence, "Reused drivers require original source/build evidence"
        prior = a.source_evidence.resolve()
        manifest = json.loads((prior / "BUILD_OUTPUTS.json").read_text())
        # Check the original full source packet and MLIR source, not just flags/hashes.
        assert (prior / "build.packet").read_bytes() == packet
        assert (prior / "mlir/main.mind").read_bytes() == source
        old_inputs = json.loads((prior / "INPUTS.json").read_text())
        # Runner revisions do not change compiled source; record them explicitly.
        # Every other source/compiler/std/tool input and the test contract stay exact.
        runner = str(Path(__file__).resolve())
        assert {k: v for k, v in old_inputs.items() if k != runner} == {
            k: v for k, v in inputs.items() if k != runner}, "Reused build dependencies differ"
        assert (prior / "cases.json").read_bytes() == data, "Reused case contract differs"
        assert sha((prior / "harness.snapshot.py").read_bytes()) == old_inputs[runner]
        native, oracle = a.native.resolve(), a.oracle.resolve()
        assert sha(native.read_bytes()) == a.native_sha256 == manifest["native"]["sha256"]
        assert sha(oracle.read_bytes()) == a.oracle_sha256 == manifest["mlir"]["sha256"]
        (e / "REUSE.json").write_text(json.dumps(dict(source_evidence=str(prior),
            manifest_sha256=sha((prior / "BUILD_OUTPUTS.json").read_bytes()),
            packet_sha256=sha(packet), producer_harness_sha256=old_inputs[runner],
            runner_harness_sha256=inputs[runner]), indent=2) + "\n")
    else:
        q = run("native-build", [a.seed], packet)
        assert q.returncode == 0 and q.stdout.startswith(b"\x7fELF") and not q.stderr, (
            q.returncode,
            q.stderr[-1000:],
        )
        native = e / "binding.native"
        native.write_bytes(q.stdout)
        native.chmod(0o700)
        project = e / "mlir"
        project.mkdir()
        (project / "Mind.toml").write_text(
            '[package]\nname="binding-diagnostic"\nversion="0.0.0"\n[build]\nentry="main.mind"\n'
        )
        (project / "main.mind").write_bytes(source)
        oracle = project / "binding.mlir"
        q = run(
            "mlir-build",
            [
                a.mindc,
                "--allow-nondeterministic",
                "build",
                "--backend",
                "mlir",
                "--target",
                "cpu",
                "--emit",
                "binary",
                "--optimize",
                "release",
                "--no-cache",
                "--out",
                oracle,
            ],
            cwd=project,
        )
        assert q.returncode == 0 and oracle.exists(), (q.returncode, q.stderr[-1500:])
    intact()
    output_hashes = {"native": dict(path=str(native), sha256=sha(native.read_bytes())),
                     "mlir": dict(path=str(oracle), sha256=sha(oracle.read_bytes()))}
    (e / "BUILD_OUTPUTS.json").write_text(json.dumps(output_hashes, indent=2) + "\n")
    reference = e / "views-reference"
    q = run(
        "c-view-build",
        [
            tools["GCC"]["path"],
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            HERE / "testdata/native_libc_binding/views.c",
            "-o",
            reference,
        ],
    )
    assert q.returncode == 0 and q.stderr == b""
    contracts = {r["id"]: r for r in catalog()}
    for c in cases:
        src = c["source"].encode()
        head = [
            c["mode"],
            len(src),
            c["row"],
            c["profile"],
            c["edit"],
            c["value"],
            c["length"],
            c["checkrow"],
            c["checkprofile"],
            c["scalar"],
        ]
        inp = struct.pack("<10q", *head) + src
        (e / (c["name"] + ".input")).write_bytes(inp)
        diag = [c["error"], c["offset"], c["registry_error"], 0, 0]
        want = b""
        if c["mode"] == 9 and c["error"] == 0:
            value = c["scalar"] & 0xFFFFFFFF
            if c["value"] == 2 and value >= 0x80000000:
                value -= 0x100000000
            diag[4] = value
        if c["mode"] == 0 and c["error"] == 0:
            want = expected(
                contracts[c["contract"]], src, c["row"], c["safe"], c["sysv"]
            )
            diag[4] = len(want)
        output = struct.pack("<5q", *diag) + want
        assert output.hex() == c["expected_stdout_hex"]
        for side, exe in [("native", native), ("mlir", oracle)]:
            q = run(c["name"] + "-" + side, [exe], inp, timeout=25)
            assert (q.returncode, q.stdout, q.stderr) == (0, output, b""), (
                c["name"],
                side,
                q.returncode,
                q.stderr,
                q.stdout[:80],
                output[:80],
            )
        if c["mode"] == 9 and c["error"] == 0:
            q = run(
                c["name"] + "-c-reference",
                [reference],
                struct.pack("<2q", c["value"], c["scalar"]),
                timeout=15,
            )
            assert (q.returncode, q.stdout, q.stderr) == (0, output[32:40], b"")
        print("PASS", c["name"], flush=True)
    print("PASS", len(cases), "native/MLIR exact declaration-binding pairs", flush=True)
    if a.mutation_source:
        assert a.seed and sha(a.seed.read_bytes()) == a.seed_sha256
        packet = a.mutation_source.read_bytes()
        assert sha(packet) == a.mutation_source_sha256
        assert packet == (e / "build.packet").read_bytes(), "Mutation source closure mismatch"
        header = list(struct.unpack_from("<6q", packet))
        combined = packet[48:]
        assert header[5] == len(combined)
        product = (HERE / "native_libc_binding.mind").read_bytes()
        assert combined.count(product) == 1
        plans = mutation_plans(product)
        frozen = []
        for name, case_name, changed in plans:
            case = next(c for c in cases if c["name"] == case_name)
            (e / (name + ".mutant.mind")).write_bytes(changed)
            frozen.append(
                dict(
                    name=name,
                    case=case_name,
                    source_sha256=sha(changed),
                    expected_original_stdout_sha256=sha(
                        bytes.fromhex(case["expected_stdout_hex"])
                    ),
                )
            )
        (e / "MUTANTS_FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(
            "FROZEN MUTANTS", sha((e / "MUTANTS_FROZEN.json").read_bytes()), flush=True
        )
        for name, case_name, changed in plans:
            altered = combined.replace(product, changed)
            head = header[:]
            head[5] = len(altered)
            mutant_packet = struct.pack("<6q", *head) + altered
            (e / (name + ".build.packet")).write_bytes(mutant_packet)
            q = run(name + "-build", [a.seed], mutant_packet)
            assert (
                q.returncode == 0
                and q.stderr == b""
                and q.stdout.startswith(b"\x7fELF")
            ), (name, "build failure is not detection", q.returncode, q.stderr[-500:])
            exe = e / (name + ".native")
            exe.write_bytes(q.stdout)
            exe.chmod(0o700)
            q = run(
                name + "-execute",
                [exe],
                (e / (case_name + ".input")).read_bytes(),
                timeout=25,
            )
            assert q.returncode == 0 and q.stderr == b"" and len(q.stdout) >= 40
            assert struct.unpack_from("<q", q.stdout)[0] == 0, (
                name,
                "must accept normally, not fail another guard",
            )
            case = next(c for c in cases if c["name"] == case_name)
            assert q.stdout != bytes.fromhex(case["expected_stdout_hex"]), (
                name,
                "escaped detection",
            )
            if name != "whole_plan_comparison":
                size = struct.unpack_from("<q", q.stdout, 32)[0]
                assert size >= 192 and len(q.stdout) == 40 + size
            print("MUTATION DETECTED", name, "normal incorrect acceptance", flush=True)
    intact()
    for row in output_hashes.values():
        assert sha(Path(row["path"]).read_bytes()) == row["sha256"], "Executable drift"
    (e / "INPUTS_FINAL.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print("PASS complete source/compiler/std/executable drift checks", flush=True)
    hashes = {
        str(x.relative_to(e)): sha(x.read_bytes()) for x in e.rglob("*") if x.is_file()
    }
    (e / "ARTIFACTS.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
