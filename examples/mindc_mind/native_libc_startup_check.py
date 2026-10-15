#!/usr/bin/env python3
"""Actual dynamic-loader diagnostics for the standalone MIND startup adapter."""

import argparse, csv, hashlib, json, os, resource, struct, subprocess
from pathlib import Path
from native_dynamic_writer_check import encoder

HERE = Path(__file__).resolve().parent
TD = HERE / "testdata/native_libc_startup"
CASE_SHA = "44ab600c4edd01a31871e5a05cf55330a6bc47588511d4cef4d857d8d506c8fb"
MUT_SHA = "e08d8c3d44aa8da54197f8afeb65fcd4efc62d8a1d06f31d2863f3c700debbd9"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 200))


def fixture(name):
    lines = [
        ".intel_syntax noprefix",
        ".text",
        "fixture_main:",
        "push rbp",
        "mov rbp,rsp",
        "sub rsp,64",
        "mov [rbp-8],rdi",
        "mov [rbp-16],rsi",
    ]

    def addr(reg, kind):
        lines.append(f"movabs {reg},0x33333333333333{kind:02x}")

    def call(index):
        lines.extend([".byte 0xff,0x15", f".long 0x444444{index:02x}"])

    def check(condition):
        lines.append(condition)
        lines.append("jne failed")

    def finish(value):
        lines.extend([f"movabs rax,{value}", "leave", "ret"])

    if name == "arity0":
        finish(13)
    elif name == "arity1":
        lines.extend(["mov rax,[rbp-8]", "leave", "ret"])
    elif name == "exit_negative":
        finish(0xFFFFFFFFFFFFFFFF)
    elif name == "exit_high_bits":
        finish(0x10000002A)
    elif name == "stdio_flush":
        lines.append("mov edi,90")
        call(1)
        finish(17)
    elif name == "tls_errno":
        call(3)
        lines.extend(["mov r13,rax", "mov dword ptr [r13],123", "mov edi,-1"])
        call(4)
        check("cmp eax,-1")
        check("cmp dword ptr [r13],9")
        finish(0)
    else:
        addr("r12", 1)
        addr("r13", 2)
        addr("r14", 3)
        lines.append("movabs rax,0x3154524154534d")
        check("cmp [r12],rax")
        check("cmp qword ptr [r12+8],2")
        check("cmp qword ptr [r12+32],3")
        check("cmp qword ptr [r13+8],3")
        lines.extend(["mov rax,[r12+16]"])
        check("cmp qword ptr [rax],3")
        lines.extend(["add rax,8"])
        check("cmp [r12+40],rax")
        check("cmp [r13+16],rax")
        lines.extend(["mov rbx,[rax+8]"])
        check("cmp dword ptr [rbx],0x68706c61")
        check("cmp word ptr [rbx+4],0x0061")
        lines.extend(["mov rbx,[rax+16]"])
        check("cmp dword ptr [rbx],0x61746562")
        check("cmp byte ptr [rbx+4],0")
        lines.extend(["add rax,32"])
        check("cmp [r12+48],rax")
        lines.append("movabs rax,0x31545241444e494d")
        check("cmp [r14+528],rax")
        addr("rax", 4)
        lines.extend(["cmp rsp,rax", "jb failed"])
        addr("rax", 5)
        lines.extend(["cmp rsp,rax", "jae failed", "mov rax,rsp", "and eax,15"])
        check("cmp eax,0")
        if name in ("env_original", "env_constructor"):
            # Original envp must still reference the original vector/string even after
            # the controlled constructor extends and changes libc environ.
            lines.extend(
                [
                    "mov rbx,[r12+48]",
                    "scan_env:",
                    "mov rdi,[rbx]",
                    "test rdi,rdi",
                    "jz failed",
                ]
            )
            addr("rsi", 7)
            lines.extend(
                [
                    "add rsi,32",
                    "mov ecx,15",
                    "repe cmpsb",
                    "je found_env",
                    "add rbx,8",
                    "jmp scan_env",
                    "found_env:",
                ]
            )
            lines.append("movabs rax,0x6c616e696769726f")
            check("cmp [rdi],rax")
            check("cmp byte ptr [rdi+8],0")
            addr("rdi", 7)
            call(2)
            lines.extend(["test rax,rax", "jz failed", "mov rbx,rax"])
            if name == "env_constructor":
                lines.append("movabs rax,0x006465676e616863")
                check("cmp [rbx],rax")
                lines.extend(["mov rax,[r12+64]", "cmp rax,[r12+48]", "je failed"])
            else:
                lines.append("movabs rax,0x6c616e696769726f")
                check("cmp [rbx],rax")
            lines.extend(["mov rax,[r12+64]"])
            check("cmp [r13+24],rax")
        if name == "allocator_owned":
            lines.extend(
                [
                    "xor edi,edi",
                    "mov esi,64",
                    "call qword ptr [r14+536]",
                    "test rax,rax",
                    "jz failed",
                    "mov rbx,rax",
                ]
            )
            check("cmp qword ptr [rbx],0")
            lines.extend(
                [
                    "mov qword ptr [rbx],123",
                    "mov rdi,rbx",
                    "mov esi,128",
                    "call qword ptr [r14+552]",
                ]
            )
            check("cmp qword ptr [rax],123")
            check("cmp qword ptr [rax+64],0")
            lines.extend(["mov rdi,rax", "call qword ptr [r14+544]"])
            check("cmp qword ptr [r13+8],3")
        if name == "reentrant_adapter":
            lines.extend(["mov rdi,[r12+32]", "mov rsi,[r12+40]", "mov rdx,[r12+64]"])
            addr("rax", 6)
            lines.extend(["call rax"])
            check("cmp eax,70")
        if name == "stack_lower_guard":
            addr("rsi", 7)
            lines.extend(
                ["add rsi,96", "mov edi,1", "mov edx,1", "mov eax,1", "syscall"]
            )
            addr("rax", 4)
            lines.append("mov qword ptr [rax-8],0")
        if name == "callee_saved_clobber":
            lines.extend(
                [
                    "leave",
                    "mov ebx,11",
                    "mov r12d,12",
                    "mov r13d,13",
                    "mov r14d,14",
                    "mov r15d,15",
                    "mov ebp,16",
                    "xor eax,eax",
                    "ret",
                ]
            )
        else:
            finish(0)
    lines.extend(["failed:", "mov eax,77", "leave", "ret"])
    # Every declared fixture import has a distinct unreachable call site. These
    # are explicit diagnostic dependencies, not source reachability/admission.
    for index in range(1, 5):
        call(index)
    lines.append('.section .note.GNU-stack,"",@progbits')
    return "\n".join(lines) + "\n"


def render_mind(blob):
    # Development-only regeneration; product runtime remains entirely MIND.
    markers = {
        bytes([i]) + b"\x11" * 7: n
        for i, n in [(1, "context"), (2, "arena"), (3, "stack_top")]
    }
    markers.update({b"\x21\x22\x22\x22": "init_delta", b"\x22" * 4: "main_delta"})
    items = []
    plain = bytearray()
    i = 0

    def flush():
        while plain:
            chunk = bytes(plain[:7])
            del plain[:7]
            items.append(f"{int.from_bytes(chunk, 'little')}, {len(chunk)}")

    while i < len(blob):
        for marker, name in markers.items():
            if blob[i : i + len(marker)] == marker:
                flush()
                items.append(f"{name}, {len(marker)}")
                i += len(marker)
                break
        else:
            plain.append(blob[i])
            i += 1
    flush()
    parts = [items[j : j + 35] for j in range(0, len(items), 35)]
    lines = []
    args = "context: i64, arena: i64, stack_top: i64, init_delta: i64, main_delta: i64"
    vals = "context, arena, stack_top, init_delta, main_delta"
    for n, part in enumerate(parts, 1):
        lines.append(f"pub fn nsa_blob_p{n}(code: String, {args}) -> String {{")
        prev = "code"
        for k, arg in enumerate(part, 1):
            lines.append(f"    let b{k}: String = nb_le({prev}, {arg});")
            prev = "b" + str(k)
        lines.append(
            "    " + (f"nsa_blob_p{n + 1}({prev}, {vals})" if n < len(parts) else prev)
        )
        lines.append("}")
    return "\n".join(lines) + "\n"


def inspect_image(blob, reference, body_size, stack_bytes, reorder):
    # Independent ELF/model check, not values supplied by the MIND planner.
    assert blob[:7] == b"\x7fELF\x02\x01\x01" and struct.unpack_from(
        "<HHI", blob, 16
    ) == (2, 62, 1)
    entry, phoff = struct.unpack_from("<QQ", blob, 24)
    phsize, phnum = struct.unpack_from("<HH", blob, 54)
    assert phoff == 64 and phsize == 56 and phnum == 14
    ph = [struct.unpack_from("<IIQQQQQQ", blob, phoff + 56 * i) for i in range(phnum)]
    loads = [h for h in ph if h[0] == 1]
    assert [h[1] for h in loads] == [4, 5, 4, 6, 6, 4, 6, 6]
    code, meta, relro, ctx, lit, arena, stack = loads[1:]
    assert code[3] == 0x401000 and code[5] == code[6] == 365 + 2360 + body_size
    assert ctx[2] == 0 and ctx[5:7] == (0, 4096)
    assert arena[2:7] == (0, 0x1000000, 0x1000000, 0, 65536)
    assert lit[3] == 0x800000 and lit[5] == 128
    overflow_base = (0x1000000 + 65536 + 4096 + 1048575) // 1048576 * 1048576
    assert stack[3] == overflow_base + 1048576 + 4096 and stack[5:7] == (0, stack_bytes)
    assert ph[-3][0:2] == (0x6474E551, 6) and ph[-2][0] == 0x6474E552
    assert ph[-2][2:7] == relro[2:7]
    previous = 0
    for h in loads:
        assert (
            h[1] & 3 != 3
            and h[3] >= previous
            and h[5] <= h[6]
            and h[2] + h[5] <= len(blob)
        )
        previous = h[3] + h[6]
    startup = 2360 + body_size if reorder else 0
    runtime = 0 if reorder else 365
    body = 2360 if reorder else 365 + 2360
    assert entry == 0x401000 + startup
    expected = bytearray(reference)
    for kind, value in [(1, ctx[3]), (2, arena[3]), (3, stack[3] + stack[6])]:
        marker = bytes([kind]) + b"\x11" * 7
        while (at := expected.find(marker)) >= 0:
            expected[at : at + 8] = struct.pack("<Q", value)
    expected[221:225] = struct.pack("<i", runtime - startup - 225)
    expected[302:306] = struct.pack("<i", body - startup - 306)
    # The independently sorted five imports place __errno_location first.
    expected[115:119] = struct.pack("<i", relro[3] + 8 - (0x401000 + startup + 119))
    assert blob[code[2] + startup : code[2] + startup + 365] == expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--seed-sha256", required=True)
    p.add_argument("--mindc", type=Path, required=True)
    p.add_argument("--mindc-sha256", required=True)
    p.add_argument("--allocator", type=Path, required=True)
    p.add_argument("--allocator-sha256", required=True)
    p.add_argument("--image-writer", type=Path, required=True)
    p.add_argument("--image-writer-sha256", required=True)
    p.add_argument("--base-writer", type=Path, required=True)
    p.add_argument("--base-writer-sha256", required=True)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    e = a.evidence.resolve()
    e.mkdir(parents=True, exist_ok=False)
    rows = list(
        csv.DictReader(
            [
                s
                for s in (HERE / "native_libc_startup_cases.tsv")
                .read_text()
                .splitlines()
                if not s.startswith("#")
            ],
            delimiter="\t",
        )
    )
    assert (
        len(rows) == 45
        and sha((HERE / "native_libc_startup_cases.tsv").read_bytes()) == CASE_SHA
        and sha((HERE / "native_libc_startup_mutations.tsv").read_bytes()) == MUT_SHA
    )
    print("FROZEN", len(rows), CASE_SHA, MUT_SHA, flush=True)
    for path, h in [
        (a.seed, a.seed_sha256),
        (a.mindc, a.mindc_sha256),
        (a.allocator, a.allocator_sha256),
        (a.image_writer, a.image_writer_sha256),
        (a.base_writer, a.base_writer_sha256),
    ]:
        assert sha(path.read_bytes()) == h
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("LD_") or k == "GLIBC_TUNABLES":
            env.pop(k)
    env.update(LC_ALL="C", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    receipts = []

    def run(name, args, data=None, cwd=None, runtime=False, preload=False):
        child_env = (
            env
            if not runtime
            else {
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "NS_STARTUP_ENV": "original",
                "NS_FIXTURE_MODE": name,
            }
        )
        if preload:
            child_env["LD_PRELOAD"] = str(e / "preload.so")
        q = subprocess.run(
            list(map(str, args)),
            input=data,
            capture_output=True,
            cwd=cwd,
            env=child_env,
            timeout=240 if not runtime else 15,
            preexec_fn=limits,
        )
        stem = f"{len(receipts):03d}-{name}"
        (e / (stem + ".stdout")).write_bytes(q.stdout)
        (e / (stem + ".stderr")).write_bytes(q.stderr)
        receipts.append(
            dict(
                id=stem,
                argv=list(map(str, args)),
                exit=q.returncode,
                stdout_sha256=sha(q.stdout),
                stderr_sha256=sha(q.stderr),
                diagnostic_preload=str(e / "preload.so") if preload else None,
            )
        )
        (e / "executions.json").write_text(json.dumps(receipts, indent=2) + "\n")
        return q

    prefix = (
        b"\n".join(
            (HERE.parents[1] / "std" / f"{n}.mind").read_bytes()
            for n in ["vec", "string"]
        )
        + b"\n"
    )
    source = (HERE / "main.mind").read_text()
    enc = "\n".join(
        encoder(source, n) for n in ["nb_push", "nb_le", "nb_u16", "nb_u32", "nb_i64le"]
    ).encode()
    paths = [
        a.base_writer,
        HERE / "native_dynamic_layout.mind",
        a.image_writer,
        a.allocator,
        HERE / "native_libc_startup.mind",
        HERE / "native_libc_startup_fixture.mind",
    ]
    hashes = {
        str(x): sha(x.read_bytes())
        for x in paths
        + [Path(__file__), TD / "startup.s", TD / "gen_mind.py", TD / "preload.c"]
    }
    (e / "SOURCE_HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    combined = prefix + enc + b"\n" + b"\n".join(x.read_bytes() for x in paths)
    (e / "assembled.mind").write_bytes(combined)
    inp = (
        struct.pack(
            "<6q",
            int.from_bytes(b"MINDLAY\x01", "little"),
            268435456,
            134217728,
            4096,
            len(prefix),
            len(combined),
        )
        + combined
    )
    (e / "build.packet").write_bytes(inp)
    q = run("native-build", [a.seed], inp)
    assert q.returncode == 0 and not q.stderr and q.stdout.startswith(b"\x7fELF"), (
        q.returncode,
        q.stderr,
    )
    native = e / "fixture.native"
    native.write_bytes(q.stdout)
    native.chmod(0o700)
    project = e / "mlir"
    project.mkdir()
    (project / "main.mind").write_bytes(combined)
    (project / "Mind.toml").write_text(
        '[package]\nname="libc_startup_fixture"\nversion="0.1.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n'
    )
    oracle = project / "fixture.mlir"
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
            "debug",
            "--no-cache",
            "--out",
            oracle,
        ],
        cwd=project,
    )
    assert q.returncode == 0, (q.returncode, q.stderr)
    q = run(
        "preload-build",
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            TD / "preload.c",
            "-o",
            e / "preload.so",
        ],
    )
    assert q.returncode == 0 and not q.stderr
    # The assembly reference independently defines exact runtime bytes and labels.
    q = run(
        "reference-assemble", ["as", "--64", TD / "startup.s", "-o", e / "reference.o"]
    )
    assert q.returncode == 0 and not q.stderr
    q = run("reference-relocations", ["readelf", "-rW", e / "reference.o"])
    assert q.returncode == 0 and b"There are no relocations" in q.stdout
    q = run(
        "reference-extract",
        [
            "objcopy",
            "-O",
            "binary",
            "--only-section=.text",
            e / "reference.o",
            e / "reference.bin",
        ],
    )
    assert q.returncode == 0 and not q.stderr
    reference = (e / "reference.bin").read_bytes()
    assert len(reference) == 365
    q = run("reference-symbols", ["nm", "-n", e / "reference.o"])
    assert q.returncode == 0
    symbols = {
        z[2]: int(z[0], 16)
        for line in q.stdout.decode().splitlines()
        if len(z := line.split()) == 3
    }
    assert {
        n: symbols[n]
        for n in [
            "startup_entry",
            "main_adapter",
            "libc_fixup",
            "init_disp",
            "main_disp",
            "startup_end",
        ]
    } == dict(
        startup_entry=0,
        main_adapter=135,
        libc_fixup=115,
        init_disp=221,
        main_disp=302,
        startup_end=365,
    )
    # Explicit preload is used only by the four named constructor/finalizer tests.
    literals = bytearray(128)
    literals[:15] = b"NS_STARTUP_ENV\0"
    literals[32:56] = b"NS_STARTUP_ENV=original\0"
    literals[96] = 71
    for i, r in enumerate(rows):
        name = r["name"]
        mode = 20 + (i - 19) if r["category"] == "refusal" else 0
        asm = e / (name + ".s")
        asm.write_text(fixture(name))
        q = run(name + "-assemble", ["as", "--64", asm, "-o", e / (name + ".o")])
        assert q.returncode == 0 and not q.stderr
        q = run(
            name + "-extract",
            [
                "objcopy",
                "-O",
                "binary",
                "--only-section=.text",
                e / (name + ".o"),
                e / (name + ".bin"),
            ],
        )
        assert q.returncode == 0 and not q.stderr
        body = bytearray((e / (name + ".bin")).read_bytes())
        patches = []
        constants = []
        for index in range(1, 5):
            marker = struct.pack("<I", 0x44444400 + index)
            j = 0
            while (j := body.find(marker, j)) != -1:
                assert body[j - 2 : j] == b"\xff\x15"
                patches.append((j, index))
                body[j : j + 4] = bytes(4)
                j += 4
        for kind in range(1, 8):
            marker = bytes([kind]) + b"\x33" * 7
            j = 0
            while (j := body.find(marker, j)) != -1:
                constants.append((j, kind))
                body[j : j + 8] = bytes(8)
                j += 8
        arity = 0 if name == "arity0" else 1 if name == "arity1" else 2
        stack = (
            4096 if name == "stack_min" else 1048576 if name == "stack_large" else 65536
        )
        header = [
            mode,
            arity,
            stack,
            len(body),
            len(patches),
            len(constants),
            len(literals),
            int(name == "reordered_blobs"),
        ]
        data = (
            struct.pack("<8q", *header)
            + body
            + b"".join(struct.pack("<2q", *x) for x in patches + constants)
            + literals
        )
        (e / (name + ".input")).write_bytes(data)
        previous = None
        for side, exe in [("native", native), ("mlir", oracle)]:
            q = run(name + "-" + side + "-emit", [exe], data)
            if r["category"] == "refusal":
                assert (q.returncode, q.stdout, q.stderr) == (2, bytes(64), b""), (
                    name,
                    side,
                    q.returncode,
                    q.stderr,
                    q.stdout[:100],
                )
            else:
                assert (
                    q.returncode == 0
                    and not q.stderr
                    and q.stdout.startswith(b"\x7fELF")
                ), (name, side, q.returncode, q.stderr, q.stdout[:100])
                inspect_image(
                    q.stdout,
                    reference,
                    len(body),
                    stack,
                    int(name == "reordered_blobs"),
                )
                image = e / (name + "-" + side + ".elf")
                image.write_bytes(q.stdout)
                image.chmod(0o700)
                actual = run(
                    name,
                    [image, "alpha", "beta"],
                    runtime=True,
                    preload=r["category"] == "runtime-preload",
                )
                assert (actual.returncode, actual.stdout, actual.stderr) == (
                    int(r["exit"]),
                    bytes.fromhex("" if r["stdout_hex"] == "-" else r["stdout_hex"]),
                    b"",
                ), (name, side, actual.returncode, actual.stdout, actual.stderr)
            if previous is not None:
                assert previous == (q.returncode, q.stdout, q.stderr), (
                    name,
                    "backend mismatch",
                )
            previous = (q.returncode, q.stdout, q.stderr)
        print("PASS", name, flush=True)
    for path, h in hashes.items():
        assert sha(Path(path).read_bytes()) == h
    # Mutate the actual MIND emitter source, rebuild it, and require a normal
    # wrong observable result. Neither build failure nor signal detects a fault.
    source_product = (HERE / "native_libc_startup.mind").read_text()
    assert (
        source_product.split("// BEGIN GENERATED RUNTIME")[1].strip()
        == render_mind(reference).strip()
    )
    for ident, case, offset, old, new, expected in [
        (
            "raw_exit",
            "stdio_flush",
            350,
            bytes.fromhex("4883c408415f415e415d415c5b5dc3"),
            bytes.fromhex("89c7b8e70000000f050f0b90909090"),
            (17, b""),
        ),
        (
            "drop_rtld_fini",
            "rtld_finalizer",
            89,
            bytes.fromhex("4d8b4b18"),
            bytes.fromhex("4531c990"),
            (0, b""),
        ),
        (
            "ignore_init_failure",
            "dirty_allocator",
            228,
            bytes.fromhex("756a"),
            bytes.fromhex("9090"),
            (77, b"D"),
        ),
        (
            "wrong_stack",
            "stack_min",
            281,
            bytes.fromhex("48bc0311111111111111"),
            bytes.fromhex("90909090909090909090"),
            (77, b""),
        ),
        (
            "erase_original_env",
            "env_constructor",
            219,
            bytes.fromhex("40"),
            bytes.fromhex("30"),
            (77, b"D"),
        ),
    ]:
        assert reference[offset : offset + len(old)] == old and len(old) == len(new)
        altered = reference[:offset] + new + reference[offset + len(old) :]
        changed = (
            source_product.split("// BEGIN GENERATED RUNTIME")[0]
            + "// BEGIN GENERATED RUNTIME\n"
            + render_mind(altered)
        )
        assert changed != source_product
        (e / (ident + ".mutant.mind")).write_text(changed)
        mutant = combined.replace(source_product.encode(), changed.encode())
        assert mutant != combined
        packet = (
            struct.pack(
                "<6q",
                int.from_bytes(b"MINDLAY\x01", "little"),
                268435456,
                134217728,
                4096,
                len(prefix),
                len(mutant),
            )
            + mutant
        )
        (e / (ident + ".build.packet")).write_bytes(packet)
        q = run(ident + "-mutant-build", [a.seed], packet)
        assert q.returncode == 0 and not q.stderr and q.stdout.startswith(b"\x7fELF")
        exe = e / (ident + ".mutant.compiler")
        exe.write_bytes(q.stdout)
        exe.chmod(0o700)
        q = run(ident + "-mutant-emit", [exe], (e / (case + ".input")).read_bytes())
        assert q.returncode == 0 and not q.stderr and q.stdout.startswith(b"\x7fELF")
        elf = e / (ident + ".mutant.elf")
        elf.write_bytes(q.stdout)
        elf.chmod(0o700)
        q = run(
            case,
            [elf, "alpha", "beta"],
            runtime=True,
            preload=case in ["rtld_finalizer", "dirty_allocator", "env_constructor"],
        )
        assert (q.returncode, q.stdout, q.stderr) == (*expected, b""), (
            ident,
            q.returncode,
            q.stdout,
            q.stderr,
        )
        assert q.returncode >= 0
        print("MUTATION DETECTED", ident, "normal exit", q.returncode, flush=True)
    for path, h in hashes.items():
        assert sha(Path(path).read_bytes()) == h
    artifacts = {
        str(x.relative_to(e)): sha(x.read_bytes()) for x in e.rglob("*") if x.is_file()
    }
    (e / "ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print(
        "PASS",
        len(rows),
        "native/MLIR fixture pairs; 5 actual source mutants detected",
        flush=True,
    )


if __name__ == "__main__":
    main()
