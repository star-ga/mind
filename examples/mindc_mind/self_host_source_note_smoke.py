#!/usr/bin/env python3
"""Native MIND regression for PT_NOTE tagging with trailing string rodata.

Usage: python3 self_host_source_note_smoke.py --compiler <native ELF> \
    --evidence-dir <new directory>
The Python process only supplies bytes and checks observations. Both ELF
emission and record validation execute the actual MIND compiler source.
No seed is published. Every child is capped at 2 GiB address space.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import resource
import struct
import subprocess
import tempfile
import time

import self_host_loop_smoke as loop

HERE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(*, trailing: bool = False, kind: int = 1) -> tuple[bytes, int]:
    # Deliberately reorder headers: no fixed second-phdr or file-tail shortcut.
    count = 5 if trailing else 4
    offset = 64 + count * 56 + 17
    end = offset + 52
    data = bytearray(end + (127 if trailing else 0))
    data[:7] = b'\x7fELF\x02\x01\x01'
    struct.pack_into('<HHI', data, 16, 2, 62, 1)
    struct.pack_into('<Q', data, 32, 64)
    struct.pack_into('<HHH', data, 52, 64, 56, count)
    struct.pack_into('<IIQQQQQQ', data, 64 + 3 * 56, 4, 4,
                     offset, 0x400000 + offset, 0x400000 + offset, 52, 52, 4)
    data[offset:end] = struct.pack('<III', 5, 32, kind) + b'MIND\0\0\0\0' + bytes(range(32))
    if trailing:
        data[end:] = bytes((i * 7 + 3) % 256 for i in range(127))
    return bytes(data), offset


def cases() -> list[tuple[str, bytes, bytes, int, int]]:
    rows = []
    for trailing in (False, True):
        for kind in (1, 2):
            data, off = fixture(trailing=trailing, kind=kind)
            expected = bytearray(data)
            struct.pack_into('<I', expected, off + 8, 2)
            rows.append((f'valid-trailing-{trailing}-kind-{kind}', data, bytes(expected), len(data), 19))
    good, off = fixture(trailing=True)
    ph = 64 + 3 * 56
    # All refusals must return an empty result and negative status WITHOUT
    # mutating the supplied bytes. A signal/build failure never passes.
    for size in (0, 1, 51, 52, 63):
        data = good[:size]
        rows.append((f'truncated-{size}', data, data, 0, -1))
    changes = [
        ('magic', 0, '<I', 0), ('class', 4, '<B', 1),
        ('endianness', 5, '<B', 2), ('ident-version', 6, '<B', 0),
        ('object-kind', 16, '<H', 3), ('machine', 18, '<H', 183),
        ('version', 20, '<I', 0), ('header-size', 52, '<H', 63),
        ('phdr-size', 54, '<H', 55), ('phoff-header-overlap', 32, '<Q', 63),
        ('phoff-negative', 32, '<Q', 1 << 63),
        ('phoff-overflow', 32, '<Q', (1 << 63) - 1),
        ('phnum-zero', 56, '<H', 0), ('phnum-range', 56, '<H', 65535),
        ('note-absent', ph, '<I', 1),
        ('note-overlaps-headers', ph + 8, '<Q', 64),
        ('note-truncated', ph + 8, '<Q', len(good) - 51),
        ('note-negative-offset', ph + 8, '<Q', 1 << 63),
        ('note-offset-overflow', ph + 8, '<Q', (1 << 63) - 1),
        ('note-filesz', ph + 32, '<Q', 51),
        ('note-memsz', ph + 40, '<Q', 53),
        ('note-alignment', ph + 48, '<Q', 8),
        ('owner-size', off, '<I', 4), ('digest-size', off + 4, '<I', 31),
        ('note-type', off + 8, '<I', 3), ('owner', off + 12, '<I', 0),
        ('owner-padding', off + 16, '<I', 1),
    ]
    for name, pos, fmt, val in changes:
        data = bytearray(good)
        struct.pack_into(fmt, data, pos, val)
        rows.append((name, bytes(data), bytes(data), 0, -1))
    duplicate = bytearray(good)
    duplicate[64:120] = duplicate[ph:ph+56]
    rows.append(('duplicate-note', bytes(duplicate), bytes(duplicate), 0, -1))
    return rows


def supplemental_cases() -> list[tuple[str, bytes, bytes, int, int]]:
    """Additional traversal controls; the original cases() contract stays fixed."""
    rows = []
    good, _ = fixture(trailing=True)
    ph = 64 + 3 * 56
    for name, malformed in (('invalid-before-valid', 64),
                             ('valid-before-invalid', ph)):
        data = bytearray(good)
        data[64:120] = data[ph:ph+56]
        struct.pack_into('<Q', data, malformed + 32, 51)
        rows.append((name, bytes(data), bytes(data), 0, -1))

    # Exercise the complete u16 count with constant-depth iteration. No hidden
    # recursion budget may limit a structurally valid table of this size.
    count = 65535
    note = 64 + count * 56
    large = bytearray(note + 52)
    large[:64] = good[:64]
    struct.pack_into('<H', large, 56, count)
    large[note:] = struct.pack('<III', 5, 32, 1) + b'MIND\0\0\0\0' + bytes(range(32))
    header = struct.pack('<IIQQQQQQ', 4, 4, note, 0x400000 + note,
                         0x400000 + note, 52, 52, 4)
    for name, pos in (('max-count-note-first', 64),
                      ('max-count-note-last', 64 + (count - 1) * 56)):
        data = bytearray(large)
        data[pos:pos+56] = header
        expected = bytearray(data)
        struct.pack_into('<I', expected, note + 8, 2)
        rows.append((name, bytes(data), bytes(expected), len(data), 19))
    rows.append(('max-count-note-absent', bytes(large), bytes(large), 0, -1))
    duplicate = bytearray(large)
    duplicate[64:120] = header
    last = 64 + (count - 1) * 56
    duplicate[last:last+56] = header
    rows.append(('max-count-duplicate-last', bytes(duplicate), bytes(duplicate), 0, -1))
    return rows


def limits(cpu_seconds: int = 120) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, help='Explicit diagnostic instrument; otherwise build current source from the frozen bounded seed')
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--build-cpu-seconds', type=int, default=120,
                        help='Initial compiler/diagnostic-driver build CPU budget (1..3600, default120); all fixture build/run CPU120/wall150 and 2GiB limits stay unchanged')
    args = parser.parse_args()
    if not 1 <= args.build_cpu_seconds <= 3600:
        parser.error('--build-cpu-seconds must be between1 and3600')
    out = (args.evidence_dir or (Path(tempfile.mkdtemp(prefix='mind-source-note-')) / 'run')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler = (args.compiler or (HERE / 'testdata/selfhost_loop_bounded/stage1.elf')).resolve()
    compiler_hash = digest(compiler.read_bytes())
    source_hash = digest((HERE / 'main.mind').read_bytes())
    original_cases = cases()
    added_cases = supplemental_cases()
    all_cases = original_cases + added_cases
    # Enumerate and hash every contract before building or executing either side.
    contract = [{'name': n, 'input_sha256': digest(i), 'output_sha256': digest(o),
                 'length': l, 'status': s} for n, i, o, l, s in original_cases]
    (out / 'CASES.json').write_text(json.dumps(contract, indent=2) + '\n')
    (out / 'CASES.json').chmod(0o444)
    supplement = [{'name': n, 'input_sha256': digest(i), 'output_sha256': digest(o),
                   'length': l, 'status': s} for n, i, o, l, s in added_cases]
    (out / 'SUPPLEMENTAL_CASES.json').write_text(json.dumps(supplement, indent=2) + '\n')
    (out / 'SUPPLEMENTAL_CASES.json').chmod(0o444)
    records = []

    def run(name: str, exe: Path, data: bytes, *, compiler_build: bool = False) -> subprocess.CompletedProcess:
        start = time.monotonic()
        cpu_seconds = args.build_cpu_seconds if compiler_build else 120
        wall_seconds = cpu_seconds + 30
        result = subprocess.run([str(exe)], input=data, capture_output=True,
                                timeout=wall_seconds,
                                preexec_fn=functools.partial(limits, cpu_seconds))
        (out / f'{name}.stdin').write_bytes(data)
        (out / f'{name}.stdout').write_bytes(result.stdout)
        (out / f'{name}.stderr').write_bytes(result.stderr)
        records.append({'name': name, 'argv': [str(exe)], 'exit': result.returncode,
                        'elapsed': time.monotonic() - start,
                        'compiler_build': compiler_build,
                        'cpu_soft_seconds': cpu_seconds, 'cpu_hard_seconds': cpu_seconds + 5,
                        'wall_seconds': wall_seconds, 'address_space_bytes': 2 << 30,
                        'stdout_sha256': digest(result.stdout),
                        'stderr_sha256': digest(result.stderr)})
        (out / 'RESULTS.json').write_text(json.dumps(records, indent=2) + '\n')
        assert digest(compiler.read_bytes()) == compiler_hash, 'COMPILER_DRIFT'
        assert digest((HERE / 'main.mind').read_bytes()) == source_hash, 'SOURCE_DRIFT'
        return result

    # Use the same standard-library composition as the real self-host driver.
    combined, image, lo = loop.build_seed()
    bootstrap_hash = compiler_hash
    if args.compiler is None:
        # The established bounded self-host layout; never rewrite frozen seeds.
        bootstrap = struct.pack('<4q', int.from_bytes(b'MINDLAY\x01', 'little'),
                                805306368, 402653184, 1048576) + image
        built = run('compiler-build', compiler, bootstrap, compiler_build=True)
        assert built.returncode == 0 and not built.stderr and built.stdout.startswith(b'\x7fELF'), 'COMPILER_BUILD_FAILED'
        compiler = out / 'current-compiler.elf'
        compiler.write_bytes(built.stdout)
        compiler.chmod(0o500)
        compiler_hash = digest(built.stdout)
    original_driver = (HERE / 'selfhost_driver.mind').read_bytes()
    assert combined.endswith(original_driver)
    combined = combined[:-len(original_driver)] + (HERE / 'source_note_driver.mind').read_bytes()
    packet = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'),
                         268435456, 134217728, 4096, lo, len(combined)) + combined
    (out / 'SOURCE.json').write_text(json.dumps({
        'main_sha256': source_hash, 'compiler_sha256': compiler_hash,
        'bootstrap_sha256': bootstrap_hash,
        'combined_sha256': digest(combined), 'packet_sha256': digest(packet),
        'cases_sha256': digest((out / 'CASES.json').read_bytes()),
        'supplemental_cases_sha256': digest((out / 'SUPPLEMENTAL_CASES.json').read_bytes()),
    }, indent=2) + '\n')
    build = run('build', compiler, packet, compiler_build=True)
    assert build.returncode == 0 and not build.stderr and build.stdout.startswith(b'\x7fELF'), 'BUILD_FAILED'
    driver = out / 'source-note.elf'
    driver.write_bytes(build.stdout)
    driver.chmod(0o500)
    for name, data, expected, length, status in all_cases:
        result = run(name, driver, struct.pack('<q', len(data)) + data)
        want = struct.pack('<qq', length, status) + expected
        assert (result.returncode, result.stdout, result.stderr) == (0, want, b''), name
        print(f'[PASS] {name}', flush=True)
    # Exact production regression, including the note's meaning and digest.
    prefix = combined[:lo]
    for size in (43, 62, 4097):
        literal = (b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' * 67)[:size]
        source = (b'pub fn main() -> i64 { let s: String = "' + literal + b'"; '
                  b'let n: i64 = __mind_write(1, string_addr(s), string_len(s), 0 - 1); '
                  b'if n != string_len(s) { return 70; } 0 }\n')
        assembled = prefix + source
        image = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'),
                            268435456, 134217728, 4096, lo, len(assembled)) + assembled
        built = run(f'literal-{size}-build', compiler, image)
        assert built.returncode == 0 and not built.stderr and built.stdout.startswith(b'\x7fELF')
        elf = built.stdout
        phoff = struct.unpack_from('<Q', elf, 32)[0]
        phnum = struct.unpack_from('<H', elf, 56)[0]
        notes = [struct.unpack_from('<Q', elf, phoff + i * 56 + 8)[0]
                 for i in range(phnum) if struct.unpack_from('<I', elf, phoff + i * 56)[0] == 4]
        assert len(notes) == 1
        note = notes[0]
        assert elf[note:note+20] == struct.pack('<III', 5, 32, 2) + b'MIND\0\0\0\0'
        assert elf[note+20:note+52] == hashlib.sha256(assembled).digest()
        exe = out / f'literal-{size}.elf'
        exe.write_bytes(elf)
        exe.chmod(0o500)
        result = run(f'literal-{size}-run', exe, b'')
        assert (result.returncode, result.stdout, result.stderr) == (0, literal, b'')
        print(f'[PASS] literal-{size}-bytes-and-source-note', flush=True)
    print(f'SDLC-GATE source_note ran={len(all_cases) + 3} fail=0 '
          f'original={len(original_cases) + 3} supplemental={len(added_cases)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
