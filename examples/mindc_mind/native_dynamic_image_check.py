#!/usr/bin/env python3
"""Bounded standalone image-extension diagnostics, never compiler qualification."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LIMIT = 2 << 30
CASES_SHA = '7a5db32aad478fcd944b182b3e857f251e766d119b1d2baaa84c07ddf39f560f'
BASE_WRITER_SHA = '468dc87cdb41da33939d23696eda9ebd980ff75bae91cd82cf5c2e68367081aa'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def inspect(blob, mode):
    """Independent ELF/provenance/geometry model, including all metadata pointers."""
    assert blob[:7] == b'\x7fELF\x02\x01\x01'
    assert struct.unpack_from('<HHI', blob, 16) == (2, 62, 1)
    entry, phoff = struct.unpack_from('<QQ', blob, 24)
    assert entry == 0x401000 and phoff == 64
    phsize, phnum = struct.unpack_from('<HH', blob, 54)
    literal_size = {1: 1, 2: 4096, 3: 4097, 4: 43, 8: 1, 16: 8388608}.get(mode, 0)
    assert (phsize, phnum) == (56, 13 + bool(literal_size))
    ph = [struct.unpack_from('<IIQQQQQQ', blob, 64 + i * 56) for i in range(phnum)]
    types = [6, 3, 1, 1, 1, 1, 1] + ([1] if literal_size else []) + [1, 1, 2, 0x6474e551, 0x6474e552, 4]
    assert [h[0] for h in ph] == types
    loads = [h for h in ph if h[0] == 1]
    assert [h[1] for h in loads] == [4, 5, 4, 6, 6] + ([4] if literal_size else []) + [6, 6]
    previous_va = 0
    previous_file = 0
    rounded = lambda n: (n + 4095) // 4096 * 4096
    for _, flags, off, va, pa, fs, ms, align in loads:
        assert pa == va and 0 <= fs <= ms and ms > 0
        assert off + fs <= len(blob) and align == 4096 and off % align == va % align == 0
        assert va >= previous_va and flags & 3 != 3
        previous_va = va + rounded(ms)
        if fs:
            assert off >= previous_file
            previous_file = off + rounded(fs)
    assert previous_file == len(blob) and len(blob) % 4096 == 0
    assert loads[0][2:7] == (0, 0x400000, 0x400000, 4096, 4096)
    assert loads[1][2] == 4096 and loads[1][3] == 0x401000
    assert loads[1][5] == loads[1][6]
    meta, relro, context = loads[2:5]
    assert meta[2] == rounded(4096 + loads[1][5])
    assert meta[3] == 0x400000 + meta[2]
    assert relro[2] == rounded(meta[2] + meta[5]) and relro[3] == 0x400000 + relro[2]
    assert relro[5] == relro[6] and relro[5] % 4096 == 0
    assert context[2] == 0 and context[3] == relro[3] + relro[6]
    assert context[5:7] == (0, 4096) and context[3] + 4096 <= 0x800000
    arena, stack = loads[-2:]
    assert arena[2:7] == (0, 0x1000000, 0x1000000, 0, 8388608)
    overflow_base = (0x1000000 + 8388608 + 4096 + 1048575) // 1048576 * 1048576
    overflow_size = 0 if mode == 43 else 1048576
    assert stack[2] == 0 and stack[3] == overflow_base + overflow_size + 4096
    assert stack[5:7] == (0, 1048576)
    if literal_size:
        lit = loads[5]
        assert lit[2] == relro[2] + relro[5] and lit[3] == 0x800000
        assert lit[5:7] == (literal_size, literal_size)
        assert blob[lit[2]:lit[2] + literal_size] == b'A' * literal_size
        assert blob[lit[2] + literal_size:] == bytes(rounded(literal_size) - literal_size)
    assert ph[0] == (6, 4, 64, 0x400040, 0x400040, phnum * 56, phnum * 56, 8)
    interp = ph[1]
    expected_interp = b'/' + b'a' * 254 if mode == 18 else b'/lib64/ld-linux-x86-64.so.2'
    assert interp[2] == 64 + phnum * 56 and interp[3] == 0x400000 + interp[2]
    assert interp[5:7] == (len(expected_interp) + 1,) * 2
    assert blob[interp[2]:interp[2] + interp[5]] == expected_interp + b'\0'
    note = ph[-1]
    assert note[1] == 4 and note[2] == interp[2] + 256 and note[2] % 4 == 0
    assert note[3] == 0x400000 + note[2] and note[5:8] == (52, 52, 4)
    expected_note = struct.pack('<III', 5, 32, 1 if mode == 4 else 2) + b'MIND\0\0\0\0' + bytes(range(32))
    assert blob[note[2]:note[2] + 52] == expected_note
    assert blob[interp[2] + interp[5]:note[2]] == bytes(note[2] - interp[2] - interp[5])
    assert blob[note[2] + 52:4096] == bytes(4096 - note[2] - 52)
    assert ph[-3] == (0x6474e551, 6, 0, 0, 0, 0, 0, 16)
    assert ph[-2] == (0x6474e552, 4, relro[2], relro[3], relro[3], relro[5], relro[6], 1)
    dynamic = ph[-4]
    assert dynamic[0:2] == (2, 6) and dynamic[5:8] == (256, 256, 8)
    assert relro[2] <= dynamic[2] <= relro[2] + relro[5] - 256
    assert dynamic[3] == 0x400000 + dynamic[2]
    pairs = [struct.unpack_from('<QQ', blob, dynamic[2] + i * 16) for i in range(16)]
    tags = dict(pairs)
    assert len(tags) == 16 and pairs[-1] == (0, 0)
    assert tags[30] == 8 and tags[0x6ffffffb] == 1

    def offset(va, size):
        for h in loads:
            if h[3] <= va and va + size <= h[3] + h[5]:
                return h[2] + va - h[3]
        raise AssertionError(('dynamic pointer outside file-backed LOAD', va, size))

    ni = {14: 80, 42: 256}.get(mode, 2)
    assert tags[11] == 24 and tags[9] == 24 and tags[8] == ni * 24
    stroff = offset(tags[5], tags[10])

    def text(index):
        assert 0 <= index < tags[10]
        end = blob.index(0, stroff + index, stroff + tags[10])
        return blob[stroff + index:end]

    assert text(tags[1]) == b'libc.so.6'
    syms = offset(tags[6], (ni + 1) * 24)
    assert blob[syms:syms + 24] == bytes(24)
    names = []
    for i in range(ni):
        name, info, other, section, value, size = struct.unpack_from('<IBBHQQ', blob, syms + (i + 1) * 24)
        assert (info, other, section, value, size) == (18, 0, 0, 0, 0)
        names.append(text(name))
    assert names == sorted(names) and len(set(names)) == ni and names[0] == b'__libc_start_main'
    rh = offset(tags[7], ni * 24)
    for i in range(ni):
        va, info, addend = struct.unpack_from('<QQq', blob, rh + i * 24)
        assert va == relro[3] + 8 * i and info == ((i + 1) << 32) | 6 and addend == 0
        assert blob[offset(va, 8):offset(va, 8) + 8] == bytes(8)
    assert offset(tags[4], (ni + 4) * 4) >= meta[2]
    vh = offset(tags[0x6ffffffe], 16)
    vversion, nvers, soname, aux, nxt = struct.unpack_from('<HHIII', blob, vh)
    assert (vversion, soname, aux, nxt) == (1, 1, 16, 0)
    assert nvers == ni and tags[0x6fffffff] == 1
    versions = struct.unpack_from('<' + 'H' * (ni + 1), blob, offset(tags[0x6ffffff0], (ni + 1) * 2))
    assert versions[0] == 0 and set(versions[1:]) == set(range(2, ni + 2))
    for i in range(nvers):
        _, flags, ident, name, following = struct.unpack_from('<IHHII', blob, offset(tags[0x6ffffffe] + 16 + i * 16, 16))
        assert flags == 0 and ident == i + 2 and following == (16 if i < nvers - 1 else 0)
        assert text(name)
    # The old linker owns FF15 relocation resolution. Check every target lies in
    # the exact GOT, including unreachable retained symbols in the large cases.
    code = blob[4096:4096 + loads[1][5]]
    patches = [32]
    if mode in list(range(1, 12)) + [16, 43]:
        patches += [len(code) - 4]
    else:
        patches += [50]
        if mode in (14, 42):
            patches += [74 + (i - 2) * 6 + 2 for i in range(2, ni)]
    for displacement in patches:
        assert code[displacement - 2:displacement] == b'\xff\x15'
        target = 0x401000 + displacement + 4 + struct.unpack_from('<i', code, displacement)[0]
        assert relro[3] <= target < relro[3] + 8 * ni and (target - relro[3]) % 8 == 0
    return loads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--seed-sha256', required=True)
    p.add_argument('--mindc', type=Path, required=True)
    p.add_argument('--mindc-sha256', required=True)
    p.add_argument('--evidence', type=Path, required=True)
    args = p.parse_args()
    e = args.evidence.resolve()
    e.mkdir(parents=True, exist_ok=False)
    seed, mindc = args.seed.resolve(), args.mindc.resolve()
    assert digest(seed.read_bytes()) == args.seed_sha256
    assert digest(mindc.read_bytes()) == args.mindc_sha256
    packet = (HERE / 'native_dynamic_image_cases.tsv').read_bytes()
    assert digest(packet) == CASES_SHA
    cases = list(csv.DictReader(packet.decode().splitlines(), delimiter='\t'))
    assert len(cases) == 82
    (e / 'cases.tsv').write_bytes(packet)
    corrections = (HERE / 'native_dynamic_image_expectation_corrections.tsv').read_bytes()
    expected_correction = {'id': '12', 'field': 'runtime_exit', 'original': '127', 'corrected': '1', 'reason': 'GNU loader missing-version diagnostic exits 1; missing-symbol case 13 remains 127. Original frozen packet retained.'}
    assert list(csv.DictReader(corrections.decode().splitlines(), delimiter='\t')) == [expected_correction]
    assert cases[12]['id'] == '12' and cases[12]['runtime_exit'] == '127'
    cases[12]['runtime_exit'] = '1'
    (e / 'expectation-corrections.tsv').write_bytes(corrections)
    print('EXPLICIT EXPECTATION CORRECTION', digest(corrections), 'case12 loader exit127->1; original packet unchanged', flush=True)
    mutation_packet = (HERE / 'native_dynamic_image_mutations.tsv').read_bytes()
    mutation_cases = list(csv.DictReader(mutation_packet.decode().splitlines(), delimiter='\t'))
    assert [(m['name'], m['case']) for m in mutation_cases] == [('skip_expected_layout_comparison', '100'), ('skip_note_structure_validation', '25'), ('context_readonly_header', '0')]
    (e / 'mutations.tsv').write_bytes(mutation_packet)
    print('PRELISTED MUTATIONS', len(mutation_cases), digest(mutation_packet), flush=True)
    print('PRELISTED', len(cases), CASES_SHA, flush=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('LD_') or key in ('GLIBC_TUNABLES', 'MINDC_NATIVE_ELF'):
            env.pop(key)
    env.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    preload = Path('/etc/ld.so.preload')
    assert not preload.exists() or not any(x.partition('#')[0].strip() for x in preload.read_text().splitlines())
    rows = []

    def run(label, argv, data=None, cwd=None, timeout=180):
        r = subprocess.run(list(map(str, argv)), input=data, capture_output=True,
                           cwd=cwd, env=env, timeout=timeout, preexec_fn=limited)
        stem = f'{len(rows):03d}-{label}'
        (e / (stem + '.stdout')).write_bytes(r.stdout)
        (e / (stem + '.stderr')).write_bytes(r.stderr)
        rows.append(f'{stem}\t{r.returncode}\t{digest(r.stdout)}\t{digest(r.stderr)}\t{list(map(str, argv))!r}\n')
        (e / 'executions.tsv').write_text('id\texit\tstdout_sha256\tstderr_sha256\targv\n' + ''.join(rows))
        return r

    from native_dynamic_writer_check import encoder
    original = (HERE / 'native_dynamic.mind').read_bytes()
    assert digest(original) == BASE_WRITER_SHA
    source = (HERE / 'main.mind').read_text()
    encoders = '\n'.join(encoder(source, n) for n in ('nb_push', 'nb_le', 'nb_u16', 'nb_u32', 'nb_i64le')).encode()
    parts = [encoders, original]
    for name in ['native_dynamic_layout.mind', 'native_dynamic_image.mind', 'native_dynamic_image_fixture.mind']:
        parts.append((HERE / name).read_bytes())
    user = b'\n'.join(parts)
    # Exact same standalone source on both backends. The full native std seed
    # includes unrelated argv-only helpers unavailable in the MLIR instrument.
    # This component needs only vec/string; no production seed is changed.
    prefix = b'\n'.join((HERE.parents[1] / 'std' / f'{name}.mind').read_bytes()
                        for name in ['vec', 'string']) + b'\n'
    combined = prefix + user
    (e / 'assembled.mind').write_bytes(combined)
    source_files = ['main.mind', 'native_dynamic.mind', 'native_dynamic_layout.mind', 'native_dynamic_image.mind', 'native_dynamic_image_fixture.mind', 'native_dynamic_image_cases.tsv', 'native_dynamic_image_expectation_corrections.tsv', 'native_dynamic_image_mutations.tsv', 'native_dynamic_image_check.py']
    (e / 'INPUT_HASHES.txt').write_text(''.join(f'{digest((HERE / f).read_bytes())}  {f}\n' for f in source_files) + f'{digest(prefix)}  stdlib_prefix\n{digest(combined)}  assembled\n{args.seed_sha256}  {seed}\n{args.mindc_sha256}  {mindc}\n')
    image = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'), 268435456, 134217728, 4096, len(prefix), len(combined)) + combined
    (e / 'input.packet').write_bytes(image)
    nb = run('native-build', [seed], image, timeout=240)
    assert nb.returncode == 0 and nb.stderr == b'' and nb.stdout.startswith(b'\x7fELF'), ('native build', nb.returncode, nb.stderr[-800:])
    native = e / 'writer.native.elf'
    native.write_bytes(nb.stdout)
    native.chmod(0o700)
    project = e / 'mlir-project'
    project.mkdir()
    (project / 'Mind.toml').write_text('[package]\nname = "dynamic-image-diagnostic"\nversion = "0.0.0"\n[build]\nentry = "main.mind"\n')
    (project / 'main.mind').write_bytes(combined)
    mlir = e / 'writer.mlir.elf'
    mb = run('mlir-build', [mindc, '--allow-nondeterministic', 'build', '--backend', 'mlir', '--target', 'cpu', '--emit', 'binary', '--optimize', 'debug', '--no-cache', '--out', mlir], cwd=project, timeout=300)
    assert mb.returncode == 0 and mlir.exists(), ('MLIR build', mb.returncode, mb.stderr[-1200:])
    (e / 'ARTIFACTS.txt').write_text(f'{digest(native.read_bytes())}  {native}\n{digest(mlir.read_bytes())}  {mlir}\n')
    print('BUILD both MIND artifacts PASS', flush=True)
    accepted = 0
    refused = 0
    runtime = 0
    for c in cases:
        mode = int(c['id'])
        outputs = []
        for backend, exe in [('native', native), ('mlir', mlir)]:
            r = run(f'emit-{mode}-{backend}', [exe], data=struct.pack('<q', mode))
            if c['category'] == 'refuse':
                assert (r.returncode, r.stdout, r.stderr) == (23, b'', b'ndx image refused\n'), (mode, backend, 'normal refusal', r.returncode, r.stderr)
                outputs.append(None)
            else:
                assert r.returncode == 0 and r.stderr == b'', (mode, backend, r.returncode, r.stderr)
                inspect(r.stdout, mode)
                outputs.append(r.stdout)
        assert outputs[0] == outputs[1], ('native/MLIR bytes differ', mode)
        if c['category'] == 'refuse':
            refused += 1
            print('REFUSE', mode, c['name'], flush=True)
            continue
        accepted += 1
        elf = e / f'case-{mode}.elf'
        elf.write_bytes(outputs[0])
        elf.chmod(0o700)
        if c['category'] != 'structure':
            repetitions = []
            for leg in ['native-origin', 'mlir-origin']:
                r = run(f'execute-{mode}-{leg}', [elf])
                assert r.returncode == int(c['runtime_exit']) and r.stdout == bytes.fromhex(c['stdout_hex']), (mode, leg, r.returncode, r.stdout, r.stderr)
                if c['category'] == 'loader':
                    marker = b'GLIBC_999.999' if mode == 12 else b'ndx_missing_symbol_20261015'
                    assert marker in r.stderr and (b'not found' in r.stderr or b'undefined symbol' in r.stderr)
                else:
                    assert r.stderr == b'', (mode, leg, r.stderr)
                repetitions.append((r.returncode, r.stdout, r.stderr))
                runtime += 1
            assert repetitions[0] == repetitions[1]
        print('PASS', mode, c['name'], digest(outputs[0]), flush=True)
    loader = Path('/lib64/ld-linux-x86-64.so.2').resolve()
    listing = run('loader-list', [loader, '--list', e / 'case-0.elf'])
    assert listing.returncode == 0 and listing.stderr == b''
    import re
    match = re.search(rb'libc\.so\.6 => (/[^\s]+)', listing.stdout)
    assert match
    libc = Path(os.fsdecode(match[1])).resolve()
    (e / 'DEPENDENCIES.txt').write_text(f'{digest(loader.read_bytes())}  {loader}\n{digest(libc.read_bytes())}  {libc}\n')
    for mode in [0, 1, 14, 42]:
        r = run(f'readelf-{mode}', [shutil.which('readelf'), '-W', '-h', '-l', '-d', '-r', '-V', '-n', e / f'case-{mode}.elf'])
        assert r.returncode == 0 and not r.stderr
    env['MIND_DYNAMIC_SEED'] = str(seed)
    env['MIND_DYNAMIC_SEED_SHA256'] = args.seed_sha256
    env['MIND_DYNAMIC_EVIDENCE'] = str(e / 'original-writer')
    old = run('original-writer-regression', [sys.executable, HERE / 'native_dynamic_writer_check.py'], timeout=300)
    assert old.returncode == 0, ('original writer', old.stderr[-1200:])
    original_elf = (e / 'original-writer/getpid.elf').read_bytes()
    new_elf = (e / 'case-0.elf').read_bytes()
    assert new_elf[4096:] == original_elf[4096:], 'existing linked payload changed'
    assert digest((HERE / 'native_dynamic.mind').read_bytes()) == BASE_WRITER_SHA
    edits = [
        (b'if ndx_same_plan(plan, expected) == 0 {', b'if 0 != 0 {'),
        (b'if ndx_note_ok(note) == 0 {', b'if 0 != 0 {'),
        (b'out = nd_ph(out, 1, 6, 0, nd_word(plan, 11), 0, 4096, 4096);', b'out = nd_ph(out, 1, 4, 0, nd_word(plan, 11), 0, 4096, 4096);'),
    ]
    mutant_rows = []
    for i, (m, (before, after)) in enumerate(zip(mutation_cases, edits)):
        assert combined.count(before) == 1
        mutant = combined.replace(before, after)
        (e / f'mutant-{i}.mind').write_bytes(mutant)
        data = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'), 268435456, 134217728, 4096, len(prefix), len(mutant)) + mutant
        built = run(f'mutant-{i}-build', [seed], data, timeout=240)
        assert built.returncode == 0 and built.stderr == b'' and built.stdout.startswith(b'\x7fELF'), ('mutation build failure is not detection', m)
        exe = e / f'mutant-{i}.elf'
        exe.write_bytes(built.stdout)
        exe.chmod(0o700)
        result = run(f'mutant-{i}-case-{m["case"]}', [exe], struct.pack('<q', int(m['case'])))
        assert result.returncode == 0 and result.stderr == b'' and result.stdout.startswith(b'\x7fELF'), ('crash/refusal is not planned wrong-result detection', m, result.returncode)
        if i == 2:
            try:
                inspect(result.stdout, 0)
            except AssertionError:
                pass
            else:
                raise AssertionError('independent structural checker missed readonly context')
        mutant_rows.append(f'{m["name"]}\t{digest(mutant)}\t{digest(built.stdout)}\t{m["case"]}\tDETECTED_NORMAL_WRONG_RESULT\n')
        print('MUTATION DETECTED', m['name'], 'normal exit0 with incorrect emitted result', flush=True)
    (e / 'mutation-results.tsv').write_text('mutation\tsource_sha256\texecutable_sha256\tcase\tverdict\n' + ''.join(mutant_rows))
    print(f'PASS {len(cases)} cases: {accepted} accepted images, {refused} normal refusals, {runtime} loader/runtime outcomes; original writer unchanged', flush=True)
    (e / 'VERDICT.txt').write_text(f'PASS\ncases={len(cases)}\naccepted={accepted}\nrefused={refused}\nruntime={runtime}\nmutations={len(mutation_cases)}\n')


if __name__ == '__main__':
    main()
