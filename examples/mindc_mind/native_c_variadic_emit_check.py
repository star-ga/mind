#!/usr/bin/env python3
"""Capture bytes or delegate transport. All call assertions execute in MIND."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import struct

HERE = Path(__file__).resolve().parent
DATA = HERE / 'testdata/native_c_variadic_emit'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enumerate', action='store_true')
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--seed', type=Path)
    p.add_argument('--seed-sha256')
    p.add_argument('--mindc', type=Path)
    p.add_argument('--mindc-sha256')
    p.add_argument('--seal-sha256', help='Owner-reviewed capture manifest digest')
    a = p.parse_args()
    out = a.evidence.resolve()
    if a.enumerate:
        from _stdlib_manifest import seed_blob, all_modules, STD_DIR
        out.mkdir(parents=True, exist_ok=False)
        shutil.copytree(DATA, out / 'fixtures')
        prefix = seed_blob()
        (out / 'stdlib.mind').write_bytes(prefix)
        (out / 'std').mkdir()
        for module in all_modules():
            shutil.copyfile(STD_DIR / (module + '.mind'), out / 'std' / (module + '.mind'))
        names = ('native_c_call_plan.mind', 'native_c_call_emit.mind',
                 'native_c_variadic_plan.mind', 'native_c_variadic_emit.mind',
                 'native_c_variadic_emit_driver.mind', 'native_c_variadic_verify.mind',
                 'main.mind', '_stdlib_manifest.py', 'native_c_call_emit_check.py',
                 'native_c_call_emit_driver.mind')
        for name in names:
            shutil.copyfile(HERE / name, out / name)
        shutil.copyfile(HERE / 'testdata/stdlib_manifest.txt', out / 'stdlib_manifest.txt')
        shutil.copyfile(Path(__file__), out / 'transport.py')
        # Capture unchanged small encoders verbatim, before any execution.
        import re
        whole = (out / 'main.mind').read_text()
        encoders = []
        for name in ('nb_push', 'nb_le', 'nb_u32', 'nb_i32le_signed',
                     'nb_load_rax', 'nb_store_rax', 'nb_fp_movsd_load_xmmk',
                     'nb_fp_movsd_store_xmm0'):
            match = re.search(r'(?m)^pub fn ' + name + r'\(', whole)
            if not match:
                p.error('ENCODER_MISSING')
            begin = whole.index('{', match.start())
            end, depth = begin + 1, 1
            while depth:
                depth += (whole[end] == '{') - (whole[end] == '}')
                end += 1
            encoders.append(whole[match.start():end] + '\n')
        (out / 'encoders.mind').write_text('\n'.join(encoders))
        variants = ['baseline', 'allocation-zero'] + [f'M{i:02d}' for i in range(1, 9)]
        for variant in variants + ['verifier']:
            target = out / variant
            target.mkdir()
            if variant == 'verifier':
                user = (out / 'native_c_variadic_verify.mind').read_bytes()
            else:
                planner = (out / 'native_c_variadic_plan.mind').read_bytes()
                emitter = (out / 'native_c_variadic_emit.mind').read_bytes()
                if variant != 'baseline':
                    import subprocess
                    patched = target / 'patched.mind'
                    patched.write_bytes(planner if variant == 'allocation-zero' else emitter)
                    subprocess.run(['patch', '--silent', str(patched),
                                    str(out / 'fixtures' / (variant + '.patch'))], check=True)
                    if variant == 'allocation-zero':
                        planner = patched.read_bytes()
                    else:
                        emitter = patched.read_bytes()
                (target / 'planner.mind').write_bytes(planner)
                (target / 'emitter.mind').write_bytes(emitter)
                user = (out / 'encoders.mind').read_bytes() + b'\n'
                for name in ('native_c_call_plan.mind', 'native_c_call_emit.mind'):
                    user += (out / name).read_bytes() + b'\n'
                user += planner + b'\n' + emitter + b'\n'
                user += (out / 'native_c_variadic_emit_driver.mind').read_bytes()
            (target / 'main.mind').write_bytes(user)
            packet = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'),
                                 268435456, 134217728, 4096, len(prefix), len(prefix + user)) + prefix + user
            (target / 'input.packet').write_bytes(packet)
            (target / 'Mind.toml').write_text('[package]\nname="variadic_emit_check"\nversion="0.1.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
        paths = sorted(f for f in out.rglob('*') if f.is_file())
        (out / 'PREEXECUTION.sha256').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(out)) + '\n' for f in paths))
        print('ENUMERATED: 55 controls, 8 mutations; no MIND execution')
        return
    if not all((a.seed, a.seed_sha256, a.mindc, a.mindc_sha256, a.seal_sha256)):
        p.error('Execution requires compiler paths, both hashes and the reviewed seal')
    manifest = (out / 'PREEXECUTION.sha256').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != a.seal_sha256:
        p.error('CAPTURE_SEAL')
    seen = set()
    for line in manifest.decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = out / name
        if (name in seen or Path(name).is_absolute() or '..' in Path(name).parts
                or path.is_symlink() or not path.resolve().is_relative_to(out)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            p.error('CAPTURE_IDENTITY')
        seen.add(name)
    if not {'fixtures/run.sh', 'transport.py'}.issubset(seen):
        p.error('CAPTURE_IDENTITY')
    if (out / 'transport.py').read_bytes() != Path(__file__).read_bytes():
        p.error('CAPTURE_IDENTITY')
    os.execv('/bin/bash', ['bash', str(out / 'fixtures/run.sh'), str(out),
                          str(a.seed.resolve()), a.seed_sha256,
                          str(a.mindc.resolve()), a.mindc_sha256])


if __name__ == '__main__':
    main()
