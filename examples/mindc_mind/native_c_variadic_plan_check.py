#!/usr/bin/env python3
"""Capture bytes or delegate transport. All plan assertions execute in MIND."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import struct

HERE = Path(__file__).resolve().parent
DATA = HERE / 'testdata/native_c_variadic_plan'


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
        for name in ('native_c_call_plan.mind', 'native_c_variadic_plan.mind',
                     'native_c_variadic_plan_driver.mind', '_stdlib_manifest.py'):
            shutil.copyfile(HERE / name, out / name)
        shutil.copyfile(HERE / 'testdata/stdlib_manifest.txt', out / 'stdlib_manifest.txt')
        shutil.copyfile(Path(__file__), out / 'transport.py')
        for variant in ['baseline', 'allocation-zero'] + [f'M{i:02d}' for i in range(1, 7)]:
            # Patches were authored and locked before enumeration. No execution.
            source = (out / 'native_c_variadic_plan.mind').read_bytes()
            if variant != 'baseline':
                import subprocess
                patched = out / 'patch-input.mind'
                patched.write_bytes(source)
                subprocess.run(['patch', '--silent', str(patched),
                                str(out / 'fixtures' / (variant + '.patch'))], check=True)
                source = patched.read_bytes()
                patched.unlink()
            target = out / variant
            target.mkdir()
            user = (out / 'native_c_call_plan.mind').read_bytes() + b'\n' + source + b'\n' + (out / 'native_c_variadic_plan_driver.mind').read_bytes()
            (target / 'planner.mind').write_bytes(source)
            (target / 'main.mind').write_bytes(user)
            packet = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'),
                                 268435456, 134217728, 4096, len(prefix), len(prefix + user)) + prefix + user
            (target / 'input.packet').write_bytes(packet)
            (target / 'Mind.toml').write_text('[package]\nname="variadic_plan_check"\nversion="0.1.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
        paths = sorted(f for f in out.rglob('*') if f.is_file())
        (out / 'PREEXECUTION.sha256').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(out)) + '\n' for f in paths))
        print('ENUMERATED: 65 controls, 6 mutations; no MIND execution')
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
