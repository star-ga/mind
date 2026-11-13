#!/usr/bin/env python3
"""Frozen scalar SysV plan cases, compared with native and fresh MLIR drivers.

This checks metadata decisions, not actual foreign-call execution. Enumeration
does not build or execute MIND. Every runtime outcome must be a normal exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
from _stdlib_manifest import seed_blob

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def words(values):
    return struct.pack('<' + 'q' * len(values), *values)


def ty(tags, length=None):
    return words([len(tags) if length is None else length]) + bytes(tags).ljust(40, b'\0')


WIDTH = {0: 0, 1: 8, 2: 8, 3: 16, 4: 16, 5: 32, 6: 32,
         7: 64, 8: 64, 9: 64, 10: 64, 12: 64, 13: 8, 14: 64}
NORM = {1: 1, 3: 1, 5: 1, 2: 2, 4: 2, 6: 2, 13: 3}


def cases():
    result = []

    def add(name, parameters, places, ret=(0,), *, cc=0, safety=1,
            varargs=0, count=None, error=0, raw_result=None):
        count = len(parameters) if count is None else count
        data = words([cc, safety, varargs, count]) + (raw_result or ty(ret))
        data += b''.join(p if isinstance(p, bytes) else ty(p) for p in parameters)
        if error:
            expected = words([error, 0])
        else:
            codes = [14 if len(p) > 1 else p[0] for p in parameters]
            rt = 14 if len(ret) > 1 else ret[0]
            gp = sum(kind == 1 for kind, _ in places)
            sse = sum(kind == 2 for kind, _ in places)
            spilled = 8 * sum(kind == 3 for kind, _ in places)
            plan = [1, count, gp, sse, spilled, (spilled + 15) // 16 * 16,
                    rt, 0 if rt == 0 else (2 if rt == 12 else 1),
                    WIDTH[rt], NORM.get(rt, 0)]
            for t, (kind, where) in zip(codes, places, strict=True):
                plan += [t, kind, where, WIDTH[t], NORM.get(t, 0)]
            expected = words([0, len(plan) * 8] + plan)
        result.append((name, data, expected))

    add('void-no-args', [], [])
    for tag in list(range(1, 11)) + [12, 13]:
        add(f'scalar-{tag}-arg-return', [(tag,)], [(2 if tag == 12 else 1, 0)], (tag,))
    add('pointer-const', [(32, 2)], [(1, 0)], (32, 2))
    add('pointer-nested-f32-pointee', [(33, 32, 11)], [(1, 0)], (33, 32, 11))
    add('pointer-max-depth', [tuple([32, 33] * 16 + [5])], [(1, 0)], (7,))
    add('explicit-sysv-safe', [(5,)], [(1, 0)], (5,), cc=1, safety=0)
    for n in (6, 7, 8):
        add(f'gp-{n}', [(7,)] * n,
            [(1, i) for i in range(6)] + [(3, i * 8) for i in range(n - 6)])
    for n in (8, 9, 10):
        add(f'sse-{n}', [(12,)] * n,
            [(2, i) for i in range(8)] + [(3, i * 8) for i in range(n - 8)])
    add('sse-exhaustion-leaves-gp', [(12,)] * 9 + [(5,)],
        [(2, i) for i in range(8)] + [(3, 0), (1, 0)])
    add('gp-exhaustion-leaves-sse', [(7,)] * 7 + [(12,)],
        [(1, i) for i in range(6)] + [(3, 0), (2, 0)])
    add('interleaved-spills-stay-source-ordered', [(7,)] * 6 + [(12,)] * 9 + [(5,), (12,)],
        [(1, i) for i in range(6)] + [(2, i) for i in range(8)] + [(3, 0), (3, 8), (3, 16)])
    add('alternating-independent-pools', [(7,), (12,)] * 8,
        [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2),
         (1, 3), (2, 3), (1, 4), (2, 4), (1, 5), (2, 5),
         (3, 0), (2, 6), (3, 8), (2, 7)])
    add('gp-max256', [(7,)] * 256, [(1, i) for i in range(6)] + [(3, i * 8) for i in range(250)])
    add('sse-max256', [(12,)] * 256, [(2, i) for i in range(8)] + [(3, i * 8) for i in range(248)])
    for cc in (-1, 2, 3, 9223372036854775807):
        add(f'unsupported-convention-{cc}', [], [], cc=cc, error=1)
    for safety in (-1, 2):
        add(f'invalid-safety-{safety}', [], [], safety=safety, error=2)
    for varargs in (-1, 1, 2):
        add(f'unsupported-varargs-{varargs}', [], [], varargs=varargs, error=3)
    for count in (-1, 257, 9223372036854775807):
        add(f'invalid-count-{count}', [], [], count=count, error=4)
    add('f32-parameter', [(11,)], [], error=6)
    add('f32-result', [], [], ret=(11,), error=6)
    for label, record in [('empty', ty(())), ('negative-length', ty((), -1)),
                          ('length34', ty([32] * 33 + [5])),
                          ('invalid-tag', ty((15,))), ('terminal-pointer', ty((32,))),
                          ('bad-prefix', ty((7, 5))), ('void-pointee', ty((32, 0)))]:
        add(f'bad-parameter-{label}', [record], [], error=5)
        add(f'bad-result-{label}', [], [], raw_result=record, error=5)
    add('void-parameter', [(0,)], [], error=5)
    add('first-parameter-error-preserved', [(0,), (11,)], [], error=5)
    add('first-f32-error-preserved', [(11,), (0,)], [], error=6)
    return result


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def build_drivers(args):
    """Build the two MIND drivers using explicit hash-checked instruments."""
    assert args.seed_sha256 and args.mindc and args.mindc_sha256
    seed, mindc = args.seed.resolve(), args.mindc.resolve()
    assert sha(seed.read_bytes()) == args.seed_sha256, 'SEED_IDENTITY'
    assert sha(mindc.read_bytes()) == args.mindc_sha256, 'MINDC_IDENTITY'
    out = args.evidence.resolve() / 'build'
    out.mkdir(exist_ok=False)
    library = (HERE / 'native_c_call_plan.mind').read_bytes()
    driver = (HERE / 'native_c_call_plan_driver.mind').read_bytes()
    user = library + b'\n' + driver
    prefix = seed_blob()
    combined = prefix + user
    packet = words([int.from_bytes(b'MINDLAY\x01', 'little'), 268435456,
                    134217728, 4096, len(prefix), len(combined)]) + combined
    (out / 'assembled.mind').write_bytes(combined)
    (out / 'input.packet').write_bytes(packet)
    hashes = {'library': sha(library), 'driver': sha(driver),
              'stdlib': sha(prefix), 'packet': sha(packet),
              'seed': args.seed_sha256, 'mindc': args.mindc_sha256,
              'harness': sha(Path(__file__).read_bytes())}
    (out / 'INPUTS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('LD_') or key in ('GLIBC_TUNABLES', 'MINDC_NATIVE_ELF'):
            env.pop(key)
    env.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def compiler_limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def build(label, argv, data=None, cwd=None):
        r = subprocess.run(list(map(str, argv)), input=data, capture_output=True,
                           timeout=240, preexec_fn=compiler_limits, env=env, cwd=cwd)
        (out / (label + '.stdout')).write_bytes(r.stdout)
        (out / (label + '.stderr')).write_bytes(r.stderr)
        (out / (label + '.exit')).write_text(str(r.returncode) + '\n')
        (out / (label + '.argv.json')).write_text(json.dumps(list(map(str, argv))) + '\n')
        assert r.returncode == 0, (label, r.returncode)
        return r

    # Execute an owned copy so immutable non-executable compiler evidence is
    # usable without changing the original file's permission or contents.
    copied = out / 'compiler.elf'
    copied.write_bytes(seed.read_bytes())
    copied.chmod(0o500)
    result = build('native', [copied], packet)
    assert result.stdout.startswith(b'\x7fELF') and not result.stderr
    native = out / 'native.elf'
    native.write_bytes(result.stdout)
    native.chmod(0o500)
    project = out / 'mlir'
    project.mkdir()
    (project / 'main.mind').write_bytes(user)
    (project / 'Mind.toml').write_text(
        '[package]\nname="c_call_plan_check"\nversion="0.1.0"\n'
        'authors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
    oracle = project / 'oracle.elf'
    build('oracle', [mindc, 'build', '--backend', 'mlir', '--target', 'cpu',
                     '--emit', 'binary', '--release', '--no-cache', '--out', oracle], cwd=project)
    assert oracle.is_file()
    assert sha(seed.read_bytes()) == args.seed_sha256
    assert sha(mindc.read_bytes()) == args.mindc_sha256
    assert sha((HERE / 'native_c_call_plan.mind').read_bytes()) == hashes['library']
    assert sha((HERE / 'native_c_call_plan_driver.mind').read_bytes()) == hashes['driver']
    return native, oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enumerate', action='store_true')
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--native', type=Path)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--seed', type=Path, help='Build both drivers instead of accepting --native/--oracle')
    p.add_argument('--seed-sha256')
    p.add_argument('--mindc', type=Path)
    p.add_argument('--mindc-sha256')
    args = p.parse_args()
    rows = cases()
    manifest = [{'name': n, 'input_sha256': sha(i), 'stdout_sha256': sha(o),
                 'exit': 0, 'stderr_sha256': sha(b'')} for n, i, o in rows]
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    if args.enumerate:
        args.evidence.mkdir(parents=True, exist_ok=False)
        (args.evidence / 'CASES.json').write_bytes(payload)
        (args.evidence / 'CASES.json').chmod(0o444)
        for n, i, o in rows:
            (args.evidence / (n + '.input')).write_bytes(i)
            (args.evidence / (n + '.expected')).write_bytes(o)
        print(f'ENUMERATED {len(rows)} {sha(payload)}')
        return 0
    assert (args.evidence / 'CASES.json').read_bytes() == payload, 'CASE_DRIFT'
    if args.seed:
        assert not args.native and not args.oracle, 'ambiguous artifact source'
        args.native, args.oracle = build_drivers(args)
    assert args.native and args.oracle, 'both independently built drivers required'
    receipts = []
    for label, exe in [('native', args.native), ('oracle', args.oracle)]:
        identity = sha(exe.read_bytes())
        for name, data, want in rows:
            assert (args.evidence / (name + '.input')).read_bytes() == data
            assert (args.evidence / (name + '.expected')).read_bytes() == want
            r = subprocess.run([str(exe.resolve())], input=data, capture_output=True,
                               timeout=15, preexec_fn=limited)
            stem = args.evidence / (label + '-' + name)
            stem.with_suffix('.stdout').write_bytes(r.stdout)
            stem.with_suffix('.stderr').write_bytes(r.stderr)
            okay = r.returncode == 0 and not r.stderr and r.stdout == want
            receipts.append({'side': label, 'name': name, 'executable_sha256': identity,
                             'exit': r.returncode, 'stdout_sha256': sha(r.stdout),
                             'stderr_sha256': sha(r.stderr), 'pass': okay})
            (args.evidence / 'RESULTS.json').write_text(json.dumps(receipts, indent=2) + '\n')
        assert sha(exe.read_bytes()) == identity, 'EXECUTABLE_DRIFT'
    passed = sum(r['pass'] for r in receipts)
    print(f'OUTCOMES {passed}/{len(receipts)}')
    return int(passed != len(receipts))


if __name__ == '__main__':
    raise SystemExit(main())
