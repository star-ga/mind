#!/usr/bin/env python3
"""Explicit-instrument NSI_STDIN_V1 diagnostic, not compiler qualification.

All adapters, fault seams and note fixtures execute MIND. Python enumerates
immutable byte inputs, orchestrates bounded processes and compares evidence.
No compiler, pin, toolchain or source is downloaded or silently substituted.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

from _selfhost_sources import capture_bundle
import self_host_source_note_smoke as historical

HERE = Path(__file__).resolve().parent
DATA = HERE / 'testdata/native_source_driver'
COMPONENTS = tuple('examples/mindc_mind/' + name for name in (
    'native_dynamic_layout.mind', 'native_emission_context.mind',
    'native_dynamic.mind', 'native_dynamic_image.mind',
    'native_dynamic_allocator.mind', 'native_libc_startup.mind',
    'native_libc_io.mind', 'native_source_image.mind'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, DATA / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_hash(path):
    return digest(Path(path).read_bytes())


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('seed', 'mindc', 'prefix'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--llvm-dir', type=Path, default=Path('/usr/lib/llvm-20/bin'))
    p.add_argument('--libc', type=Path, default=Path('/lib/x86_64-linux-gnu/libc.so.6'))
    p.add_argument('--loader', type=Path, default=Path('/lib64/ld-linux-x86-64.so.2'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--reuse-manifest', type=Path)
    p.add_argument('--reuse-manifest-sha256')
    return p.parse_args()


def main():
    require(__debug__, 'do not run the diagnostic with Python assertions disabled')
    args = arguments()
    args.repo = args.repo.resolve(strict=True)
    for name in ('seed', 'mindc', 'prefix'):
        path = getattr(args, name).resolve(strict=True)
        setattr(args, name, path)
        require(file_hash(path) == getattr(args, name + '_sha256'), name + ' hash mismatch')
    require(bool(args.reuse_manifest) == bool(args.reuse_manifest_sha256),
            'reuse requires an explicit manifest and expected hash')
    out = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / 'prepared'
    prepared.mkdir()
    bundle = capture_bundle(args.repo, COMPONENTS,
                            driver='examples/mindc_mind/native_source_driver.mind')
    prefix = args.prefix.read_bytes()
    require(prefix == bundle.std_prefix,
            'prefix does not equal captured ordered current std; do not substitute a historical seed header')
    captured = out / 'captured'
    captured.mkdir()
    for source in bundle.files:
        target = captured / source.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.data)
    input_hashes = {source.path: source.sha256 for source in bundle.files}
    write_json(out / 'SOURCE_BUNDLE.json', {'identity': bundle.identity, 'files': input_hashes,
                                          'native_std_paths': bundle.native_std_paths})
    driver = bundle.by_role('driver')[0].data.decode()
    parts = [s.data for s in bundle.by_role('main') + bundle.by_role('component')]
    (prepared / 'source.mind').write_bytes(bundle.library_bytes + b'\n' + driver.encode())
    (prepared / 'prefix.mind').write_bytes(prefix)
    (prepared / 'driver.mind').write_text(driver)
    rows, mutations = load_helper('catalog').prepare(prepared, prefix, parts, driver)
    require(len(rows) == 38 and len(mutations) == 4, 'original adapter inventory drift')
    note_dir = prepared / 'notes'
    note_rows, note_mutations = load_helper('note_catalog').prepare(
        note_dir, parts[0].decode(), bundle.source('std/string.mind').data.decode(),
        driver, (DATA / 'note_fixture_tail.mind').read_text(), historical)
    sources = {'real': prepared / 'source.mind'}
    sources.update({p.stem: p for p in sorted((prepared / 'variants').glob('*.mind'))})
    sources.update({'note-' + p.stem: p for p in sorted(note_dir.glob('*.mind'))})
    hash_rows, hash_sources = load_helper('hash_refusal_catalog').prepare(
        prepared, (prepared / 'variants/unit.mind').read_text(),
        (prepared / rows[0]['file']).read_bytes())
    sources.update(hash_sources)
    # Full std closure is materialized once; CPU builds must use this exact copy.
    std_hashes = {s.path: s.sha256 for s in bundle.by_role('stdlib')}
    tool_paths = {'seed': args.seed, 'mindc': args.mindc,
                  'mlir_opt': args.llvm_dir / 'mlir-opt',
                  'mlir_translate': args.llvm_dir / 'mlir-translate',
                  'clang': args.llvm_dir / 'clang', 'libc': args.libc, 'loader': args.loader}
    tools = {k: {'path': str(p.resolve(strict=True)), 'sha256': file_hash(p)}
             for k, p in tool_paths.items()}
    write_json(out / 'TOOLS.json', tools)
    transport = [Path(__file__).resolve(), HERE / '_selfhost_sources.py',
                 HERE / 'self_host_source_note_smoke.py', HERE / 'self_host_loop_smoke.py',
                 DATA / 'catalog.py', DATA / 'note_catalog.py', DATA / 'note_fixture_tail.mind',
                 DATA / 'hash_refusal_catalog.py']
    transport_hashes = {str(p): file_hash(p) for p in transport}
    write_json(out / 'TRANSPORT.json', transport_hashes)
    reuse = None
    if args.reuse_manifest:
        require(file_hash(args.reuse_manifest) == args.reuse_manifest_sha256, 'reuse manifest hash mismatch')
        reuse = json.loads(args.reuse_manifest.read_text())
        require(reuse['version'] == 1, 'unsupported reuse manifest')
        require(reuse['std'] == std_hashes, 'reuse std closure mismatch')
        require(reuse['tools'] == {k: v['sha256'] for k, v in tools.items()}, 'reuse tool mismatch')
        write_json(out / 'REUSE_INPUT.json', reuse)
    files = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'PREEXECUTION.json', files)
    pre_hash = file_hash(out / 'PREEXECUTION.json')
    for name in files:
        (out / name).chmod(0o444)
    (out / 'PREEXECUTION.json').chmod(0o444)
    print(json.dumps({'preexecution_sha256': pre_hash, 'adapter_cases': len(rows),
                      'adapter_mutations': len(mutations), 'note_cases': len(note_rows),
                      'note_mutations': len(note_mutations), 'hash_refusal_cases': len(hash_rows),
                      'hash_refusal_mutations': 1, 'qualification': False}), flush=True)
    if args.prepare_only:
        return 0

    cgroup_lines = Path('/proc/self/cgroup').read_text().splitlines()
    cgroup = Path('/sys/fs/cgroup') / next(line[3:].lstrip('/') for line in cgroup_lines
                                         if line.startswith('0::'))
    maximum = (cgroup / 'memory.max').read_text().strip()
    swap = (cgroup / 'memory.swap.max').read_text().strip()
    quota, period = (cgroup / 'cpu.max').read_text().split()
    require(maximum != 'max' and int(maximum) <= 2 << 30, 'run inside an aggregate memory cgroup <=2GiB')
    require(swap == '0', 'run with cgroup swap disabled')
    require(quota != 'max' and int(quota) <= int(period), 'run with cgroup CPU quota <=one CPU')
    for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'memory.events'):
        (out / ('before-' + name)).write_bytes((cgroup / name).read_bytes())
    (out / 'affinity.json').write_text(json.dumps(sorted(os.sched_getaffinity(0))) + '\n')

    def resource_receipt():
        for name in ('memory.peak', 'memory.events', 'memory.swap.current', 'memory.current'):
            (out / ('after-' + name)).write_bytes((cgroup / name).read_bytes())
    atexit.register(resource_receipt)

    def intact():
        require(file_hash(out / 'PREEXECUTION.json') == pre_hash, 'preexecution manifest drift')
        for name, sha in files.items():
            require(file_hash(out / name) == sha, 'captured input drift: ' + name)
        for name, sha in input_hashes.items():
            require(file_hash(args.repo / name) == sha, 'source checkout drift: ' + name)
        current_std = {str(p.relative_to(args.repo)) for p in (args.repo / 'std').glob('*.mind')}
        require(current_std == set(std_hashes), 'source std membership drift')
        for role, tool in tools.items():
            require(file_hash(tool['path']) == tool['sha256'], 'tool drift: ' + role)
        for path, sha in transport_hashes.items():
            require(file_hash(path) == sha, 'transport drift: ' + path)

    env = os.environ.copy()
    for key in list(env):
        if key.startswith('LD_') or key.startswith('MIND') or key == 'GLIBC_TUNABLES':
            env.pop(key)
    env.update(LC_ALL='C', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MIND_STDLIB_PATH=str(captured / 'std'),
               MLIR_OPT=tools['mlir_opt']['path'], MLIR_TRANSLATE=tools['mlir_translate']['path'],
               CLANG=tools['clang']['path'], CC=tools['clang']['path'], MINDC_BUILD_TOOL_TIMEOUT_SECS='300')
    env['PATH'] = str(args.llvm_dir.resolve()) + ':' + env.get('PATH', '/usr/bin:/bin')
    run_dir = out / 'run'
    run_dir.mkdir()
    executions, outcomes, failures, mutation_results = [], [], [], []
    built, artifact_hashes, build_receipts = {}, {}, {}

    def limits(ignored=False):
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
        signal.signal(signal.SIGPIPE, signal.SIG_IGN if ignored else signal.SIG_DFL)

    def run(label, argv, data=b'', cwd=None, kind='', timeout=650):
        intact()
        for path, sha in artifact_hashes.items():
            require(file_hash(path) == sha, 'executable drift: ' + path)
        stem = run_dir / (str(len(executions)).zfill(4) + '-' + label)
        receipt = {'id': label, 'argv': list(map(str, argv)), 'stdin_sha256': digest(data),
                   'cwd': str(cwd or run_dir), 'timeout': timeout, 'as_limit': 2 << 30,
                   'cpu_seconds': 600, 'pipe_mode': kind}
        closefd, kwargs = None, {}
        if kind in ('closed-stdout', 'closed-stdout-ignored', 'closed-stderr'):
            readfd, closefd = os.pipe()
            os.close(readfd)
            kwargs['stdout' if kind != 'closed-stderr' else 'stderr'] = closefd
        start = time.monotonic()
        try:
            process = subprocess.Popen(receipt['argv'], stdin=subprocess.PIPE,
                stdout=kwargs.get('stdout', subprocess.PIPE), stderr=kwargs.get('stderr', subprocess.PIPE),
                cwd=cwd or run_dir, env=env, preexec_fn=lambda: limits(kind == 'closed-stdout-ignored'),
                restore_signals=False)
            try:
                stdout, stderr = process.communicate(data, timeout=timeout)
                receipt['exit'] = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                receipt['exit'] = 'TIMEOUT'
        finally:
            if closefd is not None:
                os.close(closefd)
        stdout, stderr = stdout or b'', stderr or b''
        Path(str(stem) + '.stdout').write_bytes(stdout)
        Path(str(stem) + '.stderr').write_bytes(stderr)
        receipt.update(elapsed=time.monotonic() - start, stdout_sha256=digest(stdout),
                       stderr_sha256=digest(stderr), stdout_bytes=len(stdout), stderr_bytes=len(stderr))
        executions.append(receipt)
        write_json(run_dir / 'EXECUTIONS.json', executions)
        intact()
        return receipt['exit'], stdout, stderr

    def build(variant):
        if variant in built:
            return built[variant]
        source = sources[variant].read_bytes()
        # The independent focused fixture is self-contained, without std prefix.
        native_prefix = b'' if variant.startswith('note-') else prefix
        combined = native_prefix + source
        packet = struct.pack('<6q', int.from_bytes(b'MINDLAY\x01', 'little'),
                             805306368, 402653184, 1048576, len(native_prefix), len(combined)) + combined
        project = run_dir / variant
        project.mkdir()
        (project / 'main.mind').write_bytes(source)
        (project / 'build.packet').write_bytes(packet)
        (project / 'Mind.toml').write_text('[package]\nname="source-driver-diagnostic"\nversion="0.0.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
        source_identity = {'source_sha256': digest(source), 'packet_sha256': digest(packet)}
        record = reuse.get('variants', {}).get(variant) if reuse else None
        executables = [project / 'native.elf', project / 'cpu.elf']
        if record:
            require(record['source'] == source_identity, 'reuse source/packet mismatch: ' + variant)
            require(record.get('producer_receipts'), 'reuse must identify original build receipts')
            for receipt in record['producer_receipts']:
                require(file_hash(receipt['path']) == receipt['sha256'], 'reuse producer receipt drift')
            for leg, target in zip(('native', 'cpu'), executables):
                previous = record['artifacts'][leg]
                data = Path(previous['path']).read_bytes()
                require(digest(data) == previous['sha256'], 'reuse executable hash mismatch')
                require(data.startswith(b'\x7fELF'), 'reuse executable is not ELF')
                target.write_bytes(data)
                target.chmod(0o500)
            mode = 'REUSED_PRODUCER_BUILD_NOT_FRESH_REPRODUCTION'
        else:
            rc, stdout, stderr = run(variant + '-native-build', [args.seed], packet)
            require(rc == 0 and not stderr and stdout.startswith(b'\x7fELF'), 'native build failed: ' + variant)
            executables[0].write_bytes(stdout)
            executables[0].chmod(0o500)
            rc, _, _ = run(variant + '-cpu-build', [args.mindc, 'build', '--backend', 'mlir',
                '--target', 'cpu', '--emit', 'binary', '--release', '--no-cache', '--out', executables[1]], cwd=project)
            require(rc == 0 and executables[1].is_file(), 'CPU build failed: ' + variant)
            mode = 'FRESH_BUILD'
        built[variant] = executables
        artifact_hashes.update({str(p): file_hash(p) for p in executables})
        build_receipts[variant] = {'source': source_identity, 'mode': mode,
                                  'artifacts': {str(p): file_hash(p) for p in executables}}
        write_json(run_dir / 'BUILDS.json', build_receipts)
        return executables

    def note_digest(elf):
        require(len(elf) >= 64 and elf[:4] == b'\x7fELF', 'invalid ELF')
        phoff = struct.unpack_from('<Q', elf, 32)[0]
        entry, count = struct.unpack_from('<HH', elf, 54)
        require(entry == 56 and count > 0 and phoff + entry * count <= len(elf), 'invalid header table')
        notes = []
        for i in range(count):
            p = phoff + i * entry
            if struct.unpack_from('<I', elf, p)[0] == 4:
                off = struct.unpack_from('<Q', elf, p + 8)[0]
                size = struct.unpack_from('<Q', elf, p + 32)[0]
                require(size == 52 and off + size <= len(elf), 'invalid note bounds')
                notes.append(elf[off:off + size])
        require(len(notes) == 1 and notes[0][:20] == struct.pack('<III', 5, 32, 2) + b'MIND\0\0\0\0',
                'wrong source-note type/shape')
        return notes[0][20:]

    reference, image_digests = None, {}
    for row in rows:
        pair, wire = [], (prepared / row['file']).read_bytes()
        for leg, executable in zip(('native', 'cpu'), build(row['variant'])):
            rc, stdout, stderr = run(row['id'] + '-' + leg, [executable], wire, kind=row['kind'], timeout=150)
            good = rc == row['exit'] and stderr == row['stderr'].encode()
            detail = []
            if row['kind'] == 'image' and good:
                try:
                    value = note_digest(stdout)
                    good = value == hashlib.sha256(wire[48:]).digest()
                    image_digests[row['id']] = value.hex()
                    artifact = run_dir / (row['id'] + '-' + leg + '.elf')
                    artifact.write_bytes(stdout)
                    artifact.chmod(0o500)
                    rr, ro, re = run(row['id'] + '-' + leg + '-program', [artifact, *row['args']], timeout=30)
                    good = good and (rr, ro, re) == (row['runtime'], b'', b'')
                    if row['id'] == 'C01':
                        reference = stdout
                    if row['id'] in ('C04', 'C05', 'C24', 'C26'):
                        good = good and stdout == reference
                except (RuntimeError, struct.error) as error:
                    good = False
                    detail.append(str(error))
            elif row['kind'] == 'fake':
                good = good and stdout == b'\x7fELF' + bytes(range(65, 80))
            else:
                good = good and stdout == b''
            pair.append((rc, stdout, stderr))
            outcomes.append({'id': row['id'], 'leg': leg, 'passed': good, 'detail': detail})
            if not good:
                failures.append({'id': row['id'], 'leg': leg, 'exit': rc, 'stdout_bytes': len(stdout)})
        if pair[0] != pair[1]:
            failures.append({'id': row['id'], 'error': 'native/CPU stream mismatch'})
        write_json(run_dir / 'OUTCOMES.json', outcomes)
        write_json(run_dir / 'FAILURES.json', failures)
    require(image_digests.get('C06a') != image_digests.get('C06b') and
            image_digests.get('C24') != image_digests.get('C25'), 'source-change digest control failed')
    for mutation in mutations:
        row = next(r for r in rows if r['id'] == mutation['case'])
        wire = (prepared / row['file']).read_bytes()
        for leg, executable in zip(('native', 'cpu'), build(mutation['id'])):
            rc, stdout, stderr = run(mutation['id'] + '-' + leg, [executable], wire, timeout=60)
            expected = (mutation['expected_exit'], bytes.fromhex(mutation.get('expected_stdout_hex', '')),
                        row['stderr'].encode())
            detected = isinstance(rc, int) and 0 <= rc < 128 and (rc, stdout, stderr) != expected
            mutation_results.append({'id': mutation['id'], 'leg': leg, 'detected_normal_mismatch': detected,
                                     'exit': rc, 'stdout_bytes': len(stdout)})
            if not detected:
                failures.append({'id': mutation['id'], 'leg': leg, 'error': 'mutation undetected or abnormal'})
    note_outcomes, note_mutant_results = [], []
    for row in note_rows:
        data = (note_dir / 'packets' / (row['id'] + '.stdin')).read_bytes()
        expected = (note_dir / 'packets' / (row['id'] + '.expected')).read_bytes()
        for leg, executable in zip(('native', 'cpu'), build('note-baseline')):
            actual = run('note-' + row['id'] + '-' + leg, [executable], data, timeout=150)
            good = actual == (0, expected, b'')
            note_outcomes.append({'id': row['id'], 'leg': leg, 'passed': good})
            if not good:
                failures.append({'id': row['id'], 'leg': leg, 'error': 'note supplement mismatch'})
    for mutation in note_mutations:
        data = (note_dir / 'packets' / (mutation['case'] + '.stdin')).read_bytes()
        expected = (note_dir / 'packets' / (mutation['case'] + '.expected')).read_bytes()
        for leg, executable in zip(('native', 'cpu'), build('note-' + mutation['id'])):
            rc, stdout, stderr = run('note-' + mutation['id'] + '-' + leg, [executable], data, timeout=60)
            detected = rc == 0 and not stderr and stdout != expected
            note_mutant_results.append({'id': mutation['id'], 'leg': leg, 'detected_normal_mismatch': detected})
            if not detected:
                failures.append({'id': mutation['id'], 'leg': leg, 'error': 'note mutation undetected or abnormal'})
    write_json(run_dir / 'MUTATION_RESULTS.json', mutation_results)
    write_json(run_dir / 'NOTE_OUTCOMES.json', note_outcomes)
    write_json(run_dir / 'NOTE_MUTATION_RESULTS.json', note_mutant_results)
    hash_outcomes, hash_mutations = [], []
    hash_wire = (prepared / 'hash-refusals/input.bin').read_bytes()
    for row in hash_rows:
        pair = []
        for leg, executable in zip(('native', 'cpu'), build(row['id'])):
            actual = run(row['id'] + '-' + leg, [executable], hash_wire, timeout=60)
            good = actual == (row['exit'], b'', row['stderr'].encode())
            pair.append(actual)
            hash_outcomes.append({'id': row['id'], 'leg': leg, 'passed': good})
            if not good:
                failures.append({'id': row['id'], 'leg': leg, 'error': 'SHA return category mismatch'})
        require(pair[0] == pair[1], 'SHA refusal backend mismatch')
    for leg, executable in zip(('native', 'cpu'), build('hash-allocation-mutant')):
        actual = run('hash-allocation-mutant-' + leg, [executable], hash_wire, timeout=60)
        detected = actual == (65, b'', b'source/image refused\n')
        hash_mutations.append({'leg': leg, 'detected_exact_normal_wrong_category': detected})
        if not detected:
            failures.append({'id': 'hash-allocation-mutant', 'leg': leg, 'error': 'wrong or abnormal mutation result'})
    write_json(run_dir / 'HASH_REFUSAL_OUTCOMES.json', hash_outcomes)
    write_json(run_dir / 'HASH_REFUSAL_MUTATIONS.json', hash_mutations)
    intact()
    verdict = {'adapter_cases': len(rows), 'adapter_observations': len(outcomes),
               'adapter_passed': sum(x['passed'] for x in outcomes),
               'adapter_mutants': mutation_results, 'note_observations': len(note_outcomes),
               'note_passed': sum(x['passed'] for x in note_outcomes),
               'note_mutants': note_mutant_results, 'hash_refusals': hash_outcomes,
               'hash_refusal_mutants': hash_mutations, 'failures': failures,
               'qualification': False, 'sha_full_wrapper_native': 'NOT_QUALIFIED'}
    write_json(run_dir / 'VERDICT.json', verdict)
    print(json.dumps(verdict), flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
