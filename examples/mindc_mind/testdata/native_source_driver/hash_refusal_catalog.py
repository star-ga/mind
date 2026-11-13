"""Additive normal SHA-return fault seams; original adapter cases unchanged."""
import json


def prepare(out, driver, wire):
    rows = []
    sources = {}
    call = 'sha256(source,source_length,digest)'
    allocation = 'if hash_status==2{return nsd_fail(control,source,0,70);}'
    assert driver.count(call) == 1 and driver.count(allocation) == 1
    directory = out / 'hash-refusals'
    directory.mkdir()
    (directory / 'input.bin').write_bytes(wire)
    for status in (1, 2, 9):
        name = 'hash-status-' + str(status)
        source = driver.replace(call, 'nsd_test_hash_status(source,source_length,digest)')
        source += '\npub fn nsd_test_hash_status(p:i64,n:i64,d:i64)->i64{' + str(status) + '}\n'
        path = directory / (name + '.mind')
        path.write_text(source)
        sources[name] = path
        rows.append({'id': name, 'exit': 70 if status == 2 else 65,
                     'stderr': 'allocation refused\n' if status == 2 else 'source/image refused\n'})
        if status == 2:
            mutant = directory / 'hash-allocation-mutant.mind'
            mutant.write_text(source.replace(allocation, ''))
            sources['hash-allocation-mutant'] = mutant
    (directory / 'CASES.json').write_text(json.dumps(rows, indent=2) + '\n')
    (directory / 'MUTATION.json').write_text(json.dumps({
        'id': 'hash-allocation-mutant', 'case': 'hash-status-2',
        'required_actual_exit': 65, 'required_actual_stdout': '',
        'required_actual_stderr': 'source/image refused\n',
        'interpretation': 'normal wrong category; simulated return, not heap-pressure evidence'
    }, indent=2) + '\n')
    return rows, sources
