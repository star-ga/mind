"""Independent 76-input note supplement and two actual MIND mutations."""
import hashlib
import json
import struct


def function(text, name):
    start = text.index('pub fn ' + name + '(')
    opening = text.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end] + '\n'


def prepare(out, main, strings, driver, tail, historical):
    out.mkdir()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    start = strings.index('struct String {')
    support = strings[start:strings.index('}', start) + 1] + '\n'
    support += ''.join(function(strings, n) for n in ('string_new', 'string_len', 'string_addr'))
    support += ''.join(function(main, n) for n in
                       ('hm_source_note_offset', 'hm_checked_note_record', 'hm_source_note_step'))
    wrapper = function(driver, 'nsd_compile_once')
    (out / 'baseline.mind').write_text(support + wrapper + tail)
    changes = [
        ('remove-type2', ' __mind_store_i32(note+8,2);', '',
         'historical-valid-trailing-True-kind-1'),
        ('remove-digest-check',
         'if __mind_load_i8(note+20+i)!=__mind_load_i8(hash+i){', 'if 0!=0{',
         'forged-digest-byte-31'),
    ]
    mutants = []
    for name, old, new, case in changes:
        if wrapper.count(old) != 1:
            raise ValueError('source mutation no longer applies exactly once: ' + name)
        source = (support + wrapper.replace(old, new) + tail).encode()
        (out / (name + '.mind')).write_bytes(source)
        mutants.append({'id': name, 'case': case, 'source_sha256': sha(source),
                        'required_detection': 'normal exit0, empty stderr, exact expected stdout mismatch'})
    packets = out / 'packets'
    packets.mkdir()
    rows = []

    def add(name, image, digest, expected, diag=0, initial=0):
        offset = -1
        if expected:
            count = struct.unpack_from('<H', expected, 56)[0]
            phoff = struct.unpack_from('<Q', expected, 32)[0]
            offset = next(struct.unpack_from('<Q', expected, phoff + i * 56 + 8)[0]
                          for i in range(count)
                          if struct.unpack_from('<I', expected, phoff + i * 56)[0] == 4)
        packet = struct.pack('<qq', len(image), initial) + digest + image
        output = struct.pack('<8q', len(expected), diag, 1, 1, 17, 19, 23, offset) + expected
        (packets / (name + '.stdin')).write_bytes(packet)
        (packets / (name + '.expected')).write_bytes(output)
        rows.append({'id': name, 'input_sha256': sha(packet), 'expected_sha256': sha(output),
                     'expected_exit': 0, 'expected_stderr_hex': '', 'diagnostic': diag})

    # Historical inputs/positive image bytes stay unchanged. Adapter refusal
    # envelopes differ explicitly from the original EmitState status contract.
    for name, data, expected, length, status in historical.cases() + historical.supplemental_cases():
        add('historical-' + name, data, bytes(range(32)), expected if length else b'',
            0 if length or not data else 7)
    good, _ = historical.fixture(trailing=True)
    for i in range(32):
        digest = bytearray(range(32))
        digest[i] ^= 255
        add('forged-digest-byte-' + str(i), good, bytes(digest), b'', 7)
    add('empty-lower-error-preserved', b'', bytes(range(32)), b'', 23, 23)
    if len(rows) != 76:
        raise ValueError('historical fixture inventory changed; review supplement contract')
    (out / 'CASES.json').write_text(json.dumps(rows, indent=2) + '\n')
    (out / 'MUTATIONS.json').write_text(json.dumps(mutants, indent=2) + '\n')
    return rows, mutants
