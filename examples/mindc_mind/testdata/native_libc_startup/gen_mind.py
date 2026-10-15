"""Emit MIND byte constructors from the independently assembled ABI reference.

Development utility only. No product compiler/runtime subprocess dependency.
Input must have no unresolved ELF relocations before extracting raw .text.
"""

import sys
from pathlib import Path

PH = {
    bytes([i]) + b"\x11" * 7: n
    for i, n in [(1, "context"), (2, "arena"), (3, "stack_top")]
}
PH.update({b"\x21\x22\x22\x22": "init_delta", b"\x22" * 4: "main_delta"})


def main(path):
    b = Path(path).read_bytes()
    items = []
    i = 0
    plain = bytearray()
    seen = {x: 0 for x in PH}

    def flush():
        while plain:
            chunk = bytes(plain[:7])
            del plain[:7]
            items.append(
                ("nb_le", str(int.from_bytes(chunk, "little")) + ", " + str(len(chunk)))
            )

    while i < len(b):
        for marker, name in PH.items():
            if b[i : i + len(marker)] == marker:
                flush()
                items.append(("nb_le", name + ", " + str(len(marker))))
                seen[marker] += 1
                i += len(marker)
                break
        else:
            plain.append(b[i])
            i += 1
    flush()
    assert all(v > 0 for v in seen.values()), seen
    parts = [items[j : j + 35] for j in range(0, len(items), 35)]
    args = "context: i64, arena: i64, stack_top: i64, init_delta: i64, main_delta: i64"
    vals = "context, arena, stack_top, init_delta, main_delta"
    for n, part in enumerate(parts, 1):
        print(f"pub fn nsa_blob_p{n}(code: String, {args}) -> String {{")
        prev = "code"
        for k, (fn, arg) in enumerate(part, 1):
            print(f"    let b{k}: String = {fn}({prev}, {arg});")
            prev = "b" + str(k)
        print(
            "    " + (f"nsa_blob_p{n + 1}({prev}, {vals})" if n < len(parts) else prev)
        )
        print("}")


if __name__ == "__main__":
    main(sys.argv[1])
