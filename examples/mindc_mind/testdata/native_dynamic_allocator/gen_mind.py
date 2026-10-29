"""rt.bin -> MIND byte-emitter functions (nb_le 7-byte chunks; nb_i64le for
the per-image constants). Each function takes the code String plus the eight
layout constants and returns the extended String; the chain is split into
parts of at most PER lets so no single fn grows large."""
import sys

PH = {1:"a",2:"a8",3:"st",4:"ob",5:"first",6:"ob8",7:"oe",8:"header"}
PER = 40
ARGS = "a: i64, a8: i64, st: i64, ob: i64, first: i64, ob8: i64, oe: i64, header: i64"
CALL = "a, a8, st, ob, first, ob8, oe, header"


def items(b: bytes):
    out, chunk, i = [], [], 0

    def flush():
        while chunk:
            part = bytes(chunk[:7])
            del chunk[:7]
            out.append(("le", int.from_bytes(part, "little"), len(part)))
    while i < len(b):
        if i + 8 <= len(b) and b[i + 1:i + 8] == b"\x11" * 7 and b[i] in PH:
            flush()
            out.append(("ph", PH[b[i]]))
            i += 8
        else:
            chunk.append(b[i])
            i += 1
    flush()
    return out


def main(path: str, prefix: str) -> None:
    b = open(path, "rb").read()
    its = items(b)
    parts = [its[k:k + PER] for k in range(0, len(its), PER)]
    fns = []
    for pi, part in enumerate(parts, 1):
        lines = [f"pub fn {prefix}{pi}(code: String, {ARGS}) -> String {{"]
        prev = "code"
        for j, it in enumerate(part, 1):
            name = f"k{j}"
            if it[0] == "le":
                lines.append(f"    let {name}: String = nb_le({prev}, {it[1]}, {it[2]});")
            else:
                lines.append(f"    let {name}: String = nb_i64le({prev}, {it[1]});")
            prev = name
        if pi < len(parts):
            lines.append(f"    {prefix}{pi + 1}({prev}, {CALL})")
        else:
            lines.append(f"    {prev}")
        lines.append("}")
        fns.append("\n".join(lines))
    print(f"// {len(b)} bytes, {len(its)} chunks, {len(parts)} parts")
    print("\n".join(fns))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
