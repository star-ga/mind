# Borrowed String append checks

Author: STARGA Inc.

`std/string.mind` uses `cap=0` for borrowed views, including read-only literals
and interior slices. `string_push_byte` allocates a fresh buffer and copies
exactly the existing bytes before appending. It never passes a borrowed address
to realloc or free, even for an empty view with a nonzero address. The initial
capacity is at least16 and grows enough to hold the complete prefix and new byte.
Owned buffers retain the existing spare-capacity reuse and realloc growth policy.
A returned header is fresh; realloc may invalidate old owned-buffer aliases.

`string_push_capacity(len,cap)` checks integer capacity calculations independently
of allocation. It returns0 for negative metadata, owned len>cap, or a length whose
increment is unrepresentable. Near the signed maximum, growth selects exactly the
required length instead of overflowing a doubling. The append operation still
requires a valid live String, representable length increment, valid ownership and
successful allocation under the selected allocator's existing policy. The helper's
zero result is not a recoverable malformed-String append API. It does not establish
pointer validity, arbitrary alias safety or common recoverable OOM behavior across
runtimes.

The executable gate explicitly embeds the current MIND module and renames every
`string_*` function to `cow_string_*` in the test snapshot. Both backends therefore
execute the same MIND implementation; a weak C compatibility symbol cannot replace
it. The compiler identities, exact embedded source, original source hash, commands,
stdout/stderr and exits are retained in the evidence directory. Native compilation
uses the explicit bounded source-image protocol, not a compiler-pin promotion.

Freeze before execution, using hash-verified compiler paths:

```sh
python3 examples/mindc_mind/string_borrowed_check.py \
  --enumerate --evidence /tmp/string-borrowed
python3 examples/mindc_mind/string_borrowed_check.py \
  --evidence /tmp/string-borrowed \
  --seed /path/to/bounded-native-compiler --seed-sha256 "$NATIVE_SHA256" \
  --mindc /path/to/mindc --mindc-sha256 "$CPU_SHA256"
```

Run the command within a2GiB/no-swap cgroup, pinned to one available CPU; the
harness additionally sets2GiB per-child address-space limits. Configure the
supported CPU MLIR tools as for other compiler gates. The40 frozen cases cover
borrowed lengths through4096, interior views, parent preservation, owned growth,
byte canaries, an empty borrowed address1 and capacity arithmetic boundaries.
Five compiled source mutations must produce normal wrong results on both backends;
signals, build failures, timeouts and unrelated traps receive no detection credit.

The helper is part of the std module surface: current module resolution auto-exports
functions when no explicit export block is present. The CPU reference shim exports
the matching weak symbol. This gate establishes the native and CPU body paths; it
does not claim support in other runtime targets.

The existing weak C shim is reference infrastructure and is tested separately:

```sh
clang -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  examples/mindc_mind/testdata/string_borrowed/reference.c -lm -pthread \
  -o /tmp/string-reference
ASAN_OPTIONS=detect_leaks=1 /tmp/string-reference
```

ASan requires a large sparse virtual shadow reservation, so keep the2GiB physical
cgroup cap and finite timeout but omit the address-space ulimit for this sanitizer
leg. It checks literal/view copying, owned realloc, suffix canaries and arithmetic
boundaries in the actual included reference source. This separate C check is not a
substitute for the MIND-body checks or whole-compiler/consumer qualification.
