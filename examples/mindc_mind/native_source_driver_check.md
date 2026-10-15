# Reproducing the source-input adapter diagnostic

`native_source_driver_check.py` tests the MIND `NSI_STDIN_V1` adapter using
explicit native and CPU compiler instruments. It does not select a compiler
pin, enable an official native bridge profile, or qualify consumer programs.
The product adapter and all fault seams are MIND. Python supplies fixed bytes,
builds fixtures, controls subprocesses and records their observations.

The diagnostic retains two separate contracts:

* The original38 adapter controls and4 actual source mutations cover framing,
  exact EOF, lengths, flags, short reads/writes, closed streams, SIGPIPE,
  lowering refusal, exact source notes, repeated image bytes and runtime output.
  Fake-image seams are explicitly labeled; they test transport, not compilation.
* The independent76-input note supplement and2 actual source mutations use
  the exact adapter wrapper and existing checked note-parser functions with a
  MIND fixture generator. It checks43 historical structured-note inputs,
  all32 digest-byte mismatches and preservation of an empty lower error.
  It requires one generator invocation, unchanged caller bytes, exact note
  bytes, preserved diagnostic words and normal execution. It does not replace
  the original static source-note gate or its3 literal-runtime regressions.

An additive SHA-return supplement tests statuses1,2 and9 through MIND fault
seams. It requires normal source refusal65, allocation refusal70 and source
refusal65 respectively, with exact empty stdout and error lines. Removing the
allocation branch must produce the exact normal wrong-category result65.
These three controls and one mutation simulate hash return values; they are
not evidence of actual heap exhaustion. The original contracts remain intact.

All inputs and mutations are enumerated and hashed before execution. A crash,
signal, timeout or failed build never counts as successful mutation detection.
The original4 mutation criteria and the stricter independent2 criteria remain
separate in the evidence. Unsupported instruments cause failure, not `SKIP`.

## Required inputs

Use Linux x86-64, Python3.11 or later, the supported LLVM20 CPU configuration,
and source-bound compiler instruments supporting this candidate's constructs.
The native instrument must support the existing bounded `MINDLAY1` protocol;
the CPU instrument must support `mindc build --backend mlir --target cpu`.
Supply each executable's independently established expected SHA256. Installing
a similarly numbered release is not a substitute for matching its hash.

The source checkout must contain the adapter, its captured component closure
and the reviewed SHA256 lifetime correction. The test captures the entire
current `std/*.mind` set and the ordered21-module native prefix through
`_selfhost_sources`. The supplied prefix must equal those captured bytes exactly.
It will not silently replace a library, use a historical seed manifest's hash,
or claim a successful full native SHA wrapper qualification. That wider wrapper
obligation remains separate from the reviewed raw SHA lifetime correction.

Create a prefix from the chosen checkout before computing its expected hash:

```sh
PYTHONPATH=examples/mindc_mind python3 - <<'PY'
from pathlib import Path
from _selfhost_sources import capture_bundle
Path('/tmp/mind-source-prefix.mind').write_bytes(capture_bundle(Path.cwd()).std_prefix)
PY
sha256sum /tmp/mind-source-prefix.mind
```

Run enumeration first, with actual paths and previously checked hashes:

```sh
python3 examples/mindc_mind/native_source_driver_check.py \
  --repo "$PWD" \
  --seed "$NATIVE_COMPILER" --seed-sha256 "$NATIVE_COMPILER_SHA256" \
  --mindc "$CPU_COMPILER" --mindc-sha256 "$CPU_COMPILER_SHA256" \
  --prefix /tmp/mind-source-prefix.mind --prefix-sha256 "$PREFIX_SHA256" \
  --evidence /tmp/source-driver-enumeration --prepare-only
```

Use a new evidence directory for execution. The runner requires a cgroup with
at most2GiB aggregate memory, swap disabled and at most one CPU's quota. It
also enforces2GiB address space and bounded CPU/wall time per subprocess.
Choose an available CPU in coordination with other sessions; do not assume the
whole machine is reserved. For example, wrap the command without `--prepare-only`:

```sh
systemd-run --user --wait --pipe \
  -p MemoryMax=2147483648 -p MemorySwapMax=0 -p CPUQuota=100% \
  -p TasksMax=64 -p RuntimeMaxSec=3600 -p KillMode=control-group \
  taskset -c "$AVAILABLE_CPU" nice -n19 \
  python3 examples/mindc_mind/native_source_driver_check.py \
  --repo "$PWD" \
  --seed "$NATIVE_COMPILER" --seed-sha256 "$NATIVE_COMPILER_SHA256" \
  --mindc "$CPU_COMPILER" --mindc-sha256 "$CPU_COMPILER_SHA256" \
  --prefix /tmp/mind-source-prefix.mind --prefix-sha256 "$PREFIX_SHA256" \
  --evidence /tmp/source-driver-run
```

`--llvm-dir`, `--libc` and `--loader` expose the CPU runtime/tool configuration.
The selected tools are hashed. Inherited compiler/runtime overrides are removed,
and CPU builds use the captured std directory. Source, std membership,
transport, executable and tool drift cause failure. `PREEXECUTION.json`,
`SOURCE_BUNDLE.json`, `TOOLS.json`, per-process streams/exits, `BUILDS.json`,
resource counters and `VERDICT.json` retain the evidence. A passing verdict
always contains `qualification: false`.

## Optional replay of previously built instruments

To avoid rebuilding unchanged full adapter instruments, an owner may supply
`--reuse-manifest FILE --reuse-manifest-sha256 HASH`. This is explicit replay
of trusted producer build evidence, not an independent rebuild or an
authenticated proof of source-to-binary correspondence. Runtime cases and
mutation controls still execute anew. Variants absent from the manifest build
fresh; no binary is downloaded.

Version1 is a JSON object with:

* `version: 1`;
* `tools`: exact SHA256 values for `seed`, `mindc`, `mlir_opt`,
  `mlir_translate`, `clang`, `libc`, `loader`;
* `std`: every captured repository-relative `std/*.mind` path and its SHA256;
* `variants`: each reusable variant's `source` object containing
  `source_sha256` and `packet_sha256`, `artifacts.native` and `artifacts.cpu`
  containing explicit `path`/`sha256`, and nonempty `producer_receipts` entries
  containing explicit `path`/`sha256` for the original build records.

The source packet, complete std set, tools, executable bytes and original
receipt files must all match. Output records distinguish
`REUSED_PRODUCER_BUILD_NOT_FRESH_REPRODUCTION` from `FRESH_BUILD`.

## Observed candidate results and remaining limits

The original frozen corrected adapter run passed76/76 paired observations and
8/8 normal source-mutant detections. Its17 variants required34 successful
builds, and20 generated programs executed successfully. Independent focused
tests passed152/152 observations and4/4 normal mutant detections. A separate
four-source independent supplement passed29 checks from24 processes, including
12 exact type2 source notes, byte-identical repeated/native/CPU image triples,
and12 expected program outcomes. Resource peaks were below2GiB with no OOM
events. These are reported candidate observations, not a claim that every
future toolchain or all consumer acceptance gates pass.

The portable runner also replayed all38/4 adapter controls using17 explicitly
reused producer builds, then freshly built the3 small note variants on both
backends. It passed76 adapter observations,8 adapter mutant detections,
152 note observations and4 note mutant detections across266 process receipts.
Peak aggregate memory was190,509,056 bytes, with zero OOM events and zero swap.
Five separately prelisted forged reuse records (tool, std, source, executable
and producer receipt hashes) each failed before any compiler/runtime call.
Prepared inputs and fixture sources matched all216 earlier frozen files byte
for byte. This replay validates the portable transport while preserving the
distinction between reused full builds and freshly rebuilt small fixtures.

The prior adapter run's18 positive failures are preserved: it labeled an exact
source digest as type1, which denotes an IR trace digest. The correction checks
all32 source-digest bytes and marks type2. Acceptance does not authenticate a
signer, sign an artifact, prove source/executable semantic equivalence or close
formal mathematical review. Original consumer qualification, frozen compiler
seed/pin work, full SHA wrapper support and performance gates remain separate.

## Final allocation-category correction

Final review found that SHA allocation refusal2 was being reported as source
refusal65. The adapter now preserves the original allocation category70.
The corrected driver SHA256 is
`d0e16c60e2d58cae29fae545ed0b5dfda36e693b2969abd5bdcac83095d11b7d`.
The final runner SHA256 is
`1a1094780698c7ff0f5511304a189afae9cd12b1861c7c18e7c2d1066b564c23`.

The final run freshly built all24 variants on both backends:48 successful
builds and316 recorded processes. It passed76 original adapter observations,
8 original adapter mutation detections,152 independent note observations,
4 note mutation detections,6 additive hash-return observations and2 exact
wrong-category mutation detections. All104 earlier adapter runtime observations
still matched exit, stdout and stderr byte for byte. Peak aggregate memory was
974,458,880 bytes; all memory-event counters and swap were zero.

The pre-execution inventory hash is
`10677a536d822ed2fcad94527b3f3d8e7a37082956a886bc655ca0f51493df37`.
The source-level follow-up review confirmed the SHA return-code contract and
the exact path exercised by the new fault seams. A preceding setup attempt
was stopped because the new small fixture retained a compiler dependency;
its evidence remains separate. The final fixture uses the existing labeled
unit seam and must return before invoking it. No expected result changed.

The [portable observation archive](testdata/native_source_driver/RESULTS.tar.gz)
has528 members and SHA256
`3037df713592006133db4820118d6de3a30e73f86bc43f604a009b739219faca`.
It includes source/tool/build identities, exact non-ELF runtime streams,
process exit/output hashes, resource receipts and its own file inventory.
Generated ELF outputs and build streams are indexed by hash rather than
embedded; machine-specific command paths are omitted. This is a diagnostic
receipt archive, not a signed compiler release or mathematical proof.
