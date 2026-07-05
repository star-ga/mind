# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 3035 | **Est. tokens:** ~7,276,680
**Generated:** 2026-07-05 19:12 UTC

## Token Budget Guide

| Size | Tokens | Read strategy |
|------|--------|---------------|
| tiny | <50 | Always safe to read |
| small | 50-200 | Read freely |
| medium | 200-500 | Read if relevant |
| large | 500-1500 | Use summary first, read specific sections |
| huge | >1500 | Avoid full read — use grep or read specific lines |

## Directory Overview

| Directory | Files | Est. tokens |
|-----------|-------|-------------|
| `./` | 34 | ~27,391 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~644 |
| `audits/` | 6 | ~607 |
| `bench/` | 1 | ~693 |
| `benches/` | 26 | ~79,398 |
| `bench/fft/` | 8 | ~8,060 |
| `benchmarks/` | 12 | ~20,415 |
| `benchmarks/autograd_comparison/` | 8 | ~9,411 |
| `benchmarks/cupy_comparison/` | 6 | ~7,733 |
| `benchmarks/determinism/` | 3 | ~4,601 |
| `benchmarks/inference/` | 4 | ~4,008 |
| `benchmarks/jax_comparison/` | 5 | ~4,642 |
| `benchmarks/mojo/` | 8 | ~4,300 |
| `benchmarks/pytorch_comparison/` | 5 | ~4,828 |
| `.cargo/` | 1 | ~130 |
| `config/` | 1 | ~1,163 |
| `docs/` | 31 | ~62,217 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,315 |
| `docs/design/` | 3 | ~8,181 |
| `docs/mindcraft/` | 3 | ~7,023 |
| `docs/research/` | 1 | ~117 |
| `docs/rfcs/` | 28 | ~118,403 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 19 | ~36,399 |
| `examples/c/` | 2 | ~400 |
| `examples/compliance/` | 3 | ~5,294 |
| `examples/distribution-crossisa/` | 4 | ~3,379 |
| `examples/emit_ir/` | 5 | ~13,648 |
| `examples/lexer/` | 6 | ~8,888 |
| `examples/mindc_mind/` | 40 | ~78,057 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~915 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 1 | ~83 |
| `examples/native/` | 4 | ~527 |
| `examples/parser/` | 5 | ~17,923 |
| `examples/typecheck/` | 5 | ~14,553 |
| `examples/zoo/` | 6 | ~12,518 |
| `experiments/global-vs-local/` | 7 | ~6,492 |
| `.githooks/` | 1 | ~255 |
| `.github/` | 3 | ~148 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 9 | ~10,691 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 1 | ~17,416 |
| `scripts/` | 8 | ~10,825 |
| `scripts/mind-vs-rust/` | 3 | ~933 |
| `scripts/mind-vs-rust/src/` | 1 | ~2,372 |
| `sdk/ts/mic-map/` | 6 | ~22,706 |
| `sdk/ts/mic-map/dist/` | 36 | ~29,044 |
| `sdk/ts/mic-map/node_modules/` | 1 | ~13,764 |
| `sdk/ts/mic-map/node_modules/@ampproject/remapping/` | 3 | ~5,225 |
| `sdk/ts/mic-map/node_modules/@ampproject/remapping/dist/` | 4 | ~13,315 |
| `sdk/ts/mic-map/node_modules/@ampproject/remapping/dist/types/` | 5 | ~1,201 |
| `sdk/ts/mic-map/node_modules/ansi-regex/` | 5 | ~1,469 |
| `sdk/ts/mic-map/node_modules/ansi-styles/` | 5 | ~4,374 |
| `sdk/ts/mic-map/node_modules/assertion-error/` | 5 | ~1,459 |
| `sdk/ts/mic-map/node_modules/@babel/helper-string-parser/` | 3 | ~551 |
| `sdk/ts/mic-map/node_modules/@babel/helper-string-parser/lib/` | 2 | ~7,405 |
| `sdk/ts/mic-map/node_modules/@babel/helper-validator-identifier/` | 3 | ~555 |
| `sdk/ts/mic-map/node_modules/@babel/helper-validator-identifier/lib/` | 6 | ~11,653 |
| `sdk/ts/mic-map/node_modules/@babel/parser/` | 4 | ~10,281 |
| `sdk/ts/mic-map/node_modules/@babel/parser/bin/` | 1 | ~91 |
| `sdk/ts/mic-map/node_modules/@babel/parser/typings/` | 1 | ~2,330 |
| `sdk/ts/mic-map/node_modules/@babel/types/` | 3 | ~656 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/` | 2 | ~7,645 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/asserts/` | 2 | ~328 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/asserts/generated/` | 2 | ~36,667 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/ast-types/generated/` | 1 | ~13 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/` | 4 | ~738 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/flow/` | 4 | ~1,367 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/generated/` | 5 | ~39,282 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/react/` | 2 | ~643 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/typescript/` | 2 | ~590 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/clone/` | 10 | ~4,053 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/comments/` | 14 | ~2,126 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/constants/` | 2 | ~1,854 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/constants/generated/` | 2 | ~3,756 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/converters/` | 22 | ~10,397 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/definitions/` | 19 | ~57,666 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/` | 10 | ~2,668 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/flow/` | 2 | ~1,722 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/typescript/` | 2 | ~1,870 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/retrievers/` | 8 | ~6,041 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/traverse/` | 4 | ~2,130 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/utils/` | 6 | ~1,703 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/utils/react/` | 2 | ~1,004 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/` | 36 | ~12,627 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/generated/` | 1 | ~24,081 |
| `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/react/` | 4 | ~407 |
| `sdk/ts/mic-map/node_modules/balanced-match/` | 3 | ~1,123 |
| `sdk/ts/mic-map/node_modules/balanced-match/dist/commonjs/` | 5 | ~1,665 |
| `sdk/ts/mic-map/node_modules/balanced-match/dist/esm/` | 5 | ~1,612 |
| `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/` | 9 | ~4,418 |
| `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/dist/lib/` | 29 | ~49,561 |
| `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/dist/lib/_src/` | 8 | ~6,434 |
| `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/src/lib/` | 8 | ~6,434 |
| `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/src/test/` | 1 | ~2,390 |
| `sdk/ts/mic-map/node_modules/.bin/` | 13 | ~2,453,048 |
| `sdk/ts/mic-map/node_modules/brace-expansion/` | 3 | ~1,280 |
| `sdk/ts/mic-map/node_modules/brace-expansion/dist/commonjs/` | 5 | ~5,336 |
| `sdk/ts/mic-map/node_modules/brace-expansion/dist/esm/` | 5 | ~5,292 |
| `sdk/ts/mic-map/node_modules/cac/` | 6 | ~4,928 |
| `sdk/ts/mic-map/node_modules/cac/deno/` | 6 | ~5,240 |
| `sdk/ts/mic-map/node_modules/cac/dist/` | 3 | ~10,288 |
| `sdk/ts/mic-map/node_modules/chai/` | 7 | ~2,397 |
| `sdk/ts/mic-map/node_modules/chai/lib/` | 1 | ~319 |
| `sdk/ts/mic-map/node_modules/chai/lib/chai/` | 2 | ~2,579 |
| `sdk/ts/mic-map/node_modules/chai/lib/chai/interface/` | 3 | ~24,312 |
| `sdk/ts/mic-map/node_modules/chai/lib/chai/utils/` | 25 | ~10,808 |
| `sdk/ts/mic-map/node_modules/check-error/` | 4 | ~2,656 |
| `sdk/ts/mic-map/node_modules/color-convert/` | 7 | ~6,800 |
| `sdk/ts/mic-map/node_modules/color-name/` | 4 | ~1,675 |
| `sdk/ts/mic-map/node_modules/cross-spawn/` | 4 | ~2,019 |
| `sdk/ts/mic-map/node_modules/cross-spawn/lib/` | 2 | ~1,135 |
| `sdk/ts/mic-map/node_modules/cross-spawn/lib/util/` | 3 | ~874 |
| `sdk/ts/mic-map/node_modules/debug/` | 3 | ~6,184 |
| `sdk/ts/mic-map/node_modules/debug/src/` | 4 | ~4,516 |
| `sdk/ts/mic-map/node_modules/deep-eql/` | 4 | ~5,974 |
| `sdk/ts/mic-map/node_modules/eastasianwidth/` | 3 | ~3,411 |
| `sdk/ts/mic-map/node_modules/emoji-regex/` | 9 | ~12,599 |
| `sdk/ts/mic-map/node_modules/emoji-regex/es2015/` | 6 | ~11,885 |
| `sdk/ts/mic-map/node_modules/esbuild/` | 4 | ~3,383 |
| `sdk/ts/mic-map/node_modules/esbuild/lib/` | 2 | ~27,662 |
| `sdk/ts/mic-map/node_modules/@esbuild/linux-x64/` | 2 | ~129 |
| `sdk/ts/mic-map/node_modules/es-module-lexer/` | 4 | ~9,729 |
| `sdk/ts/mic-map/node_modules/es-module-lexer/dist/` | 3 | ~12,272 |
| `sdk/ts/mic-map/node_modules/es-module-lexer/types/` | 1 | ~1,361 |
| `sdk/ts/mic-map/node_modules/estree-walker/` | 3 | ~860 |
| `sdk/ts/mic-map/node_modules/estree-walker/src/` | 4 | ~2,275 |
| `sdk/ts/mic-map/node_modules/estree-walker/types/` | 4 | ~1,269 |
| `sdk/ts/mic-map/node_modules/expect-type/` | 4 | ~11,908 |
| `sdk/ts/mic-map/node_modules/expect-type/dist/` | 10 | ~18,059 |
| `sdk/ts/mic-map/node_modules/foreground-child/` | 3 | ~1,995 |
| `sdk/ts/mic-map/node_modules/foreground-child/dist/commonjs/` | 17 | ~7,664 |
| `sdk/ts/mic-map/node_modules/foreground-child/dist/esm/` | 17 | ~7,430 |
| `sdk/ts/mic-map/node_modules/glob/` | 3 | ~12,941 |
| `sdk/ts/mic-map/node_modules/glob/dist/commonjs/` | 29 | ~50,191 |
| `sdk/ts/mic-map/node_modules/glob/dist/esm/` | 33 | ~57,500 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/balanced-match/` | 4 | ~1,723 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/balanced-match/.github/` | 1 | ~14 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/brace-expansion/` | 4 | ~2,916 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/brace-expansion/.github/` | 1 | ~14 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/` | 3 | ~5,317 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/dist/commonjs/` | 25 | ~57,174 |
| `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/dist/esm/` | 25 | ~56,603 |
| `sdk/ts/mic-map/node_modules/has-flag/` | 5 | ~1,106 |
| `sdk/ts/mic-map/node_modules/html-escaper/` | 5 | ~2,263 |
| `sdk/ts/mic-map/node_modules/html-escaper/cjs/` | 2 | ~455 |
| `sdk/ts/mic-map/node_modules/html-escaper/esm/` | 1 | ~437 |
| `sdk/ts/mic-map/node_modules/html-escaper/test/` | 2 | ~120 |
| `sdk/ts/mic-map/node_modules/@isaacs/cliui/` | 4 | ~1,563 |
| `sdk/ts/mic-map/node_modules/@isaacs/cliui/build/` | 2 | ~2,863 |
| `sdk/ts/mic-map/node_modules/@isaacs/cliui/build/lib/` | 1 | ~2,525 |
| `sdk/ts/mic-map/node_modules/isexe/` | 7 | ~1,493 |
| `sdk/ts/mic-map/node_modules/isexe/test/` | 1 | ~1,249 |
| `sdk/ts/mic-map/node_modules/is-fullwidth-code-point/` | 5 | ~1,251 |
| `sdk/ts/mic-map/node_modules/@istanbuljs/schema/` | 7 | ~4,489 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-coverage/` | 5 | ~3,346 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-coverage/lib/` | 5 | ~5,249 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-report/` | 5 | ~2,671 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-report/lib/` | 8 | ~6,745 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-source-maps/` | 5 | ~3,914 |
| `sdk/ts/mic-map/node_modules/istanbul-lib-source-maps/lib/` | 6 | ~5,521 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/` | 5 | ~5,455 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/clover/` | 1 | ~1,154 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/cobertura/` | 1 | ~1,185 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/` | 3 | ~6,960 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/assets/` | 3 | ~3,686 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/assets/vendor/` | 2 | ~4,562 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/` | 3 | ~1,463 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/assets/` | 1 | ~1,010 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/src/` | 9 | ~6,421 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/json/` | 1 | ~257 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/json-summary/` | 1 | ~330 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/lcov/` | 1 | ~228 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/lcovonly/` | 1 | ~654 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/none/` | 1 | ~68 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/teamcity/` | 1 | ~476 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text/` | 1 | ~1,973 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text-lcov/` | 1 | ~89 |
| `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text-summary/` | 1 | ~433 |
| `sdk/ts/mic-map/node_modules/jackspeak/` | 3 | ~3,789 |
| `sdk/ts/mic-map/node_modules/jackspeak/dist/commonjs/` | 9 | ~35,406 |
| `sdk/ts/mic-map/node_modules/jackspeak/dist/esm/` | 9 | ~35,068 |
| `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/` | 3 | ~2,698 |
| `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/dist/` | 4 | ~7,822 |
| `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/dist/types/` | 4 | ~1,643 |
| `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/src/` | 4 | ~5,354 |
| `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/types/` | 16 | ~5,944 |
| `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/` | 3 | ~1,493 |
| `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/dist/` | 4 | ~11,761 |
| `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/dist/types/` | 1 | ~38 |
| `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/` | 3 | ~3,321 |
| `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/dist/` | 4 | ~11,694 |
| `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/src/` | 4 | ~3,953 |
| `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/types/` | 16 | ~2,796 |
| `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/` | 3 | ~4,438 |
| `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/dist/` | 4 | ~14,197 |
| `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/src/` | 9 | ~7,779 |
| `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/types/` | 36 | ~10,286 |
| `sdk/ts/mic-map/node_modules/loupe/` | 4 | ~6,149 |
| `sdk/ts/mic-map/node_modules/loupe/lib/` | 60 | ~8,524 |
| `sdk/ts/mic-map/node_modules/lru-cache/` | 3 | ~3,678 |
| `sdk/ts/mic-map/node_modules/lru-cache/dist/commonjs/` | 5 | ~35,154 |
| `sdk/ts/mic-map/node_modules/lru-cache/dist/esm/` | 5 | ~35,099 |
| `sdk/ts/mic-map/node_modules/magicast/` | 4 | ~2,430 |
| `sdk/ts/mic-map/node_modules/magicast/dist/` | 8 | ~6,167 |
| `sdk/ts/mic-map/node_modules/magicast/dist/shared/` | 3 | ~6,936 |
| `sdk/ts/mic-map/node_modules/magic-string/` | 3 | ~3,875 |
| `sdk/ts/mic-map/node_modules/magic-string/dist/` | 7 | ~82,920 |
| `sdk/ts/mic-map/node_modules/make-dir/` | 5 | ~2,481 |
| `sdk/ts/mic-map/node_modules/minimatch/` | 3 | ~5,808 |
| `sdk/ts/mic-map/node_modules/minimatch/dist/commonjs/` | 25 | ~64,385 |
| `sdk/ts/mic-map/node_modules/minimatch/dist/esm/` | 25 | ~63,855 |
| `sdk/ts/mic-map/node_modules/minipass/` | 3 | ~7,654 |
| `sdk/ts/mic-map/node_modules/minipass/dist/commonjs/` | 5 | ~42,121 |
| `sdk/ts/mic-map/node_modules/minipass/dist/esm/` | 5 | ~41,927 |
| `sdk/ts/mic-map/node_modules/ms/` | 4 | ~1,681 |
| `sdk/ts/mic-map/node_modules/nanoid/` | 10 | ~4,585 |
| `sdk/ts/mic-map/node_modules/nanoid/async/` | 7 | ~2,524 |
| `sdk/ts/mic-map/node_modules/nanoid/bin/` | 1 | ~283 |
| `sdk/ts/mic-map/node_modules/nanoid/non-secure/` | 4 | ~675 |
| `sdk/ts/mic-map/node_modules/nanoid/url-alphabet/` | 3 | ~123 |
| `sdk/ts/mic-map/node_modules/package-json-from-dist/` | 3 | ~1,645 |
| `sdk/ts/mic-map/node_modules/package-json-from-dist/dist/commonjs/` | 5 | ~3,790 |
| `sdk/ts/mic-map/node_modules/package-json-from-dist/dist/esm/` | 5 | ~3,683 |
| `sdk/ts/mic-map/node_modules/pathe/` | 4 | ~1,600 |
| `sdk/ts/mic-map/node_modules/pathe/dist/` | 10 | ~2,779 |
| `sdk/ts/mic-map/node_modules/pathe/dist/shared/` | 2 | ~3,330 |
| `sdk/ts/mic-map/node_modules/path-key/` | 5 | ~1,140 |
| `sdk/ts/mic-map/node_modules/path-scurry/` | 3 | ~6,436 |
| `sdk/ts/mic-map/node_modules/path-scurry/dist/commonjs/` | 4 | ~31,030 |
| `sdk/ts/mic-map/node_modules/path-scurry/dist/esm/` | 4 | ~30,597 |
| `sdk/ts/mic-map/node_modules/pathval/` | 4 | ~3,586 |
| `sdk/ts/mic-map/node_modules/picocolors/` | 7 | ~1,596 |
| `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/` | 7 | ~13,086 |
| `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/examples/` | 6 | ~1,498 |
| `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/internal/` | 4 | ~3,965 |
| `sdk/ts/mic-map/node_modules/postcss/` | 3 | ~1,188 |
| `sdk/ts/mic-map/node_modules/postcss/lib/` | 52 | ~50,155 |
| `sdk/ts/mic-map/node_modules/rollup/` | 3 | ~14,415 |
| `sdk/ts/mic-map/node_modules/rollup/dist/` | 9 | ~12,447 |
| `sdk/ts/mic-map/node_modules/rollup/dist/bin/` | 1 | ~20,568 |
| `sdk/ts/mic-map/node_modules/rollup/dist/es/` | 4 | ~702 |
| `sdk/ts/mic-map/node_modules/rollup/dist/es/shared/` | 1 | ~21,538 |
| `sdk/ts/mic-map/node_modules/rollup/dist/shared/` | 5 | ~37,026 |
| `sdk/ts/mic-map/node_modules/rollup/node_modules/@types/estree/` | 5 | ~6,546 |
| `sdk/ts/mic-map/node_modules/@rollup/rollup-linux-x64-gnu/` | 2 | ~144 |
| `sdk/ts/mic-map/node_modules/semver/` | 6 | ~7,896 |
| `sdk/ts/mic-map/node_modules/semver/bin/` | 1 | ~1,240 |
| `sdk/ts/mic-map/node_modules/semver/classes/` | 4 | ~7,059 |
| `sdk/ts/mic-map/node_modules/semver/functions/` | 25 | ~2,328 |
| `sdk/ts/mic-map/node_modules/semver/internal/` | 6 | ~2,732 |
| `sdk/ts/mic-map/node_modules/semver/ranges/` | 11 | ~3,771 |
| `sdk/ts/mic-map/node_modules/shebang-command/` | 4 | ~640 |
| `sdk/ts/mic-map/node_modules/shebang-regex/` | 5 | ~710 |
| `sdk/ts/mic-map/node_modules/siginfo/` | 6 | ~1,199 |
| `sdk/ts/mic-map/node_modules/signal-exit/` | 3 | ~1,449 |
| `sdk/ts/mic-map/node_modules/signal-exit/dist/cjs/` | 13 | ~8,980 |
| `sdk/ts/mic-map/node_modules/signal-exit/dist/mjs/` | 13 | ~8,822 |
| `sdk/ts/mic-map/node_modules/source-map-js/` | 5 | ~8,483 |
| `sdk/ts/mic-map/node_modules/source-map-js/lib/` | 13 | ~26,491 |
| `sdk/ts/mic-map/node_modules/stackback/` | 7 | ~1,724 |
| `sdk/ts/mic-map/node_modules/std-env/` | 3 | ~1,280 |
| `sdk/ts/mic-map/node_modules/std-env/dist/` | 5 | ~5,237 |
| `sdk/ts/mic-map/node_modules/string-width/` | 5 | ~1,447 |
| `sdk/ts/mic-map/node_modules/string-width-cjs/` | 5 | ~1,292 |
| `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/ansi-regex/` | 5 | ~1,405 |
| `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/emoji-regex/` | 6 | ~6,514 |
| `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/emoji-regex/es2015/` | 2 | ~5,553 |
| `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/strip-ansi/` | 5 | ~1,010 |
| `sdk/ts/mic-map/node_modules/strip-ansi/` | 5 | ~1,072 |
| `sdk/ts/mic-map/node_modules/strip-ansi-cjs/` | 5 | ~1,010 |
| `sdk/ts/mic-map/node_modules/strip-ansi-cjs/node_modules/ansi-regex/` | 5 | ~1,405 |
| `sdk/ts/mic-map/node_modules/supports-color/` | 5 | ~1,761 |
| `sdk/ts/mic-map/node_modules/test-exclude/` | 7 | ~2,867 |
| `sdk/ts/mic-map/node_modules/tinybench/` | 3 | ~3,754 |
| `sdk/ts/mic-map/node_modules/tinybench/dist/` | 4 | ~11,986 |
| `sdk/ts/mic-map/node_modules/tinyexec/` | 3 | ~2,067 |
| `sdk/ts/mic-map/node_modules/tinyexec/dist/` | 4 | ~9,459 |
| `sdk/ts/mic-map/node_modules/tinypool/` | 3 | ~898 |
| `sdk/ts/mic-map/node_modules/tinypool/dist/` | 5 | ~8,853 |
| `sdk/ts/mic-map/node_modules/tinypool/dist/entry/` | 6 | ~1,292 |
| `sdk/ts/mic-map/node_modules/tinyrainbow/` | 3 | ~603 |
| `sdk/ts/mic-map/node_modules/tinyrainbow/dist/` | 6 | ~1,377 |
| `sdk/ts/mic-map/node_modules/tinyspy/` | 3 | ~618 |
| `sdk/ts/mic-map/node_modules/tinyspy/dist/` | 4 | ~3,784 |
| `sdk/ts/mic-map/node_modules/typescript/` | 5 | ~14,036 |
| `sdk/ts/mic-map/node_modules/typescript/bin/` | 2 | ~25 |
| `sdk/ts/mic-map/node_modules/typescript/lib/` | 106 | ~124,785 |
| `sdk/ts/mic-map/node_modules/@types/estree/` | 5 | ~6,541 |
| `sdk/ts/mic-map/node_modules/@types/node/` | 47 | ~358,563 |
| `sdk/ts/mic-map/node_modules/@types/node/assert/` | 1 | ~751 |
| `sdk/ts/mic-map/node_modules/@types/node/compatibility/` | 4 | ~758 |
| `sdk/ts/mic-map/node_modules/@types/node/dns/` | 1 | ~5,275 |
| `sdk/ts/mic-map/node_modules/@types/node/fs/` | 1 | ~13,924 |
| `sdk/ts/mic-map/node_modules/@types/node/readline/` | 1 | ~1,610 |
| `sdk/ts/mic-map/node_modules/@types/node/stream/` | 3 | ~8,635 |
| `sdk/ts/mic-map/node_modules/@types/node/timers/` | 1 | ~945 |
| `sdk/ts/mic-map/node_modules/@types/node/ts5.6/` | 3 | ~6,899 |
| `sdk/ts/mic-map/node_modules/@types/node/web-globals/` | 6 | ~2,706 |
| `sdk/ts/mic-map/node_modules/undici-types/` | 41 | ~20,935 |
| `sdk/ts/mic-map/node_modules/vite/` | 5 | ~3,157 |
| `sdk/ts/mic-map/node_modules/vite/bin/` | 2 | ~1,091 |
| `sdk/ts/mic-map/node_modules/vite/dist/client/` | 2 | ~6,107 |
| `sdk/ts/mic-map/node_modules/vite/dist/node/` | 6 | ~23,946 |
| `sdk/ts/mic-map/node_modules/vite/dist/node/chunks/` | 2 | ~9,156 |
| `sdk/ts/mic-map/node_modules/vite-node/` | 4 | ~2,177 |
| `sdk/ts/mic-map/node_modules/vite-node/dist/` | 33 | ~48,912 |
| `sdk/ts/mic-map/node_modules/vitest/` | 24 | ~21,679 |
| `sdk/ts/mic-map/node_modules/@vitest/coverage-v8/` | 2 | ~801 |
| `sdk/ts/mic-map/node_modules/@vitest/coverage-v8/dist/` | 6 | ~1,131 |
| `sdk/ts/mic-map/node_modules/vitest/dist/` | 33 | ~28,304 |
| `sdk/ts/mic-map/node_modules/vitest/dist/chunks/` | 38 | ~88,069 |
| `sdk/ts/mic-map/node_modules/vitest/dist/workers/` | 5 | ~1,844 |
| `sdk/ts/mic-map/node_modules/@vitest/expect/` | 4 | ~678 |
| `sdk/ts/mic-map/node_modules/@vitest/expect/dist/` | 3 | ~40,696 |
| `sdk/ts/mic-map/node_modules/@vitest/mocker/` | 3 | ~801 |
| `sdk/ts/mic-map/node_modules/@vitest/mocker/dist/` | 19 | ~29,363 |
| `sdk/ts/mic-map/node_modules/@vitest/pretty-format/` | 2 | ~528 |
| `sdk/ts/mic-map/node_modules/@vitest/pretty-format/dist/` | 2 | ~10,848 |
| `sdk/ts/mic-map/node_modules/@vitest/runner/` | 5 | ~607 |
| `sdk/ts/mic-map/node_modules/@vitest/runner/dist/` | 8 | ~21,714 |
| `sdk/ts/mic-map/node_modules/@vitest/snapshot/` | 5 | ~1,263 |
| `sdk/ts/mic-map/node_modules/@vitest/snapshot/dist/` | 8 | ~20,059 |
| `sdk/ts/mic-map/node_modules/@vitest/spy/` | 3 | ~513 |
| `sdk/ts/mic-map/node_modules/@vitest/spy/dist/` | 2 | ~4,924 |
| `sdk/ts/mic-map/node_modules/@vitest/utils/` | 5 | ~740 |
| `sdk/ts/mic-map/node_modules/@vitest/utils/dist/` | 14 | ~38,505 |
| `sdk/ts/mic-map/node_modules/vite/types/` | 8 | ~1,559 |
| `sdk/ts/mic-map/node_modules/.vite/vitest/` | 1 | ~76 |
| `sdk/ts/mic-map/node_modules/which/` | 5 | ~2,249 |
| `sdk/ts/mic-map/node_modules/which/bin/` | 1 | ~247 |
| `sdk/ts/mic-map/node_modules/why-is-node-running/` | 7 | ~1,823 |
| `sdk/ts/mic-map/node_modules/why-is-node-running/.github/` | 1 | ~5 |
| `sdk/ts/mic-map/node_modules/wrap-ansi/` | 5 | ~2,946 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/` | 4 | ~2,664 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/ansi-regex/` | 5 | ~1,405 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/ansi-styles/` | 5 | ~4,247 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/emoji-regex/` | 6 | ~6,514 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/emoji-regex/es2015/` | 2 | ~5,553 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/string-width/` | 5 | ~1,292 |
| `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/strip-ansi/` | 5 | ~1,010 |
| `sdk/ts/mic-map/scripts/` | 1 | ~499 |
| `sdk/ts/mic-map/src/` | 9 | ~12,181 |
| `sdk/ts/mic-map/test/` | 4 | ~7,843 |
| `sdk/ts/mic-map/test/fixtures/` | 2 | ~96 |
| `skills/write-mind/` | 1 | ~6,002 |
| `src/` | 7 | ~17,449 |
| `src/ast/` | 1 | ~7,430 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 2 | ~25,459 |
| `src/build/` | 2 | ~11,359 |
| `src/cache/` | 4 | ~3,525 |
| `src/check/` | 3 | ~9,753 |
| `src/deps/` | 1 | ~8,217 |
| `src/diagnostics/` | 1 | ~2,230 |
| `src/distributed/` | 6 | ~7,433 |
| `src/doc/` | 3 | ~10,474 |
| `src/eval/` | 12 | ~61,758 |
| `src/eval/stdlib/` | 2 | ~8,529 |
| `src/exec/` | 3 | ~4,592 |
| `src/ffi/` | 3 | ~3,919 |
| `src/fmt/` | 3 | ~19,662 |
| `src/ir/` | 5 | ~32,357 |
| `src/ir/compact/` | 3 | ~15,194 |
| `src/ir/compact/v2/` | 8 | ~37,885 |
| `src/ir/compact/v3/` | 4 | ~45,526 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~9,211 |
| `src/mlir/` | 3 | ~5,415 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 4 | ~9,649 |
| `src/package/` | 2 | ~1,877 |
| `src/parser/` | 1 | ~3,811 |
| `src/project/` | 3 | ~28,061 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 1 | ~5,999 |
| `src/type_checker/` | 1 | ~11,258 |
| `src/types/` | 4 | ~3,336 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 40 | ~186,313 |
| `tests/` | 272 | ~459,464 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 1 | ~668 |
| `tests/conformance/cpu_baseline/` | 9 | ~171 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4,052 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~648 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~363 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~813 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~707 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~616 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~896 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~594 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~519 |
| `tests/cross_substrate_identity/lorenz-q16/` | 2 | ~1,243 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~788 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1,310 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~746 |
| `tests/fixtures/` | 6 | ~228 |
| `tests/ir_verification/` | 2 | ~108 |
| `tests/lexical/` | 3 | ~191 |
| `tests/mindcraft/` | 1 | ~408 |
| `tests/mindcraft/check/` | 4 | ~48 |
| `tests/mindcraft/check/subdir/` | 1 | ~9 |
| `tests/mindcraft/fmt/` | 14 | ~474 |
| `tests/mindcraft/lint/` | 2 | ~21 |
| `tests/mindcraft/lint/naming_convention/` | 4 | ~176 |
| `tests/mindcraft/lint/q16_overflow/` | 3 | ~191 |
| `tests/mindcraft/lint/shadowing/` | 2 | ~87 |
| `tests/mindcraft/lint/unused_import/` | 2 | ~99 |
| `tests/mindfuzz_cross_substrate/staged/` | 15 | ~2,832 |
| `tests/runtime/` | 2 | ~135 |
| `tests/selfhost_gaps/` | 67 | ~5,301 |
| `tests/shapes/` | 3 | ~260 |
| `tests/type_checker/` | 2 | ~140 |
| `tools/` | 4 | ~4,578 |
| `tools/mindfuzz/` | 7 | ~15,992 |
| `tools/mindfuzz/seeds/` | 6 | ~1,330 |
| `tools/mindfuzz/violations/` | 1 | ~0 |
| `tools/pytorch_bridge/` | 6 | ~4,673 |
| `tools/pytorch_bridge/tests/` | 2 | ~1,244 |
| `.wrangler/cache/` | 1 | ~21 |

## Files

### `./`

- `a.out` (~3774 tok, huge) — ELF>@x4@8
- `ARCHITECTURE.md` (~300 tok, medium) — MIND Architecture (high level)
- `AUDIT_REPORT.md` (~1151 tok, large) — Audit Report
- `.bench-baseline-2026-04-27.txt` (~531 tok, large) —    Compiling mind v0.2.3 (.)
- `.bench-baseline-2026-04-28-pratt.txt` (~185 tok, small) — === Pratt parser baseline (mindc 0.2.5, 2026-04-28) ===
- `.bench-baseline-2026-05-17-phase10-6.txt` (~408 tok, medium) — === Phase 10.6 surface-syntax baseline (mindc 0.2.10, 2026-05-17) ===
- `.bench-baseline-2026-05-17-phase10-7.txt` (~565 tok, large) — === Phase 10.7 surface baseline (mindc 0.2.11, 2026-05-17) ===
- `.bench-baseline-2026-05-18-rfc0005.txt` (~781 tok, large) — === RFC 0005 Phase 2 baseline (mindc 0.4.0, 2026-05-18) ===
- `.bench-baseline-2026-06-01-correctness.txt` (~784 tok, large) — === Correctness-milestone baseline (mindc 0.7.0, 2026-06-01) ===
- `.bench-pre-pratt.txt` (~32 tok, tiny) — === captured pre-Pratt baseline (Phase 10.5 in main) ===
- `bounties.md` (~888 tok, large) — MIND Bounty Board
- `build.rs` (~234 tok, medium) — Copyright 2025 STARGA Inc.
- `Cargo.toml` (~1633 tok, huge) — [package]
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~1348 tok, large) — Contributing to MIND
- `deny.toml` (~89 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~130 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~502 tok, large) — # Rust
- `incompatible` (~0 tok, tiny)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `Mind.toml` (~106 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~5621 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~614 tok, large) — Security Policy
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `STATUS.md` (~3819 tok, huge) — MIND Compiler Status
- `test_real_compile_time.py` (~265 tok, medium) — Quick test of real MIND compilation time using Python bindings."""
### `agents/`

- `mind-developer.md` (~436 tok, medium) — MIND Developer Agent
### `.agents/skills/mindc-development/`

- `SKILL.md` (~235 tok, medium) — MIND Compiler (mindc) Development
### `.arch-mind/`

- `rules.mind` (~557 tok, large) — mind (language compiler / runtime root) architectural-governance rules
- `scan.json` (~87 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `audits/`

- `arch-mind-2026-05-18-post-phase-6-1.json` (~169 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.0.json` (~86 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.1.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.2.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.3.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.4.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `benches/`

- `autodiff.rs` (~1661 tok, huge) — Simple linear function
- `bench_aes_gcm.rs` (~2590 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_ecdsa_p256.rs` (~2786 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hkdf.rs` (~4423 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hpack.rs` (~3926 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_http2_frame.rs` (~5035 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_keccak.rs` (~2576 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_mlkem768.rs` (~3700 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_rsa_pss.rs` (~3317 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_sha256.rs` (~2594 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_tls13_record.rs` (~6120 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x25519.rs` (~2468 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x509.rs` (~3990 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `cerebras_stencil.rs` (~831 tok, large) — Copyright 2025-2026 STARGA Inc.
- `compiler.rs` (~3782 tok, huge) — Small program: Simple matrix multiplication
- `cross_module.rs` (~609 tok, large) — Copyright 2025 STARGA Inc.
- `det_matmul_i16.rs` (~4621 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i8.rs` (~5094 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16_mt.rs` (~4049 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16.rs` (~4972 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `fft_q16.rs` (~5352 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindcraft_fmt.rs` (~908 tok, large) — File readers
- `operations.rs` (~1076 tok, large) — Element-wise operations
- `shapes.rs` (~1208 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~643 tok, large) — Benchmark source programs that are known to work
- `std_surface.rs` (~1067 tok, large) — Copyright 2025 STARGA Inc.
### `bench/fft/`

- `build.sh` (~986 tok, large) — build.sh — self-contained build for the deterministic Q16.16 N=256 FFT bench.
- `fft_driver.c` (~1205 tok, large) — Standalone correctness + timing driver for the C reference Q16.16 FFT.
- `fft_ref.c` (~473 tok, medium) — Q16.16 deterministic radix-2 DIT FFT, N=256 — BYTE-IDENTICAL algorithm to
- `fft_verify.c` (~889 tok, large) — Cross-check harness: load the MIND-compiled fft256 from a .so and assert its
- `.gitignore` (~38 tok, tiny) — # Build artifacts — regenerated by build.sh, never committed.
- `harness.c` (~1485 tok, large) — Self-contained benchmark harness for the deterministic Q16.16 N=256 FFT.
- `README.md` (~1677 tok, huge) — Deterministic Q16.16 N=256 FFT — MIND vs gcc / clang / nvcc
- `RESULTS-fft-2026-06-15.md` (~1307 tok, large) — RESULTS — Deterministic Q16.16 N=256 FFT (MIND vs gcc / clang / nvcc)
### `benchmarks/autograd_comparison/`

- `autograd_results.json` (~424 tok, medium) — Keys: system_info, benchmarks
- `benchmark_autograd.py` (~2444 tok, huge)
- `benchmark_python_bindings.py` (~1566 tok, huge)
- `benchmark_real_autograd.py` (~2304 tok, huge)
- `README.md` (~1153 tok, large) — Autograd Comparison: MIND vs PyTorch
- `README_REAL.md` (~1185 tok, large) — Real Autograd Comparison: MIND vs PyTorch
- `real_autograd_results.json` (~328 tok, medium) — Keys: system_info, methodology, benchmarks
- `requirements.txt` (~7 tok, tiny) — torch>=1.0.0
### `benchmarks/`

- `BENCHMARK_RESULTS.md` (~4311 tok, huge) — MIND Benchmark Results
### `benchmarks/cupy_comparison/`

- `leg1_determinism.py` (~2586 tok, huge)
- `leg1_determinism_results.json` (~1451 tok, large) — Keys: leg, host, mind, cupy
- `leg2_perf.py` (~1510 tok, huge)
- `leg2_perf_results.json` (~473 tok, medium) — Keys: leg, host, config, mind, status
- `README.md` (~1619 tok, huge) — CuPy Comparison Benchmark
- `requirements.txt` (~94 tok, small) — # Leg 1 (determinism) + Leg 2 (perf) foil dependencies.
### `benchmarks/determinism/`

- `benchmark_determinism.py` (~2187 tok, huge)
- `determinism_results.json` (~1103 tok, large) — Keys: system_info, num_runs, tests, all_deterministic
- `README.md` (~1311 tok, large) — MIND Determinism Proof Benchmark
### `benchmarks/`

- `format_benchmark.py` (~2617 tok, huge)
### `benchmarks/inference/`

- `benchmark_inference.py` (~2423 tok, huge)
- `inference_results.json` (~473 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1108 tok, large) — Inference Speed Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=1.0.0
### `benchmarks/jax_comparison/`

- `benchmark_jax_compile.py` (~2719 tok, huge)
- `jax_coldstart_results.json` (~376 tok, medium) — Keys: environment, results
- `jax_results.json` (~478 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1062 tok, large) — JAX Compilation Benchmark
- `requirements.txt` (~7 tok, tiny) — jax>=0.4.0
### `benchmarks/`

- `mic_benchmark.py` (~1473 tok, large)
- `MIC_MAP_BENCHMARK_README.md` (~337 tok, medium) — MIC/MAP Patent Reference Benchmark
- `mic_map_benchmark_results.json` (~851 tok, large) — Keys: metadata, measurements, paper_figures_verified, claim_checks, all_claims_verified
- `mic_map_benchmark_v2.py` (~3151 tok, huge)
### `benchmarks/mojo/`

- `benchmark_mojo_compilation.py` (~1533 tok, huge)
- `large_matmul.mojo` (~205 tok, medium) — """
- `medium_matmul.mojo` (~205 tok, medium) — """
- `mojo_results.json` (~216 tok, medium) — Keys: scalar_math, small_matmul, medium_matmul, large_matmul
- `README.md` (~1295 tok, large) — Mojo Compilation Benchmarks
- `run_benchmarks.sh` (~581 tok, large) — Mojo Compilation Benchmark Runner
- `scalar_math.mojo` (~58 tok, small) — """
- `small_matmul.mojo` (~207 tok, medium) — """
### `benchmarks/pytorch_comparison/`

- `=2.0` (~0 tok, tiny)
- `benchmark_pytorch_compile.py` (~3420 tok, huge)
- `pytorch_results.json` (~590 tok, large) — Keys: system_info, benchmarks
- `README.md` (~814 tok, large) — PyTorch Compilation Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=2.0.0
### `benchmarks/`

- `README.md` (~1188 tok, large) — MIND Performance Benchmarks
- `resnet.md` (~74 tok, small) — ResNet Benchmarks (Preliminary)
- `run_all_benchmarks.sh` (~824 tok, large) — Master script to run all MIND patent benchmarks
- `RUN_GUIDE.md` (~1465 tok, large) — MIND Patent Benchmarks - Environment Guide
- `scientific_benchmark.py` (~1639 tok, huge)
- `scientific_benchmark_raw.py` (~2485 tok, huge)
### `bench/`

- `RESULTS-int8-2026-06-08.md` (~693 tok, large) — MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `config/`

- `capabilities.toml` (~1163 tok, large) — [ir]
### `docs/`

- `architecture.md` (~965 tok, large) — Architecture
- `autodiff.md` (~595 tok, large) — Static autodiff (public)
### `docs/backends/`

- `cerebras-stencil.md` (~1482 tok, large) — `mind.cerebras.stencil_tile` — Op Surface and Lowering Contract
### `docs/`

- `benchmarking.md` (~1917 tok, huge) — Benchmarking methodology — tiers and comparable metrics
### `docs/benchmarks/`

- `compiler_performance.md` (~4721 tok, huge) — MIND Compiler Performance Benchmarks
### `docs/`

- `benchmarks.md` (~896 tok, large) — Benchmarks
### `docs/benchmarks/`

- `mojo_comparison.md` (~2420 tok, huge) — MIND vs Mojo: Compilation Performance Comparison
- `RESULTS-mind-vs-rust-2026-06-09.md` (~2174 tok, huge) — MIND vs Rust — integer-GEMM, apples-to-apples (2026-06-09)
### `docs/`

- `byte-store-migration.md` (~3357 tok, huge) — Byte-Store Migration — closing `#306`
- `cli.md` (~627 tok, large) — MIND CLI Reference
### `docs/design/`

- `execution-plan-performance-mode.md` (~8045 tok, huge) — Design: PerformanceMode + ExecutionPlan + ExecutionProvider
- `README.md` (~26 tok, tiny) — Design Docs
- `v0.3.md` (~110 tok, small) — MIND Design v0.3 (Draft)
### `docs/`

- `determinism.md` (~2784 tok, huge) — The Determinism Contract
- `errors.md` (~701 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
- `gpu.md` (~387 tok, medium) — GPU backend profile
- `INDEPENDENCE_ROADMAP.md` (~2329 tok, huge) — MIND Rust-Independence Roadmap
- `install.md` (~1012 tok, large) — Installing mindc
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1485 tok, large) — IR stability contract
### `docs/mindcraft/`

- `fmt.md` (~2239 tok, huge) — `mindc fmt` — Canonical Formatter Reference
- `phase2-implementation-plan.md` (~2209 tok, huge) — Mindcraft Phase 2 — Implementation Plan
- `rfc0010-phase-ghi-migration-plan.md` (~2575 tok, huge) — RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)
### `docs/`

- `mlir-lowering.md` (~286 tok, medium) — MLIR lowering pipeline (public)
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `optimization-frontier.md` (~11347 tok, huge) — MIND Optimization Frontier
- `performance.md` (~880 tok, large) — Performance Guide
- `README.md` (~162 tok, small) — MIND Documentation
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
### `docs/research/`

- `README.md` (~117 tok, small) — Research notes
### `docs/rfcs/`

- `0000-template.md` (~627 tok, large) — RFC 0000: [Title]
- `0001-bitnet-native-support.md` (~3254 tok, huge) — RFC 0001: Native BitNet Support — `tri` and `q16_16` Types
- `0002-pub-fn-c-exports.md` (~2084 tok, huge) — RFC 0002: `pub fn` → C ABI Symbol Export
- `0003-cdylib-aot-emit.md` (~3195 tok, huge) — RFC 0003: cdylib AOT emit + symbol versioning
- `0004-evidence-token-types.md` (~1913 tok, huge) — RFC 0004: Compile-Time Evidence Token Types
- `0005-phase-6-2-mindc-gaps.md` (~3356 tok, huge) — RFC 0005 Phase 6.2 — mindc Feature Gaps (Design Note)
- `0005-phase-d2b-design-note.md` (~1518 tok, huge) — RFC 0005 Phase D₂b — Cross-arg Named-struct identity matching
- `0005-pure-mind-std-surface.md` (~5516 tok, huge) — RFC 0005: Pure-MIND Standard Surface
- `0006-mind-blas.md` (~5743 tok, huge) — RFC 0006: mind-blas — native BLAS surface for MIND
- `0007-mindcraft.md` (~3497 tok, huge) — RFC 0007: Mindcraft — the pure-MIND format / lint / check toolchain
- `0008-mindc-build.md` (~10964 tok, huge) — RFC 0008: mindc build + mindc test — retiring cargo from the build path
- `0009-federation-package-layer.md` (~6976 tok, huge) — RFC 0009: Federation-First MIND Package Layer
- `000-template.md` (~1 tok, tiny)
- `0010-memory-safety-and-c-abi.md` (~7359 tok, huge) — RFC 0010: Memory Safety Model + C ABI in Pure MIND
- `0011-async-and-structured-concurrency.md` (~4891 tok, huge) — RFC 0011: Async + Structured Concurrency Model
- `0012-tensor-native-syntax.md` (~11307 tok, huge) — RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer
- `0013-cli-agent-harness-stack.md` (~6781 tok, huge) — RFC 0013: CLI Agent Harness Stack
- `0014-per-substrate-mlir-lowering-contracts.md` (~5412 tok, huge) — RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts
- `0015-cross-substrate-bit-identity.md` (~5174 tok, huge) — RFC 0015: Cross-Substrate Bit-Identity Proof Obligation
- `0016-evidence-chain-emission.md` (~6226 tok, huge) — RFC 0016: Compile-Time Evidence-Chain Emission
- `0017-mindc-verify.md` (~3745 tok, huge) — RFC 0017: `mindc verify` — Artifact Verification Surface
- `0018-bare-metal-substrate.md` (~3799 tok, huge) — RFC 0018: Bare-Metal Substrate Lowering Tier
- `0019-deterministic-agent-substrate.md` (~4131 tok, huge) — RFC 0019: Deterministic Agent Substrate
- `0020-mind-bench-reproducibility-harness.md` (~4083 tok, huge) — RFC 0020: mind-bench Public Reproducibility Harness
- `0021-canonical-ir-unification.md` (~4290 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2108 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `odc-language-primitives.md` (~422 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

- `roadmap.md` (~15733 tok, huge) — Roadmap
- `runs-burndown-roadmap.md` (~3203 tok, huge) — MIND RUNS Burndown Roadmap
- `security.md` (~1492 tok, large) — Security Guide
- `self-host-trace-hash-port.md` (~1406 tok, large) — #17 — Self-compute the native PT_NOTE (pure-MIND trace-hash port)
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
- `sparse-tensor-types.md` (~740 tok, large) — Sparse Tensor Types
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~953 tok, large) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `type-system.md` (~1082 tok, large) — Type System
- `versioning.md` (~804 tok, large) — MIND Core Stability & Versioning
- `version-matrix.md` (~1796 tok, huge) — MIND Ecosystem — Version Matrix
- `whitepaper.md` (~2788 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `autodiff_demo.mind` (~1715 tok, huge) — Autodiff Demonstration
### `examples/c/`

- `min.c` (~82 tok, small)
- `mind.h` (~318 tok, medium) — Copyright 2025 STARGA Inc.
### `examples/`

- `cnn_classifier.mind` (~1060 tok, large) — CNN Classifier Example
### `examples/compliance/`

- `auditable_model.mind` (~1932 tok, huge) — auditable_model.mind -- Compliance-Ready MLP with Provenance Metadata
- `audit_report.mind` (~2289 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `README.md` (~1073 tok, large) — Compliance Example
### `examples/distribution-crossisa/`

- `data1.txt` (~212 tok, medium) — 45.96
- `distribution.cpp` (~1217 tok, large)
- `distribution_interp_f64.mind` (~1127 tok, large) — Deterministic IEEE-754 float64 piecewise-LINEAR density interpolation kernel,
- `README.md` (~823 tok, large) — Cross-ISA determinism: a piecewise-linear density kernel
### `examples/`

- `distribution_interp_f64.mind` (~1127 tok, large) — Deterministic IEEE-754 float64 piecewise-LINEAR density interpolation kernel,
### `examples/emit_ir/`

- `bootstrap_smoke.py` (~2890 tok, huge)
- `EXPECTED.md` (~1942 tok, huge) — Phase 6.4 — Expected IR Text
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `main.mind` (~6419 tok, huge) — examples/emit_ir/main.mind — RFC 0005 Phase 6.4 self-host MLIR text emitter.
- `README.md` (~2214 tok, huge) — RFC 0005 Phase 6.4 — Self-Host MLIR Text Emitter
### `examples/`

- `fft_q16.mind` (~1248 tok, large) — Deterministic Q16.16 fixed-point radix-2 DIT FFT, N=256 (complex).
- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
- `hello_stdlib.mind` (~271 tok, medium) — Hello, std.vec — minimal RFC 0005 cookbook example.
- `hello_tensor.mind` (~141 tok, small) — Hello, MIND — scalar smoke that flows through every stage of the
### `examples/lexer/`

- `bootstrap_smoke.py` (~2367 tok, huge)
- `BOOTSTRAP_SMOKE_REPORT.md` (~1931 tok, huge) — Phase 6.5 Stage 1 — Bootstrap Smoke Report
- `EXPECTED.md` (~1093 tok, large) — Phase 6.1 — Expected Token Stream
- `fixture.mind` (~67 tok, small) — Phase 6.1 lexer smoke fixture.
- `main.mind` (~2461 tok, huge) — examples/lexer/main.mind — RFC 0005 Phase 6.1 self-host smoke
- `README.md` (~969 tok, large) — RFC 0005 Phase 6.1 — Self-Host Lexer Seed
### `examples/`

- `lorenz_f64.mind` (~230 tok, medium) — Deterministic IEEE-754 float64 Lorenz-attractor integrator (forward Euler).
- `lorenz_q16.mind` (~1091 tok, large) — Deterministic Q16.16 fixed-point Lorenz-attractor integrator (forward Euler).
### `examples/mindc_mind/`

- `bootstrap_smoke.py` (~2329 tok, huge)
- `collect_field_strings_smoke.py` (~1161 tok, large)
- `cutover_coverage_measure.py` (~2238 tok, huge)
- `_dump_pure_all.py` (~668 tok, large) — Dump pure-MIND nb_build_mic3 bytes + note for all native_elf fixtures + main,
- `EXPECTED.md` (~773 tok, large) — Phase 6.5 Stage 5 — Expected IR Text (APEX)
- `fast_keystone.sh` (~922 tok, large) — fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
- `FIXED_POINT_REPORT.md` (~1770 tok, huge) — Phase 6.5 — Bootstrap Fixed-Point Report
- `fixed_point_smoke.py` (~3275 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `full_strtab_smoke.py` (~1663 tok, huge)
- `gap_corpus_smoke.py` (~1865 tok, huge)
- `.gitignore` (~5 tok, tiny) — __pycache__/
- `match_struct_smoke.py` (~1311 tok, large)
- `method_callee_smoke.py` (~1350 tok, large)
- `method_calls_smoke.py` (~1294 tok, large)
- `mic3_flip_smoke.py` (~1093 tok, large)
- `mic3_oracle_smoke.py` (~764 tok, large) — mic@3 self-host convergence — Phase 0 gate: the Rust oracle.
- `mic3_primitives_smoke.py` (~22299 tok, huge) — mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.
- `mod_operator_smoke.py` (~2100 tok, huge)
- `multi_let_smoke.py` (~1499 tok, large)
- `now_ns_smoke.py` (~678 tok, large) — # Copyright 2025 STARGA Inc.
- `param_types_smoke.py` (~1273 tok, large)
- `_ref_add.note` (~16 tok, tiny) — e7bdbecdd47c736566c96f0ca4695499c81e0b6d087919959b99b543c67235e0
- `_ref_if_ret.note` (~16 tok, tiny) — 72e8c82f9be032e5285fa76776e2fd2a4c7ea0d010ecb379d6a00c8498f92656
- `_ref_main.note` (~16 tok, tiny) — 06880c14908822679e15a66a45995f5bc11ec75310b45738588ffcaf754b1621
- `_ref_recursion.note` (~16 tok, tiny) — 04947ba7952c2360a23d5281d0a35204ce8461b7b3265bf73453c4af49dda251
- `_ref_struct_field.note` (~16 tok, tiny) — 2cb2083c37be471776166a99580c7e98553a78ffd9a58d992c40f6e7310ffdb4
- `_ref_value_if.note` (~16 tok, tiny) — b3b594d4d55dfc1360f8ad0fc873a8ba541da0c6b8e1e9d1a52a9446c40f4d64
- `_regen_oracle.py` (~1249 tok, large) — Refresh the STALE frozen native_elf oracle notes.
- `self_host_body_smoke.py` (~2995 tok, huge)
- `selfhost_driver.mind` (~623 tok, large) — ===========================================================================
- `self_host_loop_smoke.py` (~1834 tok, huge)
- `self_host_mlir_smoke.py` (~1736 tok, huge)
- `self_host_native_elf_smoke.py` (~10022 tok, huge)
- `sha256_hash_smoke.py` (~732 tok, large) — # Copyright 2025 STARGA Inc.
- `struct_fields_smoke.py` (~1076 tok, large)
- `struct_lit_smoke.py` (~1923 tok, huge)
### `examples/mindc_mind/testdata/native_elf_oracle/`

- `add.elf` (~125 tok, small) — ELF> @@@8
- `if_ret.elf` (~150 tok, small) — ELF> @@@8
- `MANIFEST.txt` (~152 tok, small) — # Frozen native-ELF oracle references, captured before #15 deletes src/native.
- `recursion.elf` (~175 tok, small) — ELF> @@@8
- `struct_field.elf` (~166 tok, small) — ELF> @@@8
- `value_if.elf` (~147 tok, small) — ELF> @@@8
### `examples/mindc_mind/testdata/selfhost_loop/`

- `MANIFEST.txt` (~83 tok, small) — # Frozen self-host bootstrap ELF (A6): the driver-bearing native compiler that
### `examples/mindc_mind/`

- `unified_dispatch_smoke.py` (~1553 tok, huge)
- `validate_real_fns_smoke.py` (~2674 tok, huge)
- `while_struct_smoke.py` (~1031 tok, large)
### `examples/`

- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
### `examples/native/`

- `ci_kernel.mind` (~39 tok, tiny)
- `ci_kernel_smoke.py` (~417 tok, medium)
- `fib.mind` (~34 tok, tiny) — fn fib(n: i64) -> i64 {
- `loop.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/parser/`

- `bootstrap_smoke.py` (~5544 tok, huge)
- `EXPECTED.md` (~2140 tok, huge) — Phase 6.2 — Expected AST Tree
- `fixture.mind` (~160 tok, small) — Phase 6.2 parser smoke fixture.
- `main.mind` (~7825 tok, huge) — examples/parser/main.mind — RFC 0005 Phase 6.2 self-host parser seed.
- `README.md` (~2254 tok, huge) — RFC 0005 Phase 6.2 — Self-Host Parser Seed
### `examples/`

- `policy.mind` (~1301 tok, large) — policy.mind — v0.1 Execution Boundary Kernel
- `README.md` (~2066 tok, huge) — MIND Examples
- `remizov_benchmark.mind` (~6400 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_feynman.mind` (~2894 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_gpu.mind` (~2662 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_inverse.mind` (~2614 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_solver.mind` (~3802 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_verify.mind` (~3721 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `tiny_edge_model.mind` (~1876 tok, huge) — Tiny Edge Model Example
### `examples/typecheck/`

- `bootstrap_smoke.py` (~2608 tok, huge)
- `EXPECTED.md` (~2015 tok, huge) — Phase 6.3 — Expected Type-Check Report
- `fixture.mind` (~198 tok, small) — Phase 6.3 type-checker smoke fixture.
- `main.mind` (~7120 tok, huge) — examples/typecheck/main.mind — RFC 0005 Phase 6.3 self-host
- `README.md` (~2612 tok, huge) — RFC 0005 Phase 6.3 — Self-Host Type-Checker Seed
### `examples/zoo/`

- `conv_classifier.mind` (~2407 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `linear_regression.mind` (~1347 tok, large) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `logistic_classifier.mind` (~1517 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `mlp_mnist.mind` (~2275 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `README.md` (~1191 tok, large) — MIND Model Zoo
- `transformer_block.mind` (~3781 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
### `experiments/global-vs-local/`

- `chern.py` (~1250 tok, large)
- `exp2.py` (~1337 tok, large)
- `exp3_universal.py` (~997 tok, large)
- `plot_chern.py` (~469 tok, medium)
- `plot.py` (~904 tok, large)
- `README.md` (~839 tok, large) — Global vs Local — "Closed-form whole-field invariant" experiments
- `topo.py` (~696 tok, large)
### `.githooks/`

- `pre-commit` (~255 tok, medium) — #!/usr/bin/env bash
### `.github/`

- `CODEOWNERS` (~8 tok, tiny) — *       @cputer
### `.github/ISSUE_TEMPLATE/`

- `bounty_claim.md` (~56 tok, small)
- `bug_report.md` (~213 tok, medium) — Describe the bug
- `feature_request.md` (~171 tok, small) — Problem Statement
### `.github/`

- `PULL_REQUEST_TEMPLATE.md` (~55 tok, small) — Summary
- `release-drafter.yml` (~85 tok, small) — name-template: 'v$NEXT_PATCH_VERSION'
### `.github/workflows/`

- `bench-gate.yml` (~1432 tok, large) — name: Bench gate
- `cargo-deny.yml` (~222 tok, medium) — name: Cargo Deny
- `ci.yml` (~4967 tok, huge) — name: CI
- `crypto-vectors.yml` (~1093 tok, large) — name: Crypto Vectors
- `docs-claims.yml` (~364 tok, medium) — name: Docs Claims
- `link-check.yml` (~221 tok, medium) — name: Link Check
- `mindcraft.yml` (~545 tok, large) — name: Mindcraft Check
- `release-drafter.yml` (~91 tok, small) — name: Release Drafter
- `release.yml` (~1756 tok, huge) — name: Release
### `mind/std/cognitive/`

- `batch_scheduler.mind` (~850 tok, large) — Batch scheduling for inference workloads
- `kv_cache.mind` (~840 tok, large) — KV-Cache for transformer inference
- `speculative.mind` (~891 tok, large) — Speculative decoding with rejection sampling
- `verification.mind` (~948 tok, large) — Verification plane for inference consistency (LCU)
### `runtime-support/`

- `mind_intrinsics.c` (~17416 tok, huge) — Copyright 2025 STARGA Inc.
### `scripts/`

- `anatomy-hook.sh` (~258 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2010 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `check_claims.py` (~2779 tok, huge) — Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.
- `check_no_ai_attribution.sh` (~310 tok, medium) — Public-artifact hygiene gate: no AI tool/model named as having worked on MIND.
- `install.ps1` (~1856 tok, huge) — # install.ps1 - mindc one-line installer for Windows (PowerShell)
- `install.sh` (~1054 tok, large) — MIND compiler (mindc) installer — downloads a pre-built binary from the
### `scripts/mind-vs-rust/`

- `Cargo.toml` (~271 tok, medium) — [package]
- `.gitignore` (~3 tok, tiny) — */target-*/
- `run.sh` (~659 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `scripts/mind-vs-rust/src/`

- `main.rs` (~2372 tok, huge) — Copyright 2026 STARGA Inc.
### `scripts/`

- `preflight.sh` (~807 tok, large) — preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
- `run_crypto_vectors.sh` (~1751 tok, huge) — Build every pure-MIND crypto/TLS std module to a shared object and run its
### `sdk/ts/mic-map/dist/`

- `errors.d.ts` (~209 tok, medium)
- `errors.d.ts.map` (~147 tok, small) — {"version":3,"file":"errors.d.ts","sourceRoot":"","sources":["../src/errors.ts"]
- `errors.js` (~350 tok, medium) — Copyright 2026 STARGA Inc.
- `errors.js.map` (~279 tok, medium) — {"version":3,"file":"errors.js","sourceRoot":"","sources":["../src/errors.ts"],"
- `framing.d.ts` (~190 tok, small)
- `framing.d.ts.map` (~82 tok, small) — {"version":3,"file":"framing.d.ts","sourceRoot":"","sources":["../src/framing.ts
- `framing.js` (~757 tok, large) — Copyright 2026 STARGA Inc.
- `framing.js.map` (~627 tok, large) — {"version":3,"file":"framing.js","sourceRoot":"","sources":["../src/framing.ts"]
- `index.d.ts` (~272 tok, medium)
- `index.d.ts.map` (~244 tok, medium) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../src/index.ts"],"
- `index.js` (~459 tok, medium) — Copyright 2026 STARGA Inc.
- `index.js.map` (~393 tok, medium) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../src/index.ts"],"na
- `map.d.ts` (~355 tok, medium)
- `map.d.ts.map` (~226 tok, medium) — {"version":3,"file":"map.d.ts","sourceRoot":"","sources":["../src/map.ts"],"name
- `map.js` (~2064 tok, huge) — Copyright 2026 STARGA Inc.
- `map.js.map` (~2167 tok, huge) — {"version":3,"file":"map.js","sourceRoot":"","sources":["../src/map.ts"],"names"
- `mic2_emit.d.ts` (~117 tok, small)
- `mic2_emit.d.ts.map` (~58 tok, small) — {"version":3,"file":"mic2_emit.d.ts","sourceRoot":"","sources":["../src/mic2_emi
- `mic2_emit.js` (~594 tok, large) — Copyright 2026 STARGA Inc.
- `mic2_emit.js.map` (~655 tok, large) — {"version":3,"file":"mic2_emit.js","sourceRoot":"","sources":["../src/mic2_emit.
- `mic2_parse.d.ts` (~70 tok, small)
- `mic2_parse.d.ts.map` (~64 tok, small) — {"version":3,"file":"mic2_parse.d.ts","sourceRoot":"","sources":["../src/mic2_pa
- `mic2_parse.js` (~2396 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2_parse.js.map` (~2613 tok, huge) — {"version":3,"file":"mic2_parse.js","sourceRoot":"","sources":["../src/mic2_pars
- `micb.d.ts` (~171 tok, small)
- `micb.d.ts.map` (~93 tok, small) — {"version":3,"file":"micb.d.ts","sourceRoot":"","sources":["../src/micb.ts"],"na
- `micb.js` (~3256 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.js.map` (~3469 tok, huge) — {"version":3,"file":"micb.js","sourceRoot":"","sources":["../src/micb.ts"],"name
- `types.d.ts` (~946 tok, large)
- `types.d.ts.map` (~772 tok, large) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
- `types.js` (~1458 tok, large) — Copyright 2026 STARGA Inc.
- `types.js.map` (~1822 tok, huge) — {"version":3,"file":"types.js","sourceRoot":"","sources":["../src/types.ts"],"na
- `varint.d.ts` (~318 tok, medium)
- `varint.d.ts.map` (~185 tok, small) — {"version":3,"file":"varint.d.ts","sourceRoot":"","sources":["../src/varint.ts"]
- `varint.js` (~612 tok, large) — Copyright 2026 STARGA Inc.
- `varint.js.map` (~554 tok, large) — {"version":3,"file":"varint.js","sourceRoot":"","sources":["../src/varint.ts"],"
### `sdk/ts/mic-map/`

- `LICENSE` (~2573 tok, huge) —                                  Apache License
### `sdk/ts/mic-map/node_modules/@ampproject/remapping/dist/`

- `remapping.mjs` (~2135 tok, huge) — import { decodedMappings, traceSegment, TraceMap } from '@jridgewell/trace-mappi
- `remapping.mjs.map` (~4353 tok, huge) — {"version":3,"file":"remapping.mjs","sources":["../src/source-map-tree.ts","../s
- `remapping.umd.js` (~2432 tok, huge) — TODO: Eventually support sourceRoot, which has to be removed because the sources are already
- `remapping.umd.js.map` (~4395 tok, huge) — {"version":3,"file":"remapping.umd.js","sources":["../src/source-map-tree.ts",".
### `sdk/ts/mic-map/node_modules/@ampproject/remapping/dist/types/`

- `build-source-map-tree.d.ts` (~200 tok, medium)
- `remapping.d.ts` (~274 tok, medium)
- `source-map.d.ts` (~156 tok, small)
- `source-map-tree.d.ts` (~419 tok, medium)
- `types.d.ts` (~152 tok, small)
### `sdk/ts/mic-map/node_modules/@ampproject/remapping/`

- `LICENSE` (~2840 tok, huge)
- `package.json` (~559 tok, large) — Keys: name, version, description, keywords, main
- `README.md` (~1826 tok, huge) — @ampproject/remapping
### `sdk/ts/mic-map/node_modules/ansi-regex/`

- `index.d.ts` (~173 tok, small)
- `index.js` (~148 tok, small) — Valid string terminator sequences are BEL, ESC\, and 0x9c
- `license` (~280 tok, medium) — MIT License
- `package.json` (~259 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~609 tok, large) — ansi-regex
### `sdk/ts/mic-map/node_modules/ansi-styles/`

- `index.d.ts` (~1297 tok, large)
- `index.js` (~1314 tok, large) — 21 isn't widely supported and 22 does the same thing
- `license` (~280 tok, medium) — MIT License
- `package.json` (~256 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~1227 tok, large) — ansi-styles
### `sdk/ts/mic-map/node_modules/assertion-error/`

- `index.d.ts` (~176 tok, small) — deno-lint-ignore ban-types
- `index.js` (~359 tok, medium) — deno-fmt-ignore-file
- `LICENSE` (~277 tok, medium) — MIT License
- `package.json` (~216 tok, medium) — Keys: name, version, description, author, license
- `README.md` (~431 tok, medium) — What is AssertionError?
### `sdk/ts/mic-map/node_modules/@babel/helper-string-parser/lib/`

- `index.js` (~1965 tok, huge)
- `index.js.map` (~5440 tok, huge) — {"version":3,"names":["isDigit","code","forbiddenNumericSeparatorSiblings","decB
### `sdk/ts/mic-map/node_modules/@babel/helper-string-parser/`

- `LICENSE` (~277 tok, medium) — MIT License
- `package.json` (~190 tok, small) — Keys: name, version, description, repository, homepage
- `README.md` (~84 tok, small) — @babel/helper-string-parser
### `sdk/ts/mic-map/node_modules/@babel/helper-validator-identifier/lib/`

- `identifier.js` (~3136 tok, huge)
- `identifier.js.map` (~6693 tok, huge) — {"version":3,"names":["nonASCIIidentifierStartChars","nonASCIIidentifierChars","
- `index.js` (~341 tok, medium)
- `index.js.map` (~127 tok, small) — {"version":3,"names":["_identifier","require","_keyword"],"sources":["../src/ind
- `keyword.js` (~395 tok, medium)
- `keyword.js.map` (~961 tok, large) — {"version":3,"names":["reservedWords","keyword","strict","strictBind","keywords"
### `sdk/ts/mic-map/node_modules/@babel/helper-validator-identifier/`

- `LICENSE` (~277 tok, medium) — MIT License
- `package.json` (~185 tok, small) — Keys: name, version, description, repository, license
- `README.md` (~93 tok, small) — @babel/helper-validator-identifier
### `sdk/ts/mic-map/node_modules/@babel/parser/bin/`

- `babel-parser.js` (~91 tok, small)
### `sdk/ts/mic-map/node_modules/@babel/parser/`

- `CHANGELOG.md` (~9560 tok, huge) — Changelog
- `LICENSE` (~272 tok, medium) — Copyright (C) 2012-2014 by various contributors (see AUTHORS)
- `package.json` (~346 tok, medium) — Keys: name, version, description, author, homepage
- `README.md` (~103 tok, small) — @babel/parser
### `sdk/ts/mic-map/node_modules/@babel/parser/typings/`

- `babel-parser.d.ts` (~2330 tok, huge) — This file is auto-generated! Do not modify it directly.
### `sdk/ts/mic-map/node_modules/@babel/types/lib/asserts/`

- `assertNode.js` (~117 tok, small)
- `assertNode.js.map` (~211 tok, medium) — {"version":3,"names":["_isNode","require","assertNode","node","isNode","_node$ty
### `sdk/ts/mic-map/node_modules/@babel/types/lib/asserts/generated/`

- `index.js` (~11399 tok, huge)
- `index.js.map` (~25268 tok, huge) — {"version":3,"names":["_is","require","_deprecationWarning","assert","type","nod
### `sdk/ts/mic-map/node_modules/@babel/types/lib/ast-types/generated/`

- `index.js` (~13 tok, tiny)
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/flow/`

- `createFlowUnionType.js` (~134 tok, small)
- `createFlowUnionType.js.map` (~302 tok, medium) — {"version":3,"names":["_index","require","_removeTypeDuplicates","createFlowUnio
- `createTypeAnnotationBasedOnTypeof.js` (~265 tok, medium)
- `createTypeAnnotationBasedOnTypeof.js.map` (~666 tok, large) — {"version":3,"names":["_index","require","_default","exports","default","createT
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/generated/`

- `index.js` (~203 tok, medium)
- `index.js.map` (~1694 tok, huge) — {"version":3,"names":["_lowercase","require","Object","keys","forEach","key","ex
- `lowercase.js` (~21434 tok, huge)
- `uppercase.js` (~6773 tok, huge)
- `uppercase.js.map` (~9178 tok, huge) — {"version":3,"names":["b","require","_deprecationWarning","alias","lowercase","A
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/`

- `productions.js` (~84 tok, small)
- `productions.js.map` (~132 tok, small) — {"version":3,"names":["_index","require","buildUndefinedNode","unaryExpression",
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/react/`

- `buildChildren.js` (~193 tok, small)
- `buildChildren.js.map` (~450 tok, medium) — {"version":3,"names":["_index","require","_cleanJSXElementLiteralChild","buildCh
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/typescript/`

- `createTSUnionType.js` (~183 tok, small)
- `createTSUnionType.js.map` (~407 tok, medium) — {"version":3,"names":["_index","require","_removeTypeDuplicates","_index2","crea
### `sdk/ts/mic-map/node_modules/@babel/types/lib/builders/`

- `validateNode.js` (~154 tok, small)
- `validateNode.js.map` (~368 tok, medium) — {"version":3,"names":["_validate","require","_index","validateNode","node","fiel
### `sdk/ts/mic-map/node_modules/@babel/types/lib/clone/`

- `cloneDeep.js` (~66 tok, small)
- `cloneDeep.js.map` (~159 tok, small) — {"version":3,"names":["_cloneNode","require","cloneDeep","node","cloneNode"],"so
- `cloneDeepWithoutLoc.js` (~76 tok, small)
- `cloneDeepWithoutLoc.js.map` (~184 tok, small) — {"version":3,"names":["_cloneNode","require","cloneDeepWithoutLoc","node","clone
- `clone.js` (~64 tok, small)
- `clone.js.map` (~157 tok, small) — {"version":3,"names":["_cloneNode","require","clone","node","cloneNode"],"source
- `cloneNode.js` (~829 tok, large)
- `cloneNode.js.map` (~2284 tok, huge) — {"version":3,"names":["_index","require","_index2","hasOwn","Function","call","b
- `cloneWithoutLoc.js` (~73 tok, small)
- `cloneWithoutLoc.js.map` (~161 tok, small) — {"version":3,"names":["_cloneNode","require","cloneWithoutLoc","node","cloneNode
### `sdk/ts/mic-map/node_modules/@babel/types/lib/comments/`

- `addComment.js` (~94 tok, small)
- `addComment.js.map` (~225 tok, medium) — {"version":3,"names":["_addComments","require","addComment","node","type","conte
- `addComments.js` (~119 tok, small)
- `addComments.js.map` (~299 tok, medium) — {"version":3,"names":["addComments","node","type","comments","key","concat","pus
- `inheritInnerComments.js` (~81 tok, small)
- `inheritInnerComments.js.map` (~144 tok, small) — {"version":3,"names":["_inherit","require","inheritInnerComments","child","paren
- `inheritLeadingComments.js` (~83 tok, small)
- `inheritLeadingComments.js.map` (~147 tok, small) — {"version":3,"names":["_inherit","require","inheritLeadingComments","child","par
- `inheritsComments.js` (~149 tok, small)
- `inheritsComments.js.map` (~299 tok, medium) — {"version":3,"names":["_inheritTrailingComments","require","_inheritLeadingComme
- `inheritTrailingComments.js` (~84 tok, small)
- `inheritTrailingComments.js.map` (~148 tok, small) — {"version":3,"names":["_inherit","require","inheritTrailingComments","child","pa
- `removeComments.js` (~81 tok, small)
- `removeComments.js.map` (~173 tok, small) — {"version":3,"names":["_index","require","removeComments","node","COMMENT_KEYS",
### `sdk/ts/mic-map/node_modules/@babel/types/lib/constants/generated/`

- `index.js` (~1591 tok, huge)
- `index.js.map` (~2165 tok, huge) — {"version":3,"names":["_index","require","STANDARDIZED_TYPES","exports","FLIPPED
### `sdk/ts/mic-map/node_modules/@babel/types/lib/constants/`

- `index.js` (~685 tok, large)
- `index.js.map` (~1169 tok, large) — {"version":3,"names":["STATEMENT_OR_BLOCK_KEYS","exports","FLATTENABLE_KEYS","FO
### `sdk/ts/mic-map/node_modules/@babel/types/lib/converters/`

- `ensureBlock.js` (~84 tok, small)
- `ensureBlock.js.map` (~256 tok, medium) — {"version":3,"names":["_toBlock","require","ensureBlock","node","key","result","
- `gatherSequenceExpressions.js` (~610 tok, large)
- `gatherSequenceExpressions.js.map` (~1555 tok, huge) — {"version":3,"names":["_getBindingIdentifiers","require","_index","_index2","_pr
- `toBindingIdentifierName.js` (~99 tok, small)
- `toBindingIdentifierName.js.map` (~169 tok, small) — {"version":3,"names":["_toIdentifier","require","toBindingIdentifierName","name"
- `toBlock.js` (~190 tok, small)
- `toBlock.js.map` (~428 tok, medium) — {"version":3,"names":["_index","require","_index2","toBlock","node","parent","is
- `toComputedKey.js` (~113 tok, small)
- `toComputedKey.js.map` (~305 tok, medium) — {"version":3,"names":["_index","require","_index2","toComputedKey","node","key",
- `toExpression.js` (~185 tok, small)
- `toExpression.js.map` (~634 tok, large) — {"version":3,"names":["_index","require","_default","exports","default","toExpre
- `toIdentifier.js` (~185 tok, small)
- `toIdentifier.js.map` (~413 tok, medium) — {"version":3,"names":["_isValidIdentifier","require","_helperValidatorIdentifier
- `toKeyAlias.js` (~262 tok, medium)
- `toKeyAlias.js.map` (~667 tok, large) — {"version":3,"names":["_index","require","_cloneNode","_removePropertiesDeep","t
- `toSequenceExpression.js` (~135 tok, small)
- `toSequenceExpression.js.map` (~452 tok, medium) — {"version":3,"names":["_gatherSequenceExpressions","require","toSequenceExpressi
- `toStatement.js` (~250 tok, medium)
- `toStatement.js.map` (~744 tok, large) — {"version":3,"names":["_index","require","_index2","_default","exports","default
- `valueToNode.js` (~708 tok, large)
- `valueToNode.js.map` (~1953 tok, huge) — {"version":3,"names":["_isValidIdentifier","require","_index","_default","export
### `sdk/ts/mic-map/node_modules/@babel/types/lib/definitions/`

- `core.js` (~13912 tok, huge)
- `deprecated-aliases.js` (~69 tok, small)
- `deprecated-aliases.js.map` (~90 tok, small) — {"version":3,"names":["DEPRECATED_ALIASES","exports","ModuleDeclaration"],"sourc
- `experimental.js` (~722 tok, large)
- `experimental.js.map` (~1742 tok, huge) — {"version":3,"names":["_utils","require","defineType","visitor","aliases","field
- `flow.js` (~4129 tok, huge)
- `flow.js.map` (~8356 tok, huge) — {"version":3,"names":["_core","require","_utils","defineType","defineAliasedType
- `index.js` (~688 tok, large)
- `index.js.map` (~693 tok, large) — {"version":3,"names":["require","_utils","_placeholders","_deprecatedAliases","O
- `jsx.js` (~1058 tok, large)
- `jsx.js.map` (~2365 tok, huge) — {"version":3,"names":["_utils","require","defineType","defineAliasedType","visit
- `misc.js` (~187 tok, small)
- `misc.js.map` (~468 tok, medium) — {"version":3,"names":["_utils","require","_placeholders","_core","defineType","d
- `placeholders.js` (~261 tok, medium)
- `placeholders.js.map` (~513 tok, large) — {"version":3,"names":["_utils","require","PLACEHOLDERS","exports","PLACEHOLDERS_
- `typescript.js` (~4215 tok, huge)
- `typescript.js.map` (~9861 tok, huge) — {"version":3,"names":["_utils","require","_core","_is","defineType","defineAlias
- `utils.js` (~2422 tok, huge)
- `utils.js.map` (~5915 tok, huge) — {"version":3,"names":["_is","require","_validate","VISITOR_KEYS","exports","ALIA
### `sdk/ts/mic-map/node_modules/@babel/types/lib/`

- `index.js` (~4327 tok, huge)
- `index.js.map` (~3318 tok, huge) — {"version":3,"names":["_isReactComponent","require","_isCompatTag","_buildChildr
### `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/`

- `appendToMemberExpression.js` (~120 tok, small)
- `appendToMemberExpression.js.map` (~278 tok, medium) — {"version":3,"names":["_index","require","appendToMemberExpression","member","ap
### `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/flow/`

- `removeTypeDuplicates.js` (~468 tok, medium)
- `removeTypeDuplicates.js.map` (~1254 tok, large) — {"version":3,"names":["_index","require","getQualifiedName","node","isIdentifier
### `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/`

- `inherits.js` (~187 tok, small)
- `inherits.js.map` (~542 tok, large) — {"version":3,"names":["_index","require","_inheritsComments","inherits","child",
- `prependToMemberExpression.js` (~138 tok, small)
- `prependToMemberExpression.js.map` (~295 tok, medium) — {"version":3,"names":["_index","require","_index2","prependToMemberExpression","
- `removePropertiesDeep.js` (~105 tok, small)
- `removePropertiesDeep.js.map` (~201 tok, medium) — {"version":3,"names":["_traverseFast","require","_removeProperties","removePrope
- `removeProperties.js` (~201 tok, medium)
- `removeProperties.js.map` (~601 tok, large) — {"version":3,"names":["_index","require","CLEAR_KEYS","CLEAR_KEYS_PLUS_COMMENTS"
### `sdk/ts/mic-map/node_modules/@babel/types/lib/modifications/typescript/`

- `removeTypeDuplicates.js` (~502 tok, large)
- `removeTypeDuplicates.js.map` (~1368 tok, large) — {"version":3,"names":["_index","require","getQualifiedName","node","isIdentifier
### `sdk/ts/mic-map/node_modules/@babel/types/lib/retrievers/`

- `getAssignmentIdentifiers.js` (~289 tok, medium)
- `getAssignmentIdentifiers.js.map` (~707 tok, large) — {"version":3,"names":["getAssignmentIdentifiers","node","search","concat","ids",
- `getBindingIdentifiers.js` (~738 tok, large)
- `getBindingIdentifiers.js.map` (~2254 tok, huge) — {"version":3,"names":["_index","require","getBindingIdentifiers","node","duplica
- `getFunctionName.js` (~431 tok, medium)
- `getFunctionName.js.map` (~1239 tok, large) — {"version":3,"names":["_index","require","getNameFromLiteralId","id","isNullLite
- `getOuterBindingIdentifiers.js` (~105 tok, small)
- `getOuterBindingIdentifiers.js.map` (~278 tok, medium) — {"version":3,"names":["_getBindingIdentifiers","require","_default","exports","d
### `sdk/ts/mic-map/node_modules/@babel/types/lib/traverse/`

- `traverseFast.js` (~241 tok, medium)
- `traverseFast.js.map` (~698 tok, large) — {"version":3,"names":["_index","require","_skip","Symbol","_stop","traverseFast"
- `traverse.js` (~307 tok, medium)
- `traverse.js.map` (~884 tok, large) — {"version":3,"names":["_index","require","traverse","node","handlers","state","e
### `sdk/ts/mic-map/node_modules/@babel/types/lib/utils/`

- `deprecationWarning.js` (~306 tok, medium)
- `deprecationWarning.js.map` (~807 tok, large) — {"version":3,"names":["warnings","Set","deprecationWarning","oldName","newName",
- `inherit.js` (~76 tok, small)
- `inherit.js.map` (~223 tok, medium) — {"version":3,"names":["inherit","key","child","parent","Array","from","Set","con
### `sdk/ts/mic-map/node_modules/@babel/types/lib/utils/react/`

- `cleanJSXElementLiteralChild.js` (~294 tok, medium)
- `cleanJSXElementLiteralChild.js.map` (~710 tok, large) — {"version":3,"names":["_index","require","_index2","cleanJSXElementLiteralChild"
### `sdk/ts/mic-map/node_modules/@babel/types/lib/utils/`

- `shallowEqual.js` (~88 tok, small)
- `shallowEqual.js.map` (~203 tok, medium) — {"version":3,"names":["shallowEqual","actual","expected","keys","Object","key"],
### `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/`

- `buildMatchMemberExpression.js` (~103 tok, small)
- `buildMatchMemberExpression.js.map` (~270 tok, medium) — {"version":3,"names":["_matchesPattern","require","buildMatchMemberExpression","
### `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/generated/`

- `index.js` (~24081 tok, huge)
### `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/`

- `isBinding.js` (~194 tok, small)
- `isBinding.js.map` (~509 tok, large) — {"version":3,"names":["_getBindingIdentifiers","require","isBinding","node","par
- `isBlockScoped.js` (~98 tok, small)
- `isBlockScoped.js.map` (~211 tok, medium) — {"version":3,"names":["_index","require","_isLet","isBlockScoped","node","isFunc
- `isImmutable.js` (~122 tok, small)
- `isImmutable.js.map` (~266 tok, medium) — {"version":3,"names":["_isType","require","_index","isImmutable","node","isType"
- `is.js` (~193 tok, small)
- `is.js.map` (~763 tok, large) — {"version":3,"names":["_shallowEqual","require","_isType","_isPlaceholderType","
- `isLet.js` (~96 tok, small)
- `isLet.js.map` (~293 tok, medium) — {"version":3,"names":["_index","require","BLOCK_SCOPED_SYMBOL","Symbol","for","i
- `isNode.js` (~68 tok, small)
- `isNode.js.map` (~134 tok, small) — {"version":3,"names":["_index","require","isNode","node","VISITOR_KEYS","type"],
- `isNodesEquivalent.js` (~371 tok, medium)
- `isNodesEquivalent.js.map` (~873 tok, large) — {"version":3,"names":["_index","require","isNodesEquivalent","a","b","type","fie
- `isPlaceholderType.js` (~118 tok, small)
- `isPlaceholderType.js.map` (~250 tok, medium) — {"version":3,"names":["_index","require","isPlaceholderType","placeholderType","
- `isReferenced.js` (~654 tok, large)
- `isReferenced.js.map` (~1759 tok, huge) — {"version":3,"names":["isReferenced","node","parent","grandparent","type","prope
- `isScope.js` (~134 tok, small)
- `isScope.js.map` (~380 tok, medium) — {"version":3,"names":["_index","require","isScope","node","parent","isBlockState
- `isSpecifierDefault.js` (~103 tok, small)
- `isSpecifierDefault.js.map` (~249 tok, medium) — {"version":3,"names":["_index","require","isSpecifierDefault","specifier","isImp
- `isType.js` (~127 tok, small)
- `isType.js.map` (~422 tok, medium) — {"version":3,"names":["_index","require","isType","nodeType","targetType","ALIAS
- `isValidES3Identifier.js` (~163 tok, small)
- `isValidES3Identifier.js.map` (~371 tok, medium) — {"version":3,"names":["_isValidIdentifier","require","RESERVED_WORDS_ES3_ONLY","
- `isValidIdentifier.js` (~146 tok, small)
- `isValidIdentifier.js.map` (~298 tok, medium) — {"version":3,"names":["_helperValidatorIdentifier","require","isValidIdentifier"
- `isVar.js` (~96 tok, small)
- `isVar.js.map` (~295 tok, medium) — {"version":3,"names":["_index","require","BLOCK_SCOPED_SYMBOL","Symbol","for","i
- `matchesPattern.js` (~356 tok, medium)
- `matchesPattern.js.map` (~974 tok, large) — {"version":3,"names":["_index","require","isMemberExpressionLike","node","isMemb
### `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/react/`

- `isCompatTag.js` (~58 tok, small)
- `isCompatTag.js.map` (~110 tok, small) — {"version":3,"names":["isCompatTag","tagName","test"],"sources":["../../../src/v
- `isReactComponent.js` (~92 tok, small)
- `isReactComponent.js.map` (~147 tok, small) — {"version":3,"names":["_buildMatchMemberExpression","require","isReactComponent"
### `sdk/ts/mic-map/node_modules/@babel/types/lib/validators/`

- `validate.js` (~373 tok, medium)
- `validate.js.map` (~795 tok, large) — {"version":3,"names":["_index","require","validate","node","key","val","fields",
### `sdk/ts/mic-map/node_modules/@babel/types/`

- `LICENSE` (~277 tok, medium) — MIT License
- `package.json` (~267 tok, medium) — Keys: name, version, description, author, homepage
- `README.md` (~112 tok, small) — @babel/types
### `sdk/ts/mic-map/node_modules/balanced-match/dist/commonjs/`

- `index.d.ts` (~84 tok, small)
- `index.d.ts.map` (~81 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~445 tok, medium)
- `index.js.map` (~1048 tok, large) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/balanced-match/dist/esm/`

- `index.d.ts` (~84 tok, small)
- `index.d.ts.map` (~81 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~402 tok, medium)
- `index.js.map` (~1039 tok, large) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/balanced-match/`

- `LICENSE.md` (~289 tok, medium)
- `package.json` (~399 tok, medium) — Keys: name, description, version, files, repository
- `README.md` (~435 tok, medium) — balanced-match
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/`

- `CHANGELOG.md` (~2455 tok, huge) — Next
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/dist/lib/`

- `ascii.d.ts` (~140 tok, small)
- `ascii.js` (~4163 tok, huge)
- `ascii.mjs` (~4109 tok, huge) — import { compareRangeCovs } from "./compare";
- `CHANGELOG.md` (~2455 tok, huge) — Next
- `clone.d.ts` (~243 tok, medium)
- `clone.js` (~1428 tok, large)
- `clone.mjs` (~1364 tok, large) — /**
- `compare.d.ts` (~199 tok, small)
- `compare.js` (~941 tok, large)
- `compare.mjs` (~883 tok, large) — /**
- `index.d.ts` (~116 tok, small)
- `index.js` (~609 tok, large)
- `index.mjs` (~421 tok, medium) — export { emitForest, emitForestLines, parseFunctionRanges, parseOffsets } from "
- `LICENSE.md` (~272 tok, medium)
- `merge.d.ts` (~377 tok, medium)
- `merge.js` (~9080 tok, huge) — assert: `scripts.length > 0`
- `merge.mjs` (~9056 tok, huge) — import { deepNormalizeScriptCov, normalizeFunctionCov, normalizeProcessCov, norm
- `normalize.d.ts` (~415 tok, medium)
- `normalize.js` (~1912 tok, huge)
- `normalize.mjs` (~1836 tok, huge) — import { compareFunctionCovs, compareRangeCovs, compareScriptCovs } from "./comp
- `package.json` (~266 tok, medium) — Keys: name, version, description, author, license
- `range-tree.d.ts` (~182 tok, small)
- `range-tree.js` (~3992 tok, huge) — Stack of parent trees and parent counts.
- `range-tree.mjs` (~3964 tok, huge) — export class RangeTree {
- `README.md` (~187 tok, small) — V8 Coverage
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/dist/lib/_src/`

- `ascii.ts` (~1130 tok, large)
- `clone.ts` (~429 tok, medium)
- `compare.ts` (~272 tok, medium)
- `index.ts` (~116 tok, small)
- `merge.ts` (~2718 tok, huge) — assert: `scripts.length > 0`
- `normalize.ts` (~594 tok, large)
- `range-tree.ts` (~1068 tok, large) — Stack of parent trees and parent counts.
- `types.ts` (~107 tok, small)
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/dist/lib/`

- `tsconfig.json` (~402 tok, medium) — Keys: compilerOptions, include, exclude
- `types.d.ts` (~112 tok, small)
- `types.js` (~228 tok, medium)
- `types.mjs` (~209 tok, medium)
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/`

- `.editorconfig` (~37 tok, tiny) — root = true
- `.gitattributes` (~15 tok, tiny) — # Enforce `lf` for text files (even on Windows)
- `gulpfile.ts` (~584 tok, large) — generateTestMain: true,
- `LICENSE.md` (~272 tok, medium)
- `LICENSE.txt` (~183 tok, small) — Copyright (c) 2017, Contributors
- `package.json` (~292 tok, medium) — Keys: name, version, description, author, license
- `README.md` (~187 tok, small) — V8 Coverage
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/src/lib/`

- `ascii.ts` (~1130 tok, large)
- `clone.ts` (~429 tok, medium)
- `compare.ts` (~272 tok, medium)
- `index.ts` (~116 tok, small)
- `merge.ts` (~2718 tok, huge) — assert: `scripts.length > 0`
- `normalize.ts` (~594 tok, large)
- `range-tree.ts` (~1068 tok, large) — Stack of parent trees and parent counts.
- `types.ts` (~107 tok, small)
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/src/test/`

- `merge.spec.ts` (~2390 tok, huge) — see: https://github.com/demurgos/v8-coverage/issues/2
### `sdk/ts/mic-map/node_modules/@bcoe/v8-coverage/`

- `tsconfig.json` (~393 tok, medium) — Keys: compilerOptions
### `sdk/ts/mic-map/node_modules/.bin/`

- `esbuild` (~2426880 tok, huge) — ELF>`�F@�@8@@
- `glob` (~3162 tok, huge) — #!/usr/bin/env node
- `nanoid` (~283 tok, medium) — #!/usr/bin/env node
- `node-which` (~247 tok, medium) — #!/usr/bin/env node
- `parser` (~91 tok, small) — #!/usr/bin/env node
- `rollup` (~20568 tok, huge) — #!/usr/bin/env node
- `semver` (~1240 tok, large) — #!/usr/bin/env node
- `tsc` (~12 tok, tiny) — #!/usr/bin/env node
- `tsserver` (~13 tok, tiny) — #!/usr/bin/env node
- `vite` (~418 tok, medium) — #!/usr/bin/env node
- `vite-node` (~12 tok, tiny) — #!/usr/bin/env node
- `vitest` (~11 tok, tiny) — #!/usr/bin/env node
- `why-is-node-running` (~111 tok, small) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/brace-expansion/dist/commonjs/`

- `index.d.ts` (~57 tok, small)
- `index.d.ts.map` (~69 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~1727 tok, huge) — I don't know why Bash 4.3 does this, but it does.
- `index.js.map` (~3476 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/brace-expansion/dist/esm/`

- `index.d.ts` (~57 tok, small)
- `index.d.ts.map` (~69 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~1681 tok, huge) — I don't know why Bash 4.3 does this, but it does.
- `index.js.map` (~3479 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/brace-expansion/`

- `LICENSE` (~286 tok, medium) — MIT License
- `package.json` (~390 tok, medium) — Keys: name, description, version, files, exports
- `README.md` (~604 tok, large) — brace-expansion
### `sdk/ts/mic-map/node_modules/cac/deno/`

- `CAC.ts` (~2034 tok, huge) — Search the default command
- `Command.ts` (~1749 tok, huge)
- `deno.ts` (~60 tok, small) — Ignore the TypeScript errors
- `index.ts` (~59 tok, small)
- `Option.ts` (~375 tok, medium) — Use the longest name (last one) as actual option name
- `utils.ts` (~963 tok, large) — We do not set default values in mri options
### `sdk/ts/mic-map/node_modules/cac/dist/`

- `index.d.ts` (~1265 tok, large)
- `index.js` (~4526 tok, huge)
- `index.mjs` (~4497 tok, huge) — import { EventEmitter } from 'events';
### `sdk/ts/mic-map/node_modules/cac/`

- `index-compat.js` (~46 tok, tiny) — For backwards compatibility
- `LICENSE` (~280 tok, medium) — The MIT License (MIT)
- `mod.js` (~18 tok, tiny) — Deno users should use mod.ts instead
- `mod.ts` (~11 tok, tiny) — For Deno
- `package.json` (~630 tok, large) — Keys: name, version, description, repository, main
- `README.md` (~3943 tok, huge) — Introduction
### `sdk/ts/mic-map/node_modules/chai/`

- `chai.js` (~7 tok, tiny)
### `sdk/ts/mic-map/node_modules/chai/lib/chai/`

- `assertion.js` (~1632 tok, huge)
- `config.js` (~947 tok, large)
### `sdk/ts/mic-map/node_modules/chai/lib/chai/interface/`

- `assert.js` (~22489 tok, huge) — Comply with Node's fail([message]) interface
- `expect.js` (~313 tok, medium)
- `should.js` (~1510 tok, huge) — explicitly define this method as function as to have it's name to include as `ssfi`
### `sdk/ts/mic-map/node_modules/chai/lib/`

- `chai.js` (~319 tok, medium) — Assertion Error
### `sdk/ts/mic-map/node_modules/chai/lib/chai/utils/`

- `addChainableMethod.js` (~1326 tok, large) — Check whether `Object.setPrototypeOf` is supported
- `addLengthGuard.js` (~633 tok, large)
- `addMethod.js` (~544 tok, large) — Setting the `ssfi` flag to `methodWrapper` causes this function to be the
- `addProperty.js` (~598 tok, large) — Setting the `ssfi` flag to `propertyGetter` causes this function to
- `compareByInspect.js` (~193 tok, small)
- `expectTypes.js` (~367 tok, medium) — Transforms ['lorem', 'ipsum'] into 'a lorem, or an ipsum'
- `flag.js` (~221 tok, medium)
- `getActual.js` (~121 tok, small)
- `getMessage.js` (~350 tok, medium)
- `getOperator.js` (~331 tok, medium)
- `getOwnEnumerableProperties.js` (~184 tok, small)
- `getOwnEnumerablePropertySymbols.js` (~200 tok, medium)
- `getProperties.js` (~204 tok, medium)
- `index.js` (~681 tok, large) — Dependencies that are used for multiple exports are required here only once
- `inspect.js` (~280 tok, medium) — This is (almost) directly from Node.js utils
- `isNaN.js` (~41 tok, tiny)
- `isProxyEnabled.js` (~151 tok, small)
- `objDisplay.js` (~310 tok, medium)
- `overwriteChainableMethod.js` (~542 tok, large)
- `overwriteMethod.js` (~781 tok, large) — Setting the `ssfi` flag to `overwritingMethodWrapper` causes this
- `overwriteProperty.js` (~809 tok, large) — Setting the `ssfi` flag to `overwritingPropertyGetter` causes this
- `proxify.js` (~1374 tok, large) — This check is here because we should not throw errors on Symbol properties
- `test.js` (~120 tok, small)
- `transferFlags.js` (~351 tok, medium)
- `type-detect.js` (~96 tok, small)
### `sdk/ts/mic-map/node_modules/chai/`

- `LICENSE` (~271 tok, medium) — MIT License
- `package.json` (~518 tok, large) — Keys: author, name, type, description, keywords
- `README.md` (~1552 tok, huge) — What is Chai?
- `register-assert.js` (~16 tok, tiny)
- `register-expect.js` (~16 tok, tiny)
- `register-should.js` (~17 tok, tiny)
### `sdk/ts/mic-map/node_modules/check-error/`

- `index.js` (~1049 tok, large) — If `errorLike` is an instance of any error we compare their constructors
- `LICENSE` (~278 tok, medium) — Copyright (c) 2013 Jake Luer <jake@alogicalparadox.com> (http://alogicalparadox.
- `package.json` (~302 tok, medium) — Keys: name, description, keywords, license, author
- `README.md` (~1027 tok, large) — What is Check-Error?
### `sdk/ts/mic-map/node_modules/color-convert/`

- `CHANGELOG.md` (~355 tok, medium) — 1.0.0 - 2016-01-07
- `conversions.js` (~4260 tok, huge) — NOTE: conversions should only return primitive values (i.e. arrays, or
- `index.js` (~427 tok, medium) — Preserve .conversion property if there is one
- `LICENSE` (~272 tok, medium) — Copyright (c) 2011-2016 Heather Arthur <fayearthur@gmail.com>
- `package.json` (~207 tok, medium) — Keys: name, description, version, author, license
- `README.md` (~714 tok, large) — color-convert
- `route.js` (~565 tok, large) — https://jsperf.com/object-keys-vs-for-in-with-closure/3
### `sdk/ts/mic-map/node_modules/color-name/`

- `index.js` (~1155 tok, large)
- `LICENSE` (~272 tok, medium) — The MIT License (MIT)
- `package.json` (~152 tok, small) — Keys: name, version, description, main, files
- `README.md` (~96 tok, small)
### `sdk/ts/mic-map/node_modules/cross-spawn/`

- `index.js` (~298 tok, medium) — Parse the arguments
### `sdk/ts/mic-map/node_modules/cross-spawn/lib/`

- `enoent.js` (~368 tok, medium) — If emitting "exit" event and exit code is 1, we need to check if
- `parse.js` (~767 tok, large) — Detect & add support for shebangs
### `sdk/ts/mic-map/node_modules/cross-spawn/lib/util/`

- `escape.js` (~346 tok, medium) — See http://www.robvanderwoude.com/escapechars.php
- `readShebang.js` (~138 tok, small) — Read the first 150 bytes from the file
- `resolveCommand.js` (~390 tok, medium) — Worker threads do not have process.chdir()
### `sdk/ts/mic-map/node_modules/cross-spawn/`

- `LICENSE` (~277 tok, medium) — The MIT License (MIT)
- `package.json` (~414 tok, medium) — Keys: name, version, description, keywords, author
- `README.md` (~1030 tok, large) — cross-spawn
### `sdk/ts/mic-map/node_modules/debug/`

- `LICENSE` (~285 tok, medium) — (The MIT License)
- `package.json` (~370 tok, medium) — Keys: name, version, repository, description, keywords
- `README.md` (~5529 tok, huge) — debug
### `sdk/ts/mic-map/node_modules/debug/src/`

- `browser.js` (~1526 tok, huge) — eslint-disable-next-line complexity
- `common.js` (~1729 tok, huge) — Disabled?
- `index.js` (~79 tok, small)
- `node.js` (~1182 tok, large) — Optional dependency (as in, doesn't need to be installed, NOT like optionalDependencies in package.json)
### `sdk/ts/mic-map/node_modules/deep-eql/`

- `index.js` (~4166 tok, huge) — Technically, WeakMap keys can *only* be objects, not primitives.
- `LICENSE` (~278 tok, medium) — Copyright (c) 2013 Jake Luer <jake@alogicalparadox.com> (http://alogicalparadox.
- `package.json` (~479 tok, medium) — Keys: name, version, description, keywords, repository
- `README.md` (~1051 tok, large) — What is Deep-Eql?
### `sdk/ts/mic-map/node_modules/eastasianwidth/`

- `eastasianwidth.js` (~3017 tok, huge) — Split a string considering surrogate-pairs.
- `package.json` (~98 tok, small) — Keys: name, version, description, main, files
- `README.md` (~296 tok, medium) — East Asian Width
### `sdk/ts/mic-map/node_modules/emoji-regex/es2015/`

- `index.d.ts` (~25 tok, tiny)
- `index.js` (~4352 tok, huge) — https://mths.be/emoji
- `RGI_Emoji.d.ts` (~27 tok, tiny)
- `RGI_Emoji.js` (~3506 tok, huge) — https://mths.be/emoji
- `text.d.ts` (~26 tok, tiny)
- `text.js` (~3949 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/emoji-regex/`

- `index.d.ts` (~23 tok, tiny)
- `index.js` (~3934 tok, huge) — https://mths.be/emoji
- `LICENSE-MIT.txt` (~270 tok, medium) — Copyright Mathias Bynens <https://mathiasbynens.be/>
- `package.json` (~333 tok, medium) — Keys: name, version, description, homepage, main
- `README.md` (~1129 tok, large) — emoji-regex [![Build status](https://travis-ci.org/mathiasbynens/emoji-regex.svg?branch=main)](https://travis-ci.org/mat
- `RGI_Emoji.d.ts` (~25 tok, tiny)
- `RGI_Emoji.js` (~3244 tok, huge) — https://mths.be/emoji
- `text.d.ts` (~24 tok, tiny)
- `text.js` (~3617 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/esbuild/`

- `install.js` (~2736 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
### `sdk/ts/mic-map/node_modules/esbuild/lib/`

- `main.d.ts` (~5748 tok, huge) — This is a full copy of the esbuild library in case you need it
- `main.js` (~21914 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
### `sdk/ts/mic-map/node_modules/esbuild/`

- `LICENSE.md` (~268 tok, medium)
### `sdk/ts/mic-map/node_modules/@esbuild/linux-x64/`

- `package.json` (~93 tok, small) — Keys: name, version, description, repository, license
- `README.md` (~36 tok, tiny) — esbuild
### `sdk/ts/mic-map/node_modules/esbuild/`

- `package.json` (~335 tok, medium) — Keys: name, version, description, repository, scripts
- `README.md` (~44 tok, tiny) — esbuild
### `sdk/ts/mic-map/node_modules/es-module-lexer/dist/`

- `lexer.asm.js` (~5140 tok, huge)
- `lexer.cjs` (~3581 tok, huge) — "use strict";var ImportType;exports.initSync=exports.init=exports.ImportType=voi
- `lexer.js` (~3551 tok, huge)
### `sdk/ts/mic-map/node_modules/es-module-lexer/`

- `lexer.js` (~6578 tok, huge) — Note: parsing is based on the _assumption_ that the source is already valid
- `LICENSE` (~274 tok, medium) — MIT License
- `package.json` (~366 tok, medium) — Keys: name, version, description, main, module
- `README.md` (~2511 tok, huge) — ES Module Lexer
### `sdk/ts/mic-map/node_modules/es-module-lexer/types/`

- `lexer.d.ts` (~1361 tok, large)
### `sdk/ts/mic-map/node_modules/estree-walker/`

- `LICENSE` (~282 tok, medium) — Copyright (c) 2015-20 [these people](https://github.com/Rich-Harris/estree-walke
- `package.json` (~180 tok, small) — Keys: name, description, version, private, author
- `README.md` (~398 tok, medium) — estree-walker
### `sdk/ts/mic-map/node_modules/estree-walker/src/`

- `async.js` (~869 tok, large) — removed
- `index.js` (~202 tok, medium)
- `sync.js` (~855 tok, large) — removed
- `walker.js` (~349 tok, medium)
### `sdk/ts/mic-map/node_modules/estree-walker/types/`

- `async.d.ts` (~364 tok, medium)
- `index.d.ts` (~215 tok, medium)
- `sync.d.ts` (~352 tok, medium)
- `walker.d.ts` (~338 tok, medium)
### `sdk/ts/mic-map/node_modules/expect-type/dist/`

- `branding.d.ts` (~600 tok, large)
- `branding.js` (~20 tok, tiny)
- `index.d.ts` (~8922 tok, huge)
- `index.js` (~846 tok, large)
- `messages.d.ts` (~1344 tok, large)
- `messages.js` (~282 tok, medium)
- `overloads.d.ts` (~3958 tok, huge)
- `overloads.js` (~20 tok, tiny)
- `utils.d.ts` (~1932 tok, huge)
- `utils.js` (~135 tok, small)
### `sdk/ts/mic-map/node_modules/expect-type/`

- `LICENSE` (~2690 tok, huge) —    Copyright 2024 Misha Kaletsky
- `package.json` (~281 tok, medium) — Keys: name, version, engines, keywords, homepage
- `README.md` (~8831 tok, huge) — expect-type
- `SECURITY.md` (~106 tok, small) — Security Policy
### `sdk/ts/mic-map/node_modules/foreground-child/dist/commonjs/`

- `all-signals.d.ts` (~23 tok, tiny)
- `all-signals.d.ts.map` (~43 tok, tiny) — {"version":3,"file":"all-signals.d.ts","sourceRoot":"","sources":["../../src/all
- `all-signals.js` (~389 tok, medium) — this is the full list of signals that Node will let us do anything with
- `all-signals.js.map` (~558 tok, large) — {"version":3,"file":"all-signals.js","sourceRoot":"","sources":["../../src/all-s
- `index.d.ts` (~705 tok, large)
- `index.d.ts.map` (~481 tok, medium) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~1019 tok, large) — SIGHUP is weird on windows
- `index.js.map` (~2459 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `proxy-signals.d.ts` (~58 tok, small)
- `proxy-signals.d.ts.map` (~56 tok, small) — {"version":3,"file":"proxy-signals.d.ts","sourceRoot":"","sources":["../../src/p
- `proxy-signals.js` (~290 tok, medium) — some signals can only be received, not sent
- `proxy-signals.js.map` (~471 tok, medium) — {"version":3,"file":"proxy-signals.js","sourceRoot":"","sources":["../../src/pro
- `watchdog.d.ts` (~104 tok, small)
- `watchdog.d.ts.map` (~53 tok, small) — {"version":3,"file":"watchdog.d.ts","sourceRoot":"","sources":["../../src/watchd
- `watchdog.js` (~394 tok, medium) — this spawns a child process that listens for SIGHUP when the
- `watchdog.js.map` (~554 tok, large) — {"version":3,"file":"watchdog.js","sourceRoot":"","sources":["../../src/watchdog
### `sdk/ts/mic-map/node_modules/foreground-child/dist/esm/`

- `all-signals.d.ts` (~23 tok, tiny)
- `all-signals.d.ts.map` (~43 tok, tiny) — {"version":3,"file":"all-signals.d.ts","sourceRoot":"","sources":["../../src/all
- `all-signals.js` (~318 tok, medium) — this is the full list of signals that Node will let us do anything with
- `all-signals.js.map` (~564 tok, large) — {"version":3,"file":"all-signals.js","sourceRoot":"","sources":["../../src/all-s
- `index.d.ts` (~705 tok, large)
- `index.d.ts.map` (~481 tok, medium) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~904 tok, large) — SIGHUP is weird on windows
- `index.js.map` (~2481 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `proxy-signals.d.ts` (~58 tok, small)
- `proxy-signals.d.ts.map` (~56 tok, small) — {"version":3,"file":"proxy-signals.d.ts","sourceRoot":"","sources":["../../src/p
- `proxy-signals.js` (~250 tok, medium) — some signals can only be received, not sent
- `proxy-signals.js.map` (~474 tok, medium) — {"version":3,"file":"proxy-signals.js","sourceRoot":"","sources":["../../src/pro
- `watchdog.d.ts` (~104 tok, small)
- `watchdog.d.ts.map` (~53 tok, small) — {"version":3,"file":"watchdog.d.ts","sourceRoot":"","sources":["../../src/watchd
- `watchdog.js` (~354 tok, medium) — this spawns a child process that listens for SIGHUP when the
- `watchdog.js.map` (~556 tok, large) — {"version":3,"file":"watchdog.js","sourceRoot":"","sources":["../../src/watchdog
### `sdk/ts/mic-map/node_modules/foreground-child/`

- `LICENSE` (~194 tok, small) — The ISC License
- `package.json` (~679 tok, large) — Keys: name, version, description, main, types
- `README.md` (~1122 tok, large) — foreground-child
### `sdk/ts/mic-map/node_modules/glob/dist/commonjs/`

- `glob.d.ts` (~3702 tok, huge)
- `glob.d.ts.map` (~1010 tok, large) — {"version":3,"file":"glob.d.ts","sourceRoot":"","sources":["../../src/glob.ts"],
- `glob.js` (~2137 tok, huge) — if no process global, just call it linux.
- `glob.js.map` (~7055 tok, huge) — {"version":3,"file":"glob.js","sourceRoot":"","sources":["../../src/glob.ts"],"n
- `has-magic.d.ts` (~190 tok, small)
- `has-magic.d.ts.map` (~65 tok, small) — {"version":3,"file":"has-magic.d.ts","sourceRoot":"","sources":["../../src/has-m
- `has-magic.js` (~265 tok, medium)
- `has-magic.js.map` (~371 tok, medium) — {"version":3,"file":"has-magic.js","sourceRoot":"","sources":["../../src/has-mag
- `ignore.d.ts` (~204 tok, medium)
- `ignore.d.ts.map` (~222 tok, medium) — {"version":3,"file":"ignore.d.ts","sourceRoot":"","sources":["../../src/ignore.t
- `ignore.js` (~1067 tok, large) — give it a pattern, and it'll be able to tell you if
- `ignore.js.map` (~1868 tok, huge) — {"version":3,"file":"ignore.js","sourceRoot":"","sources":["../../src/ignore.ts"
- `index.d.ts` (~1553 tok, huge)
- `index.d.ts.map` (~1022 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~719 tok, large) — aliases: glob.sync.stream() glob.stream.sync() glob.sync() etc
- `index.js.map` (~2093 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `pattern.d.ts` (~530 tok, large)
- `pattern.d.ts.map` (~329 tok, medium) — {"version":3,"file":"pattern.d.ts","sourceRoot":"","sources":["../../src/pattern
- `pattern.js` (~1825 tok, huge) — this is just a very light wrapper around 2 arrays with an offset index
- `pattern.js.map` (~3348 tok, huge) — {"version":3,"file":"pattern.js","sourceRoot":"","sources":["../../src/pattern.t
- `processor.d.ts` (~534 tok, large)
- `processor.d.ts.map` (~437 tok, medium) — {"version":3,"file":"processor.d.ts","sourceRoot":"","sources":["../../src/proce
- `processor.js` (~2690 tok, huge) — synchronous utility for filtering entries and calculating subwalks
- `processor.js.map` (~4761 tok, huge) — {"version":3,"file":"processor.js","sourceRoot":"","sources":["../../src/process
- `walker.d.ts` (~945 tok, large)
- `walker.d.ts.map` (~1049 tok, large) — {"version":3,"file":"walker.d.ts","sourceRoot":"","sources":["../../src/walker.t
- `walker.js` (~3218 tok, huge) — ignore, always set with maxDepth, but it's optional on the
- `walker.js.map` (~6975 tok, huge) — {"version":3,"file":"walker.js","sourceRoot":"","sources":["../../src/walker.ts"
### `sdk/ts/mic-map/node_modules/glob/dist/esm/`

- `bin.d.mts` (~17 tok, tiny) — #!/usr/bin/env node
- `bin.d.mts.map` (~27 tok, tiny) — {"version":3,"file":"bin.d.mts","sourceRoot":"","sources":["../../src/bin.mts"],
- `bin.mjs` (~3162 tok, huge) — #!/usr/bin/env node
- `bin.mjs.map` (~4645 tok, huge) — {"version":3,"file":"bin.mjs","sourceRoot":"","sources":["../../src/bin.mts"],"n
- `glob.d.ts` (~3702 tok, huge)
- `glob.d.ts.map` (~1010 tok, large) — {"version":3,"file":"glob.d.ts","sourceRoot":"","sources":["../../src/glob.ts"],
- `glob.js` (~2085 tok, huge) — if no process global, just call it linux.
- `glob.js.map` (~7095 tok, huge) — {"version":3,"file":"glob.js","sourceRoot":"","sources":["../../src/glob.ts"],"n
- `has-magic.d.ts` (~190 tok, small)
- `has-magic.d.ts.map` (~65 tok, small) — {"version":3,"file":"has-magic.d.ts","sourceRoot":"","sources":["../../src/has-m
- `has-magic.js` (~230 tok, medium)
- `has-magic.js.map` (~374 tok, medium) — {"version":3,"file":"has-magic.js","sourceRoot":"","sources":["../../src/has-mag
- `ignore.d.ts` (~204 tok, medium)
- `ignore.d.ts.map` (~222 tok, medium) — {"version":3,"file":"ignore.d.ts","sourceRoot":"","sources":["../../src/ignore.t
- `ignore.js` (~1026 tok, large) — give it a pattern, and it'll be able to tell you if
- `ignore.js.map` (~1879 tok, huge) — {"version":3,"file":"ignore.js","sourceRoot":"","sources":["../../src/ignore.ts"
- `index.d.ts` (~1553 tok, huge)
- `index.d.ts.map` (~1022 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~412 tok, medium) — aliases: glob.sync.stream() glob.stream.sync() glob.sync() etc
- `index.js.map` (~2101 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `pattern.d.ts` (~530 tok, large)
- `pattern.d.ts.map` (~329 tok, medium) — {"version":3,"file":"pattern.d.ts","sourceRoot":"","sources":["../../src/pattern
- `pattern.js` (~1791 tok, huge) — this is just a very light wrapper around 2 arrays with an offset index
- `pattern.js.map` (~3352 tok, huge) — {"version":3,"file":"pattern.js","sourceRoot":"","sources":["../../src/pattern.t
- `processor.d.ts` (~534 tok, large)
- `processor.d.ts.map` (~437 tok, medium) — {"version":3,"file":"processor.d.ts","sourceRoot":"","sources":["../../src/proce
- `processor.js` (~2614 tok, huge) — synchronous utility for filtering entries and calculating subwalks
- `processor.js.map` (~4760 tok, huge) — {"version":3,"file":"processor.js","sourceRoot":"","sources":["../../src/process
- `walker.d.ts` (~945 tok, large)
- `walker.d.ts.map` (~1049 tok, large) — {"version":3,"file":"walker.d.ts","sourceRoot":"","sources":["../../src/walker.t
- `walker.js` (~3143 tok, huge) — ignore, always set with maxDepth, but it's optional on the
- `walker.js.map` (~6989 tok, huge) — {"version":3,"file":"walker.js","sourceRoot":"","sources":["../../src/walker.ts"
### `sdk/ts/mic-map/node_modules/glob/`

- `LICENSE` (~194 tok, small) — The ISC License
### `sdk/ts/mic-map/node_modules/glob/node_modules/balanced-match/.github/`

- `FUNDING.yml` (~14 tok, tiny) — tidelift: "npm/balanced-match"
### `sdk/ts/mic-map/node_modules/glob/node_modules/balanced-match/`

- `index.js` (~305 tok, medium)
- `LICENSE.md` (~274 tok, medium)
- `package.json` (~268 tok, medium) — Keys: name, description, version, repository, homepage
- `README.md` (~876 tok, large) — balanced-match
### `sdk/ts/mic-map/node_modules/glob/node_modules/brace-expansion/.github/`

- `FUNDING.yml` (~14 tok, tiny) — tidelift: "npm/brace-expansion"
### `sdk/ts/mic-map/node_modules/glob/node_modules/brace-expansion/`

- `index.js` (~1295 tok, large) — Basically just str.split(","), but handling cases
- `LICENSE` (~274 tok, medium) — MIT License
- `package.json` (~284 tok, medium) — Keys: name, description, version, repository, homepage
- `README.md` (~1063 tok, large) — brace-expansion
### `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/dist/commonjs/`

- `assert-valid-pattern.d.ts` (~29 tok, tiny)
- `assert-valid-pattern.d.ts.map` (~50 tok, small) — {"version":3,"file":"assert-valid-pattern.d.ts","sourceRoot":"","sources":["../.
- `assert-valid-pattern.js` (~123 tok, small)
- `assert-valid-pattern.js.map` (~206 tok, medium) — {"version":3,"file":"assert-valid-pattern.js","sourceRoot":"","sources":["../../
- `ast.d.ts` (~199 tok, small)
- `ast.d.ts.map` (~217 tok, medium) — {"version":3,"file":"ast.d.ts","sourceRoot":"","sources":["../../src/ast.ts"],"n
- `ast.js` (~6947 tok, huge) — parse a single path portion
- `ast.js.map` (~12950 tok, huge) — {"version":3,"file":"ast.js","sourceRoot":"","sources":["../../src/ast.ts"],"nam
- `brace-expressions.d.ts` (~63 tok, small)
- `brace-expressions.d.ts.map` (~80 tok, small) — {"version":3,"file":"brace-expressions.d.ts","sourceRoot":"","sources":["../../s
- `brace-expressions.js` (~1441 tok, large) — translate the various posix character classes into unicode properties
- `brace-expressions.js.map` (~2638 tok, huge) — {"version":3,"file":"brace-expressions.js","sourceRoot":"","sources":["../../src
- `escape.d.ts` (~162 tok, small)
- `escape.d.ts.map` (~65 tok, small) — {"version":3,"file":"escape.d.ts","sourceRoot":"","sources":["../../src/escape.t
- `escape.js` (~242 tok, medium) — don't need to escape +@! because we escape the parens
- `escape.js.map` (~345 tok, medium) — {"version":3,"file":"escape.js","sourceRoot":"","sources":["../../src/escape.ts"
- `index.d.ts` (~1005 tok, large)
- `index.d.ts.map` (~844 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~9964 tok, huge) — shortcut: comments match nothing.
- `index.js.map` (~18737 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `unescape.d.ts` (~197 tok, small)
- `unescape.d.ts.map` (~68 tok, small) — {"version":3,"file":"unescape.d.ts","sourceRoot":"","sources":["../../src/unesca
- `unescape.js` (~244 tok, medium)
- `unescape.js.map` (~351 tok, medium) — {"version":3,"file":"unescape.js","sourceRoot":"","sources":["../../src/unescape
### `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/dist/esm/`

- `assert-valid-pattern.d.ts` (~29 tok, tiny)
- `assert-valid-pattern.d.ts.map` (~50 tok, small) — {"version":3,"file":"assert-valid-pattern.d.ts","sourceRoot":"","sources":["../.
- `assert-valid-pattern.js` (~84 tok, small)
- `assert-valid-pattern.js.map` (~202 tok, medium) — {"version":3,"file":"assert-valid-pattern.js","sourceRoot":"","sources":["../../
- `ast.d.ts` (~199 tok, small)
- `ast.d.ts.map` (~217 tok, medium) — {"version":3,"file":"ast.d.ts","sourceRoot":"","sources":["../../src/ast.ts"],"n
- `ast.js` (~6889 tok, huge) — parse a single path portion
- `ast.js.map` (~12954 tok, huge) — {"version":3,"file":"ast.js","sourceRoot":"","sources":["../../src/ast.ts"],"nam
- `brace-expressions.d.ts` (~63 tok, small)
- `brace-expressions.d.ts.map` (~80 tok, small) — {"version":3,"file":"brace-expressions.d.ts","sourceRoot":"","sources":["../../s
- `brace-expressions.js` (~1408 tok, large) — translate the various posix character classes into unicode properties
- `brace-expressions.js.map` (~2634 tok, huge) — {"version":3,"file":"brace-expressions.js","sourceRoot":"","sources":["../../src
- `escape.d.ts` (~162 tok, small)
- `escape.d.ts.map` (~65 tok, small) — {"version":3,"file":"escape.d.ts","sourceRoot":"","sources":["../../src/escape.t
- `escape.js` (~212 tok, medium) — don't need to escape +@! because we escape the parens
- `escape.js.map` (~341 tok, medium) — {"version":3,"file":"escape.js","sourceRoot":"","sources":["../../src/escape.ts"
- `index.d.ts` (~1005 tok, large)
- `index.d.ts.map` (~844 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~9591 tok, huge) — shortcut: comments match nothing.
- `index.js.map` (~18743 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `unescape.d.ts` (~197 tok, small)
- `unescape.d.ts.map` (~68 tok, small) — {"version":3,"file":"unescape.d.ts","sourceRoot":"","sources":["../../src/unesca
- `unescape.js` (~212 tok, medium)
- `unescape.js.map` (~348 tok, medium) — {"version":3,"file":"unescape.js","sourceRoot":"","sources":["../../src/unescape
### `sdk/ts/mic-map/node_modules/glob/node_modules/minimatch/`

- `LICENSE` (~194 tok, small) — The ISC License
- `package.json` (~492 tok, medium) — Keys: author, name, description, version, repository
- `README.md` (~4631 tok, huge) — minimatch
### `sdk/ts/mic-map/node_modules/glob/`

- `package.json` (~651 tok, large) — Keys: author, publishConfig, name, description, version
- `README.md` (~12096 tok, huge) — Glob
### `sdk/ts/mic-map/node_modules/has-flag/`

- `index.d.ts` (~171 tok, small) — foo.ts
- `index.js` (~83 tok, small)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~174 tok, small) — Keys: name, version, description, license, repository
- `readme.md` (~400 tok, medium) — has-flag [![Build Status](https://travis-ci.org/sindresorhus/has-flag.svg?branch=master)](https://travis-ci.org/sindreso
### `sdk/ts/mic-map/node_modules/html-escaper/cjs/`

- `index.js` (~450 tok, medium)
- `package.json` (~5 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/html-escaper/esm/`

- `index.js` (~437 tok, medium)
### `sdk/ts/mic-map/node_modules/html-escaper/`

- `index.js` (~494 tok, medium)
- `LICENSE.txt` (~273 tok, medium) — Copyright (C) 2017-present by Andrea Giammarchi - @WebReflection
- `min.js` (~114 tok, small)
- `package.json` (~305 tok, medium) — Keys: name, version, description, main, unpkg
- `README.md` (~1077 tok, large) — html-escaper [![Build Status](https://travis-ci.org/WebReflection/html-escaper.svg?branch=master)](https://travis-ci.org
### `sdk/ts/mic-map/node_modules/html-escaper/test/`

- `index.js` (~115 tok, small)
- `package.json` (~5 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/@isaacs/cliui/build/`

- `index.cjs` (~2600 tok, huge) — 'use strict';
- `index.d.cts` (~263 tok, medium) — interface UIOptions {
### `sdk/ts/mic-map/node_modules/@isaacs/cliui/build/lib/`

- `index.js` (~2525 tok, huge) — simple heuristic for layout, make sure the
### `sdk/ts/mic-map/node_modules/@isaacs/cliui/`

- `index.mjs` (~75 tok, small) — // Bootstrap cliui with ESM dependencies:
- `LICENSE.txt` (~183 tok, small) — Copyright (c) 2015, Contributors
- `package.json` (~541 tok, large) — Keys: name, version, description, main, exports
- `README.md` (~764 tok, large) — @isaacs/cliui
### `sdk/ts/mic-map/node_modules/isexe/`

- `index.js` (~298 tok, medium) — ignore EACCES because that just means we aren't allowed to run it
- `LICENSE` (~192 tok, small) — The ISC License
- `mode.js` (~228 tok, medium)
- `.npmignore` (~6 tok, tiny) — .nyc_output/
- `package.json` (~197 tok, small) — Keys: name, version, description, main, directories
- `README.md` (~349 tok, medium) — isexe
### `sdk/ts/mic-map/node_modules/isexe/test/`

- `basic.js` (~1249 tok, large) — with a pathExt of '', any filename is fine.
### `sdk/ts/mic-map/node_modules/isexe/`

- `windows.js` (~223 tok, medium)
### `sdk/ts/mic-map/node_modules/is-fullwidth-code-point/`

- `index.d.ts` (~138 tok, small)
- `index.js` (~439 tok, medium) — Code points are derived from:
- `license` (~278 tok, medium) — MIT License
- `package.json` (~185 tok, small) — Keys: name, version, description, license, repository
- `readme.md` (~211 tok, medium) — is-fullwidth-code-point [![Build Status](https://travis-ci.org/sindresorhus/is-fullwidth-code-point.svg?branch=master)](
### `sdk/ts/mic-map/node_modules/@istanbuljs/schema/`

- `CHANGELOG.md` (~763 tok, large) — Changelog
- `default-exclude.js` (~145 tok, small)
- `default-extension.js` (~22 tok, tiny)
- `index.js` (~2743 tok, huge)
- `LICENSE` (~267 tok, medium) — MIT License
- `package.json` (~163 tok, small) — Keys: name, version, description, main, scripts
- `README.md` (~386 tok, medium) — @istanbuljs/schema
### `sdk/ts/mic-map/node_modules/istanbul-lib-coverage/`

- `CHANGELOG.md` (~1975 tok, huge) — Change Log
- `index.js` (~475 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-lib-coverage/lib/`

- `coverage-map.js` (~870 tok, large)
- `coverage-summary.js` (~703 tok, large) — asserts that a data object "looks like" a summary coverage object
- `data-properties.js` (~71 tok, small)
- `file-coverage.js` (~3513 tok, huge) — returns a data object that represents empty coverage
- `percent.js` (~92 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-lib-coverage/`

- `LICENSE` (~372 tok, medium) — Copyright 2012-2015 Yahoo! Inc.
- `package.json` (~275 tok, medium) — Keys: name, version, description, author, main
- `README.md` (~249 tok, medium) — istanbul-lib-coverage
### `sdk/ts/mic-map/node_modules/istanbul-lib-report/`

- `CHANGELOG.md` (~1495 tok, large) — Change Log
- `index.js` (~247 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-lib-report/lib/`

- `context.js` (~1006 tok, large)
- `file-writer.js` (~1236 tok, large) — allow stdout to be captured for tests.
- `path.js` (~928 tok, large) — handle a weird windows case separately
- `report-base.js` (~91 tok, small) — TODO: switch to class private field when targetting node.js 12
- `summarizer-factory.js` (~1839 tok, huge)
- `tree.js` (~953 tok, large)
- `watermarks.js` (~88 tok, small)
- `xml-writer.js` (~604 tok, large)
### `sdk/ts/mic-map/node_modules/istanbul-lib-report/`

- `LICENSE` (~372 tok, medium) — Copyright 2012-2015 Yahoo! Inc.
- `package.json` (~239 tok, medium) — Keys: name, version, description, author, main
- `README.md` (~318 tok, medium) — istanbul-lib-report
### `sdk/ts/mic-map/node_modules/istanbul-lib-source-maps/`

- `CHANGELOG.md` (~3100 tok, huge) — Change Log
- `index.js` (~80 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-lib-source-maps/lib/`

- `get-mapping.js` (~1463 tok, large) — Given the generated location, find the original location of the mapping
- `mapped.js` (~708 tok, large)
- `map-store.js` (~1856 tok, huge)
- `pathutils.js` (~135 tok, small)
- `transformer.js` (~1239 tok, large) — Check if this is an implicit else
