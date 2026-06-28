# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 2950 | **Est. tokens:** ~7,056,250
**Generated:** 2026-06-28 07:15 UTC

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
| `./` | 34 | ~27,189 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~550 |
| `audits/` | 6 | ~607 |
| `bench/` | 1 | ~693 |
| `benches/` | 14 | ~35,873 |
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
| `docs/` | 29 | ~55,868 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,315 |
| `docs/design/` | 3 | ~8,181 |
| `docs/mindcraft/` | 3 | ~6,976 |
| `docs/research/` | 1 | ~117 |
| `docs/rfcs/` | 28 | ~118,265 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 16 | ~33,871 |
| `examples/c/` | 2 | ~400 |
| `examples/compliance/` | 3 | ~5,294 |
| `examples/emit_ir/` | 5 | ~13,648 |
| `examples/lexer/` | 6 | ~8,888 |
| `examples/mindc_mind/` | 31 | ~69,506 |
| `examples/native/` | 2 | ~71 |
| `examples/parser/` | 5 | ~17,923 |
| `examples/typecheck/` | 5 | ~14,553 |
| `examples/zoo/` | 6 | ~13,400 |
| `experiments/global-vs-local/` | 7 | ~6,492 |
| `.githooks/` | 1 | ~255 |
| `.github/` | 3 | ~148 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 8 | ~9,040 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 1 | ~16,548 |
| `scripts/` | 7 | ~9,074 |
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
| `src/` | 7 | ~17,517 |
| `src/ast/` | 1 | ~7,430 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 3 | ~27,431 |
| `src/build/` | 2 | ~11,359 |
| `src/cache/` | 4 | ~3,525 |
| `src/check/` | 3 | ~9,753 |
| `src/deps/` | 1 | ~8,087 |
| `src/diagnostics/` | 1 | ~2,230 |
| `src/distributed/` | 6 | ~7,433 |
| `src/doc/` | 3 | ~10,474 |
| `src/eval/` | 12 | ~57,998 |
| `src/eval/stdlib/` | 2 | ~8,529 |
| `src/exec/` | 3 | ~4,592 |
| `src/ffi/` | 3 | ~3,919 |
| `src/fmt/` | 3 | ~19,662 |
| `src/ir/` | 4 | ~28,423 |
| `src/ir/compact/` | 3 | ~15,194 |
| `src/ir/compact/v2/` | 8 | ~37,747 |
| `src/ir/compact/v3/` | 4 | ~45,452 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~9,211 |
| `src/mlir/` | 3 | ~5,415 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 4 | ~9,649 |
| `src/package/` | 2 | ~1,668 |
| `src/parser/` | 1 | ~3,811 |
| `src/project/` | 3 | ~27,842 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 1 | ~5,999 |
| `src/type_checker/` | 1 | ~8,655 |
| `src/types/` | 4 | ~3,336 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 23 | ~109,905 |
| `tests/` | 247 | ~396,019 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 1 | ~668 |
| `tests/conformance/cpu_baseline/` | 9 | ~171 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4,052 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~590 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~305 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~755 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~649 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~558 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~838 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~536 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~461 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~730 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1,238 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~688 |
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
- `Cargo.toml` (~1686 tok, huge) — [package]
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
- `Mind.toml` (~108 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~5364 tok, huge) — MIND — Machine Intelligence Native Design
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

- `rules.mind` (~463 tok, medium) — mind (language compiler / runtime root) architectural-governance rules
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

- `benchmarking.md` (~1903 tok, huge) — Benchmarking methodology — tiers and comparable metrics
### `docs/benchmarks/`

- `compiler_performance.md` (~4721 tok, huge) — MIND Compiler Performance Benchmarks
### `docs/`

- `benchmarks.md` (~805 tok, large) — Benchmarks
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

- `determinism.md` (~1851 tok, huge) — The Determinism Contract
- `errors.md` (~701 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
- `gpu.md` (~387 tok, medium) — GPU backend profile
- `install.md` (~1011 tok, large) — Installing mindc
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1485 tok, large) — IR stability contract
### `docs/mindcraft/`

- `fmt.md` (~2239 tok, huge) — `mindc fmt` — Canonical Formatter Reference
- `phase2-implementation-plan.md` (~2209 tok, huge) — Mindcraft Phase 2 — Implementation Plan
- `rfc0010-phase-ghi-migration-plan.md` (~2528 tok, huge) — RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)
### `docs/`

- `mlir-lowering.md` (~286 tok, medium) — MLIR lowering pipeline (public)
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `optimization-frontier.md` (~11269 tok, huge) — MIND Optimization Frontier
- `performance.md` (~742 tok, large) — Performance Guide
- `README.md` (~162 tok, small) — MIND Documentation
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
### `docs/research/`

- `README.md` (~117 tok, small) — Research notes
### `docs/rfcs/`

- `0000-template.md` (~627 tok, large) — RFC 0000: [Title]
- `0001-bitnet-native-support.md` (~3191 tok, huge) — RFC 0001: Native BitNet Support — `tri` and `q16_16` Types
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
- `0021-canonical-ir-unification.md` (~4240 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2083 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `odc-language-primitives.md` (~422 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

- `roadmap.md` (~14697 tok, huge) — Roadmap
- `runs-burndown-roadmap.md` (~3176 tok, huge) — MIND RUNS Burndown Roadmap
- `security.md` (~1331 tok, large) — Security Guide
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
- `sparse-tensor-types.md` (~740 tok, large) — Sparse Tensor Types
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~953 tok, large) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `type-system.md` (~947 tok, large) — Type System
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
### `examples/mindc_mind/`

- `bootstrap_smoke.py` (~2329 tok, huge)
- `collect_field_strings_smoke.py` (~1161 tok, large)
- `cutover_coverage_measure.py` (~2238 tok, huge)
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
- `multi_let_smoke.py` (~1499 tok, large)
- `native_self_host_smoke.sh` (~800 tok, large) — native_self_host_smoke.sh — the NATIVE BOOTSTRAP FIXED POINT gate (#14).
- `now_ns_smoke.py` (~678 tok, large) — # Copyright 2025 STARGA Inc.
- `param_types_smoke.py` (~1273 tok, large)
- `self_host_body_smoke.py` (~2995 tok, huge)
- `self_host_mlir_smoke.py` (~1736 tok, huge)
- `self_host_native_driver_main.frag.mind` (~277 tok, medium) — ===== Native self-host driver (appended; NOT part of canonical main.mind) =====
- `self_host_native_elf_smoke.py` (~6964 tok, huge)
- `sha256_hash_smoke.py` (~732 tok, large) — # Copyright 2025 STARGA Inc.
- `struct_fields_smoke.py` (~1076 tok, large)
- `struct_lit_smoke.py` (~1923 tok, huge)
- `unified_dispatch_smoke.py` (~1553 tok, huge)
- `validate_real_fns_smoke.py` (~2674 tok, huge)
- `while_struct_smoke.py` (~1031 tok, large)
### `examples/`

- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
### `examples/native/`

- `fib.mind` (~34 tok, tiny) — fn fib(n: i64) -> i64 {
- `loop.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/parser/`

- `bootstrap_smoke.py` (~5544 tok, huge)
- `EXPECTED.md` (~2140 tok, huge) — Phase 6.2 — Expected AST Tree
- `fixture.mind` (~160 tok, small) — Phase 6.2 parser smoke fixture.
- `main.mind` (~7825 tok, huge) — examples/parser/main.mind — RFC 0005 Phase 6.2 self-host parser seed.
- `README.md` (~2254 tok, huge) — RFC 0005 Phase 6.2 — Self-Host Parser Seed
### `examples/`

- `policy.mind` (~1221 tok, large) — policy.mind — v0.1 Execution Boundary Kernel
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

- `conv_classifier.mind` (~2621 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `linear_regression.mind` (~1423 tok, large) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `logistic_classifier.mind` (~1624 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `mlp_mnist.mind` (~2471 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `README.md` (~1191 tok, large) — MIND Model Zoo
- `transformer_block.mind` (~4070 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
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
- `ci.yml` (~4409 tok, huge) — name: CI
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

- `mind_intrinsics.c` (~16548 tok, huge) — Copyright 2025 STARGA Inc.
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
- `transform-utils.js` (~120 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-lib-source-maps/`

- `LICENSE` (~371 tok, medium) — Copyright 2015 Yahoo! Inc.
- `package.json` (~249 tok, medium) — Keys: name, version, description, author, main
- `README.md` (~114 tok, small) — istanbul-lib-source-maps
### `sdk/ts/mic-map/node_modules/istanbul-reports/`

- `CHANGELOG.md` (~4478 tok, huge) — Change Log
- `index.js` (~134 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/clover/`

- `index.js` (~1154 tok, large)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/cobertura/`

- `index.js` (~1185 tok, large)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/`

- `annotator.js` (~2600 tok, huge) — eslint-disable-next-line
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/assets/`

- `base.css` (~1349 tok, large) — body, html {
- `block-navigation.js` (~659 tok, large) — Classes of code we would like to highlight in the file view
- `sorter.js` (~1678 tok, huge) — returns the summary table element
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/assets/vendor/`

- `prettify.css` (~169 tok, small) — .pln{color:#000}@media screen{.str{color:#080}.kwd{color:#008}.com{color:#800}.t
- `prettify.js` (~4393 tok, huge)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html/`

- `index.js` (~3580 tok, huge)
- `insertion-text.js` (~780 tok, large)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/assets/`

- `spa.css` (~1010 tok, large) — /* Base */
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/`

- `.babelrc` (~24 tok, tiny) — {
- `index.js` (~1326 tok, large) — force the summarizer to nested for html-spa
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/src/`

- `fileBreadcrumbs.js` (~188 tok, small)
- `filterToggle.js` (~365 tok, medium)
- `flattenToggle.js` (~200 tok, medium)
- `getChildData.js` (~1140 tok, large) — flatten and continue looking underneath
- `index.js` (~1420 tok, large) — The index file for the spa running on the summary page
- `routing.js` (~328 tok, medium)
- `summaryHeader.js` (~488 tok, medium)
- `summaryTableHeader.js` (~979 tok, large)
- `summaryTableLine.js` (~1313 tok, large) — ignore none metrics so they don't change whats shown
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/html-spa/`

- `webpack.config.js` (~113 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/json/`

- `index.js` (~257 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/json-summary/`

- `index.js` (~330 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/lcov/`

- `index.js` (~228 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/lcovonly/`

- `index.js` (~654 tok, large) — Some versions of the instrumenter in the wild populate 'loc'
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/none/`

- `index.js` (~68 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/teamcity/`

- `index.js` (~476 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text/`

- `index.js` (~1973 tok, huge)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text-lcov/`

- `index.js` (~89 tok, small)
### `sdk/ts/mic-map/node_modules/istanbul-reports/lib/text-summary/`

- `index.js` (~433 tok, medium)
### `sdk/ts/mic-map/node_modules/istanbul-reports/`

- `LICENSE` (~372 tok, medium) — Copyright 2012-2015 Yahoo! Inc.
- `package.json` (~378 tok, medium) — Keys: name, version, description, author, main
- `README.md` (~93 tok, small) — istanbul-reports
### `sdk/ts/mic-map/node_modules/jackspeak/dist/commonjs/`

- `index.d.ts` (~3102 tok, huge)
- `index.d.ts.map` (~2139 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~9123 tok, huge) — it's a tiny API, just cast it inline, it's fine
- `index.js.map` (~20114 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `parse-args-cjs.cjs.map` (~387 tok, medium) — {"version":3,"file":"parse-args-cjs.cjs","sourceRoot":"","sources":["../../src/p
- `parse-args-cjs.d.cts.map` (~49 tok, tiny) — {"version":3,"file":"parse-args-cjs.d.cts","sourceRoot":"","sources":["../../src
- `parse-args.d.ts` (~41 tok, tiny)
- `parse-args.js` (~444 tok, medium)
### `sdk/ts/mic-map/node_modules/jackspeak/dist/esm/`

- `index.d.ts` (~3108 tok, huge)
- `index.d.ts.map` (~2139 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~8980 tok, huge) — it's a tiny API, just cast it inline, it's fine
- `index.js.map` (~20115 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `parse-args.d.ts` (~46 tok, tiny)
- `parse-args.d.ts.map` (~47 tok, tiny) — {"version":3,"file":"parse-args.d.ts","sourceRoot":"","sources":["../../src/pars
- `parse-args.js` (~181 tok, small) — Ignore because we will clobber it for commonjs
- `parse-args.js.map` (~446 tok, medium) — {"version":3,"file":"parse-args.js","sourceRoot":"","sources":["../../src/parse-
### `sdk/ts/mic-map/node_modules/jackspeak/`

- `LICENSE.md` (~388 tok, medium) — Blue Oak Model License
- `package.json` (~551 tok, large) — Keys: name, publishConfig, version, description, tshy
- `README.md` (~2850 tok, huge) — jackspeak
### `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/dist/`

- `gen-mapping.mjs` (~1993 tok, huge) — // src/set-array.ts
- `gen-mapping.mjs.map` (~1480 tok, large) — {
- `gen-mapping.umd.js` (~2793 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
- `gen-mapping.umd.js.map` (~1556 tok, huge) — {
### `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/dist/types/`

- `gen-mapping.d.ts` (~957 tok, large)
- `set-array.d.ts` (~305 tok, medium)
- `sourcemap-segment.d.ts` (~131 tok, small)
- `types.d.ts` (~250 tok, medium)
### `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/`

- `LICENSE` (~270 tok, medium) — Copyright 2024 Justin Ridgewell <justin@ridgewell.name>
- `package.json` (~551 tok, large) — Keys: name, version, description, keywords, main
- `README.md` (~1877 tok, huge) — @jridgewell/gen-mapping
### `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/src/`

- `gen-mapping.ts` (~4283 tok, huge) — encodeGeneratedRanges,
- `set-array.ts` (~591 tok, large) — The key may or may not be present. If it is present, it's a number.
- `sourcemap-segment.ts` (~120 tok, small)
- `types.ts` (~360 tok, medium) — import type { GeneratedRange, OriginalScope } from '@jridgewell/sourcemap-codec';
### `sdk/ts/mic-map/node_modules/@jridgewell/gen-mapping/types/`

- `gen-mapping.d.cts` (~968 tok, large) — import type { SourceMapInput } from '@jridgewell/trace-mapping';
- `gen-mapping.d.cts.map` (~651 tok, large) — {"version":3,"file":"gen-mapping.d.ts","sourceRoot":"","sources":["../src/gen-ma
- `gen-mapping.d.mts` (~968 tok, large) — import type { SourceMapInput } from '@jridgewell/trace-mapping';
- `gen-mapping.d.mts.map` (~651 tok, large) — {"version":3,"file":"gen-mapping.d.ts","sourceRoot":"","sources":["../src/gen-ma
- `set-array.d.cts` (~314 tok, medium) — type Key = string | number | symbol;
- `set-array.d.cts.map` (~190 tok, small) — {"version":3,"file":"set-array.d.ts","sourceRoot":"","sources":["../src/set-arra
- `set-array.d.mts` (~314 tok, medium) — type Key = string | number | symbol;
- `set-array.d.mts.map` (~190 tok, small) — {"version":3,"file":"set-array.d.ts","sourceRoot":"","sources":["../src/set-arra
- `sourcemap-segment.d.cts` (~142 tok, small) — type GeneratedColumn = number;
- `sourcemap-segment.d.cts.map` (~150 tok, small) — {"version":3,"file":"sourcemap-segment.d.ts","sourceRoot":"","sources":["../src/
- `sourcemap-segment.d.mts` (~142 tok, small) — type GeneratedColumn = number;
- `sourcemap-segment.d.mts.map` (~150 tok, small) — {"version":3,"file":"sourcemap-segment.d.ts","sourceRoot":"","sources":["../src/
- `types.d.cts` (~259 tok, medium) — import type { SourceMapSegment } from './sourcemap-segment.cts';
- `types.d.cts.map` (~298 tok, medium) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
- `types.d.mts` (~259 tok, medium) — import type { SourceMapSegment } from './sourcemap-segment.mts';
- `types.d.mts.map` (~298 tok, medium) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
### `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/dist/`

- `resolve-uri.mjs` (~2153 tok, huge) — // Matches the scheme of a URL, eg "http://"
- `resolve-uri.mjs.map` (~3566 tok, huge) — {"version":3,"file":"resolve-uri.mjs","sources":["../src/resolve-uri.ts"],"sourc
- `resolve-uri.umd.js` (~2467 tok, huge) — Matches the scheme of a URL, eg "http://"
- `resolve-uri.umd.js.map` (~3575 tok, huge) — {"version":3,"file":"resolve-uri.umd.js","sources":["../src/resolve-uri.ts"],"so
### `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/dist/types/`

- `resolve-uri.d.ts` (~38 tok, tiny)
### `sdk/ts/mic-map/node_modules/@jridgewell/resolve-uri/`

- `LICENSE` (~270 tok, medium) — Copyright 2019 Justin Ridgewell <jridgewell@google.com>
- `package.json` (~516 tok, large) — Keys: name, version, description, keywords, author
- `README.md` (~707 tok, large) — @jridgewell/resolve-uri
### `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/dist/`

- `sourcemap-codec.mjs` (~3218 tok, huge) — // src/vlq.ts
- `sourcemap-codec.mjs.map` (~2403 tok, huge) — {
- `sourcemap-codec.umd.js` (~3648 tok, huge) — src/sourcemap-codec.ts
- `sourcemap-codec.umd.js.map` (~2425 tok, huge) — {
### `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/`

- `LICENSE` (~270 tok, medium) — Copyright 2024 Justin Ridgewell <justin@ridgewell.name>
- `package.json` (~538 tok, large) — Keys: name, version, description, keywords, main
- `README.md` (~2513 tok, huge) — @jridgewell/sourcemap-codec
### `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/src/`

- `scopes.ts` (~2429 tok, huge)
- `sourcemap-codec.ts` (~801 tok, large)
- `strings.ts` (~376 tok, medium) — Provide a fallback for older environments.
- `vlq.ts` (~347 tok, medium)
### `sdk/ts/mic-map/node_modules/@jridgewell/sourcemap-codec/types/`

- `scopes.d.cts` (~288 tok, medium) — type Line = number;
- `scopes.d.cts.map` (~337 tok, medium) — {"version":3,"file":"scopes.d.ts","sourceRoot":"","sources":["../src/scopes.ts"]
- `scopes.d.mts` (~288 tok, medium) — type Line = number;
- `scopes.d.mts.map` (~337 tok, medium) — {"version":3,"file":"scopes.d.ts","sourceRoot":"","sources":["../src/scopes.ts"]
- `sourcemap-codec.d.cts` (~175 tok, small) — export { decodeOriginalScopes, encodeOriginalScopes, decodeGeneratedRanges, enco
- `sourcemap-codec.d.cts.map` (~177 tok, small) — {"version":3,"file":"sourcemap-codec.d.ts","sourceRoot":"","sources":["../src/so
- `sourcemap-codec.d.mts` (~175 tok, small) — export { decodeOriginalScopes, encodeOriginalScopes, decodeGeneratedRanges, enco
- `sourcemap-codec.d.mts.map` (~177 tok, small) — {"version":3,"file":"sourcemap-codec.d.ts","sourceRoot":"","sources":["../src/so
- `strings.d.cts` (~91 tok, small) — export declare class StringWriter {
- `strings.d.cts.map` (~106 tok, small) — {"version":3,"file":"strings.d.ts","sourceRoot":"","sources":["../src/strings.ts
- `strings.d.mts` (~91 tok, small) — export declare class StringWriter {
- `strings.d.mts.map` (~106 tok, small) — {"version":3,"file":"strings.d.ts","sourceRoot":"","sources":["../src/strings.ts
- `vlq.d.cts` (~111 tok, small) — import type { StringReader, StringWriter } from './strings.cts';
- `vlq.d.cts.map` (~113 tok, small) — {"version":3,"file":"vlq.d.ts","sourceRoot":"","sources":["../src/vlq.ts"],"name
- `vlq.d.mts` (~111 tok, small) — import type { StringReader, StringWriter } from './strings.mts';
- `vlq.d.mts.map` (~113 tok, small) — {"version":3,"file":"vlq.d.ts","sourceRoot":"","sources":["../src/vlq.ts"],"name
### `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/dist/`

- `trace-mapping.mjs` (~3834 tok, huge) — // src/trace-mapping.ts
- `trace-mapping.mjs.map` (~2802 tok, huge) — {
- `trace-mapping.umd.js` (~4668 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
- `trace-mapping.umd.js.map` (~2893 tok, huge) — {
### `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/`

- `LICENSE` (~270 tok, medium) — Copyright 2024 Justin Ridgewell <justin@ridgewell.name>
- `package.json` (~563 tok, large) — Keys: name, version, description, keywords, main
- `README.md` (~3605 tok, huge) — @jridgewell/trace-mapping
### `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/src/`

- `binary-search.ts` (~699 tok, large) — lastIndex may be -1 if the previous needle was not found.
- `by-source.ts` (~311 tok, medium) — Rebuilds the original source files, with mappings that are ordered by source line/column instead
- `flatten-map.ts` (~1355 tok, large) — We can only add so many lines before we step into the range that the next section's map
- `resolve.ts` (~175 tok, small) — The sourceRoot is always treated as a directory, if it's not empty.
- `sort.ts` (~362 tok, medium) — If we own the array (meaning we parsed it from JSON), then we're free to directly mutate it. If
- `sourcemap-segment.ts` (~167 tok, small)
- `strip-filename.ts` (~64 tok, small)
- `trace-mapping.ts` (~3854 tok, huge) — It's common for parent source maps to have pointers to lines that have no
- `types.ts` (~792 tok, large)
### `sdk/ts/mic-map/node_modules/@jridgewell/trace-mapping/types/`

- `binary-search.d.cts` (~397 tok, medium) — import type { SourceMapSegment, ReverseSegment } from './sourcemap-segment.cts';
- `binary-search.d.cts.map` (~226 tok, medium) — {"version":3,"file":"binary-search.d.ts","sourceRoot":"","sources":["../src/bina
- `binary-search.d.mts` (~397 tok, medium) — import type { SourceMapSegment, ReverseSegment } from './sourcemap-segment.mts';
- `binary-search.d.mts.map` (~226 tok, medium) — {"version":3,"file":"binary-search.d.ts","sourceRoot":"","sources":["../src/bina
- `by-source.d.cts` (~68 tok, small) — import type { ReverseSegment, SourceMapSegment } from './sourcemap-segment.cts';
- `by-source.d.cts.map` (~83 tok, small) — {"version":3,"file":"by-source.d.ts","sourceRoot":"","sources":["../src/by-sourc
- `by-source.d.mts` (~68 tok, small) — import type { ReverseSegment, SourceMapSegment } from './sourcemap-segment.mts';
- `by-source.d.mts.map` (~83 tok, small) — {"version":3,"file":"by-source.d.ts","sourceRoot":"","sources":["../src/by-sourc
- `flatten-map.d.cts` (~96 tok, small) — import { TraceMap } from './trace-mapping.cts';
- `flatten-map.d.cts.map` (~118 tok, small) — {"version":3,"file":"flatten-map.d.ts","sourceRoot":"","sources":["../src/flatte
- `flatten-map.d.mts` (~96 tok, small) — import { TraceMap } from './trace-mapping.mts';
- `flatten-map.d.mts.map` (~118 tok, small) — {"version":3,"file":"flatten-map.d.ts","sourceRoot":"","sources":["../src/flatte
- `resolve.d.cts` (~52 tok, small) — type Resolve = (source: string | null) => string;
- `resolve.d.cts.map` (~73 tok, small) — {"version":3,"file":"resolve.d.ts","sourceRoot":"","sources":["../src/resolve.ts
- `resolve.d.mts` (~52 tok, small) — type Resolve = (source: string | null) => string;
- `resolve.d.mts.map` (~73 tok, small) — {"version":3,"file":"resolve.d.ts","sourceRoot":"","sources":["../src/resolve.ts
- `sort.d.cts` (~82 tok, small) — import type { ReverseSegment, SourceMapSegment } from './sourcemap-segment.cts';
- `sort.d.cts.map` (~95 tok, small) — {"version":3,"file":"sort.d.ts","sourceRoot":"","sources":["../src/sort.ts"],"na
- `sort.d.mts` (~82 tok, small) — import type { ReverseSegment, SourceMapSegment } from './sourcemap-segment.mts';
- `sort.d.mts.map` (~95 tok, small) — {"version":3,"file":"sort.d.ts","sourceRoot":"","sources":["../src/sort.ts"],"na
- `sourcemap-segment.d.cts` (~192 tok, small) — type GeneratedColumn = number;
- `sourcemap-segment.d.cts.map` (~190 tok, small) — {"version":3,"file":"sourcemap-segment.d.ts","sourceRoot":"","sources":["../src/
- `sourcemap-segment.d.mts` (~192 tok, small) — type GeneratedColumn = number;
- `sourcemap-segment.d.mts.map` (~190 tok, small) — {"version":3,"file":"sourcemap-segment.d.ts","sourceRoot":"","sources":["../src/
- `strip-filename.d.cts` (~49 tok, tiny) — /**
- `strip-filename.d.cts.map` (~55 tok, small) — {"version":3,"file":"strip-filename.d.ts","sourceRoot":"","sources":["../src/str
- `strip-filename.d.mts` (~49 tok, tiny) — /**
- `strip-filename.d.mts.map` (~55 tok, small) — {"version":3,"file":"strip-filename.d.ts","sourceRoot":"","sources":["../src/str
- `trace-mapping.d.cts` (~1049 tok, large) — import type { SourceMapSegment } from './sourcemap-segment.cts';
- `trace-mapping.d.cts.map` (~640 tok, large) — {"version":3,"file":"trace-mapping.d.ts","sourceRoot":"","sources":["../src/trac
- `trace-mapping.d.mts` (~1049 tok, large) — import type { SourceMapSegment } from './sourcemap-segment.mts';
- `trace-mapping.d.mts.map` (~640 tok, large) — {"version":3,"file":"trace-mapping.d.ts","sourceRoot":"","sources":["../src/trac
- `types.d.cts` (~786 tok, large) — import type { SourceMapSegment } from './sourcemap-segment.cts';
- `types.d.cts.map` (~892 tok, large) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
- `types.d.mts` (~786 tok, large) — import type { SourceMapSegment } from './sourcemap-segment.mts';
- `types.d.mts.map` (~892 tok, large) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
### `sdk/ts/mic-map/node_modules/loupe/lib/`

- `arguments.d.ts` (~42 tok, tiny)
- `arguments.d.ts.map` (~62 tok, small) — {"version":3,"file":"arguments.d.ts","sourceRoot":"","sources":["../src/argument
- `arguments.js` (~62 tok, small)
- `array.d.ts` (~43 tok, tiny)
- `array.d.ts.map` (~61 tok, small) — {"version":3,"file":"array.d.ts","sourceRoot":"","sources":["../src/array.ts"],"
- `array.js` (~199 tok, small) — Object.keys will always output the Array indices first, so we can slice by
- `bigint.d.ts` (~40 tok, tiny)
- `bigint.d.ts.map` (~57 tok, small) — {"version":3,"file":"bigint.d.ts","sourceRoot":"","sources":["../src/bigint.ts"]
- `bigint.js` (~68 tok, small)
- `class.d.ts` (~48 tok, tiny)
- `class.d.ts.map` (~69 tok, small) — {"version":3,"file":"class.d.ts","sourceRoot":"","sources":["../src/class.ts"],"
- `class.js` (~146 tok, small) — Babel transforms anonymous classes to the name `_class`
- `date.d.ts` (~40 tok, tiny)
- `date.d.ts.map` (~56 tok, small) — {"version":3,"file":"date.d.ts","sourceRoot":"","sources":["../src/date.ts"],"na
- `date.js` (~122 tok, small) — If we need to - truncate the time portion, but never the date
- `error.d.ts` (~40 tok, tiny)
- `error.d.ts.map` (~57 tok, small) — {"version":3,"file":"error.d.ts","sourceRoot":"","sources":["../src/error.ts"],"
- `error.js` (~277 tok, medium)
- `function.d.ts` (~63 tok, small)
- `function.d.ts.map` (~80 tok, small) — {"version":3,"file":"function.d.ts","sourceRoot":"","sources":["../src/function.
- `function.js` (~96 tok, small)
- `helpers.d.ts` (~171 tok, small)
- `helpers.d.ts.map` (~188 tok, small) — {"version":3,"file":"helpers.d.ts","sourceRoot":"","sources":["../src/helpers.ts
- `helpers.js` (~1399 tok, large) — 5 & 6 are blinking
- `html.d.ts` (~110 tok, small)
- `html.d.ts.map` (~126 tok, small) — {"version":3,"file":"html.d.ts","sourceRoot":"","sources":["../src/html.ts"],"na
- `html.js` (~434 tok, medium)
- `index.d.ts` (~107 tok, small)
- `index.d.ts.map` (~108 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../src/index.ts"],"
- `index.js` (~1462 tok, large) — A Symbol polyfill will return `Symbol` not `symbol` from typedetect
- `map.d.ts` (~42 tok, tiny)
- `map.d.ts.map` (~65 tok, small) — {"version":3,"file":"map.d.ts","sourceRoot":"","sources":["../src/map.ts"],"name
- `map.js` (~170 tok, small) — IE11 doesn't support `map.entries()`
- `number.d.ts` (~40 tok, tiny)
- `number.d.ts.map` (~60 tok, small) — {"version":3,"file":"number.d.ts","sourceRoot":"","sources":["../src/number.ts"]
- `number.js` (~161 tok, small)
- `object.d.ts` (~40 tok, tiny)
- `object.d.ts.map` (~60 tok, small) — {"version":3,"file":"object.d.ts","sourceRoot":"","sources":["../src/object.ts"]
- `object.js` (~231 tok, medium)
- `promise.d.ts` (~60 tok, small)
- `promise.d.ts.map` (~76 tok, small) — {"version":3,"file":"promise.d.ts","sourceRoot":"","sources":["../src/promise.ts
- `promise.js` (~20 tok, tiny)
- `regexp.d.ts` (~40 tok, tiny)
- `regexp.d.ts.map` (~60 tok, small) — {"version":3,"file":"regexp.d.ts","sourceRoot":"","sources":["../src/regexp.ts"]
- `regexp.js` (~83 tok, small)
- `set.d.ts` (~40 tok, tiny)
- `set.d.ts.map` (~62 tok, small) — {"version":3,"file":"set.d.ts","sourceRoot":"","sources":["../src/set.ts"],"name
- `set.js` (~104 tok, small) — IE11 doesn't support `Array.from(set)`
- `string.d.ts` (~40 tok, tiny)
- `string.d.ts.map` (~60 tok, small) — {"version":3,"file":"string.d.ts","sourceRoot":"","sources":["../src/string.ts"]
- `string.js` (~207 tok, medium)
- `symbol.d.ts` (~25 tok, tiny)
- `symbol.d.ts.map` (~43 tok, tiny) — {"version":3,"file":"symbol.d.ts","sourceRoot":"","sources":["../src/symbol.ts"]
- `symbol.js` (~53 tok, small)
- `typedarray.d.ts` (~82 tok, small)
- `typedarray.d.ts.map` (~91 tok, small) — {"version":3,"file":"typedarray.d.ts","sourceRoot":"","sources":["../src/typedar
- `typedarray.js` (~430 tok, medium) — We need to special case Node.js' Buffers, which report to be Uint8Array
- `types.d.ts` (~107 tok, small)
- `types.d.ts.map` (~136 tok, small) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
- `types.js` (~3 tok, tiny)
### `sdk/ts/mic-map/node_modules/loupe/`

- `LICENSE` (~276 tok, medium) — (The MIT License)
- `loupe.js` (~4779 tok, huge) — src/index.ts
- `package.json` (~543 tok, large) — Keys: name, version, description, homepage, license
- `README.md` (~551 tok, large) — What is loupe?
### `sdk/ts/mic-map/node_modules/lru-cache/dist/commonjs/`

- `index.d.ts` (~13634 tok, huge)
- `index.d.ts.map` (~3486 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~13755 tok, huge) — This is a little bit ridiculous, tbh.
- `index.min.js` (~4272 tok, huge)
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/lru-cache/dist/esm/`

- `index.d.ts` (~13634 tok, huge)
- `index.d.ts.map` (~3486 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~13724 tok, huge) — This is a little bit ridiculous, tbh.
- `index.min.js` (~4249 tok, huge)
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/lru-cache/`

- `LICENSE` (~194 tok, small) — The ISC License
- `package.json` (~707 tok, large) — Keys: name, publishConfig, description, version, author
- `README.md` (~2777 tok, huge) — lru-cache
### `sdk/ts/mic-map/node_modules/magicast/dist/`

- `helpers.cjs` (~1301 tok, large) — 'use strict';
- `helpers.d.cts` (~630 tok, large) — import { a as Proxified, P as ProxifiedModule, e as ProxifiedFunctionCall, h as 
- `helpers.d.mts` (~630 tok, large) — import { a as Proxified, P as ProxifiedModule, e as ProxifiedFunctionCall, h as 
- `helpers.d.ts` (~630 tok, large)
- `helpers.mjs` (~1239 tok, large) — import { MagicastError, generateCode, parseExpression, builders } from './index.
- `index.d.cts` (~579 tok, large) — import { O as Options, P as ProxifiedModule, a as Proxified, G as GenerateOption
- `index.d.mts` (~579 tok, large) — import { O as Options, P as ProxifiedModule, a as Proxified, G as GenerateOption
- `index.d.ts` (~579 tok, large)
### `sdk/ts/mic-map/node_modules/magicast/dist/shared/`

- `magicast.54e2233d.d.cts` (~2312 tok, huge) — import { Node, ImportSpecifier, ImportDefaultSpecifier, ImportNamespaceSpecifier
- `magicast.54e2233d.d.mts` (~2312 tok, huge) — import { Node, ImportSpecifier, ImportDefaultSpecifier, ImportNamespaceSpecifier
- `magicast.54e2233d.d.ts` (~2312 tok, huge)
### `sdk/ts/mic-map/node_modules/magicast/`

- `helpers.d.ts` (~9 tok, tiny)
- `LICENSE` (~280 tok, medium) — MIT License
- `package.json` (~672 tok, large) — Keys: name, version, description, repository, license
- `README.md` (~1469 tok, large) — 🧀 Magicast
### `sdk/ts/mic-map/node_modules/magic-string/dist/`

- `magic-string.cjs.d.ts` (~2505 tok, huge)
- `magic-string.cjs.js` (~9734 tok, huge) — after split we should save the edit content record into the correct chunk
- `magic-string.cjs.js.map` (~23938 tok, huge) — {"version":3,"file":"magic-string.cjs.js","sources":["../src/BitSet.js","../src/
- `magic-string.es.d.mts` (~2505 tok, huge) — export interface BundleOptions {
- `magic-string.es.mjs` (~9685 tok, huge) — import { encode } from '@jridgewell/sourcemap-codec';
- `magic-string.es.mjs.map` (~23819 tok, huge) — {"version":3,"file":"magic-string.es.mjs","sources":["../src/BitSet.js","../src/
- `magic-string.umd.js` (~10734 tok, huge) — after split we should save the edit content record into the correct chunk
### `sdk/ts/mic-map/node_modules/magic-string/`

- `LICENSE` (~263 tok, medium) — Copyright 2018 Rich Harris
- `package.json` (~460 tok, medium) — Keys: name, version, type, description, keywords
- `README.md` (~3152 tok, huge) — magic-string
### `sdk/ts/mic-map/node_modules/make-dir/`

- `index.d.ts` (~384 tok, medium) — Multiple directories:
- `index.js` (~768 tok, large) — https://github.com/nodejs/node/issues/8987
- `license` (~280 tok, medium) — MIT License
- `package.json` (~275 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~774 tok, large) — make-dir [![codecov](https://codecov.io/gh/sindresorhus/make-dir/branch/main/graph/badge.svg)](https://codecov.io/gh/sin
### `sdk/ts/mic-map/node_modules/minimatch/dist/commonjs/`

- `assert-valid-pattern.d.ts` (~30 tok, tiny)
- `assert-valid-pattern.d.ts.map` (~50 tok, small) — {"version":3,"file":"assert-valid-pattern.d.ts","sourceRoot":"","sources":["../.
- `assert-valid-pattern.js` (~123 tok, small)
- `assert-valid-pattern.js.map` (~208 tok, medium) — {"version":3,"file":"assert-valid-pattern.js","sourceRoot":"","sources":["../../
- `ast.d.ts` (~212 tok, medium)
- `ast.d.ts.map` (~230 tok, medium) — {"version":3,"file":"ast.d.ts","sourceRoot":"","sources":["../../src/ast.ts"],"n
- `ast.js` (~7663 tok, huge) — parse a single path portion
- `ast.js.map` (~14019 tok, huge) — {"version":3,"file":"ast.js","sourceRoot":"","sources":["../../src/ast.ts"],"nam
- `brace-expressions.d.ts` (~63 tok, small)
- `brace-expressions.d.ts.map` (~80 tok, small) — {"version":3,"file":"brace-expressions.d.ts","sourceRoot":"","sources":["../../s
- `brace-expressions.js` (~1436 tok, large) — translate the various posix character classes into unicode properties
- `brace-expressions.js.map` (~2642 tok, huge) — {"version":3,"file":"brace-expressions.js","sourceRoot":"","sources":["../../src
- `escape.d.ts` (~196 tok, small)
- `escape.d.ts.map` (~70 tok, small) — {"version":3,"file":"escape.d.ts","sourceRoot":"","sources":["../../src/escape.t
- `escape.js` (~314 tok, medium) — don't need to escape +@! because we escape the parens
- `escape.js.map` (~473 tok, medium) — {"version":3,"file":"escape.js","sourceRoot":"","sources":["../../src/escape.ts"
- `index.d.ts` (~1840 tok, huge)
- `index.d.ts.map` (~931 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~11285 tok, huge) — shortcut: comments match nothing.
- `index.js.map` (~21333 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `unescape.d.ts` (~239 tok, medium)
- `unescape.d.ts.map` (~73 tok, small) — {"version":3,"file":"unescape.d.ts","sourceRoot":"","sources":["../../src/unesca
- `unescape.js` (~348 tok, medium)
- `unescape.js.map` (~520 tok, large) — {"version":3,"file":"unescape.js","sourceRoot":"","sources":["../../src/unescape
### `sdk/ts/mic-map/node_modules/minimatch/dist/esm/`

- `assert-valid-pattern.d.ts` (~30 tok, tiny)
- `assert-valid-pattern.d.ts.map` (~50 tok, small) — {"version":3,"file":"assert-valid-pattern.d.ts","sourceRoot":"","sources":["../.
- `assert-valid-pattern.js` (~84 tok, small)
- `assert-valid-pattern.js.map` (~204 tok, medium) — {"version":3,"file":"assert-valid-pattern.js","sourceRoot":"","sources":["../../
- `ast.d.ts` (~212 tok, medium)
- `ast.d.ts.map` (~230 tok, medium) — {"version":3,"file":"ast.d.ts","sourceRoot":"","sources":["../../src/ast.ts"],"n
- `ast.js` (~7604 tok, huge) — parse a single path portion
- `ast.js.map` (~14023 tok, huge) — {"version":3,"file":"ast.js","sourceRoot":"","sources":["../../src/ast.ts"],"nam
- `brace-expressions.d.ts` (~63 tok, small)
- `brace-expressions.d.ts.map` (~80 tok, small) — {"version":3,"file":"brace-expressions.d.ts","sourceRoot":"","sources":["../../s
- `brace-expressions.js` (~1403 tok, large) — translate the various posix character classes into unicode properties
- `brace-expressions.js.map` (~2638 tok, huge) — {"version":3,"file":"brace-expressions.js","sourceRoot":"","sources":["../../src
- `escape.d.ts` (~196 tok, small)
- `escape.d.ts.map` (~70 tok, small) — {"version":3,"file":"escape.d.ts","sourceRoot":"","sources":["../../src/escape.t
- `escape.js` (~284 tok, medium) — don't need to escape +@! because we escape the parens
- `escape.js.map` (~469 tok, medium) — {"version":3,"file":"escape.js","sourceRoot":"","sources":["../../src/escape.ts"
- `index.d.ts` (~1840 tok, huge)
- `index.d.ts.map` (~931 tok, large) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~10953 tok, huge) — shortcut: comments match nothing.
- `index.js.map` (~21341 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `unescape.d.ts` (~239 tok, medium)
- `unescape.d.ts.map` (~73 tok, small) — {"version":3,"file":"unescape.d.ts","sourceRoot":"","sources":["../../src/unesca
- `unescape.js` (~316 tok, medium)
- `unescape.js.map` (~516 tok, large) — {"version":3,"file":"unescape.js","sourceRoot":"","sources":["../../src/unescape
### `sdk/ts/mic-map/node_modules/minimatch/`

- `LICENSE.md` (~388 tok, medium) — Blue Oak Model License
- `package.json` (~453 tok, medium) — Keys: author, name, description, version, repository
- `README.md` (~4967 tok, huge) — minimatch
### `sdk/ts/mic-map/node_modules/minipass/dist/commonjs/`

- `index.d.ts` (~4844 tok, huge)
- `index.d.ts.map` (~12016 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~8513 tok, huge) — node core Writable streams have a pipe() method, but it throws
- `index.js.map` (~16741 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/minipass/dist/esm/`

- `index.d.ts` (~4844 tok, huge)
- `index.d.ts.map` (~12016 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~8332 tok, huge) — node core Writable streams have a pipe() method, but it throws
- `index.js.map` (~16729 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/minipass/`

- `LICENSE.md` (~388 tok, medium) — Blue Oak Model License
- `package.json` (~476 tok, medium) — Keys: name, version, description, main, types
- `README.md` (~6790 tok, huge) — minipass
### `sdk/ts/mic-map/node_modules/ms/`

- `index.js` (~756 tok, large)
- `license.md` (~270 tok, medium)
- `package.json` (~183 tok, small) — Keys: name, version, description, repository, main
- `readme.md` (~472 tok, medium) — ms
### `sdk/ts/mic-map/node_modules/nanoid/async/`

- `index.browser.cjs` (~673 tok, large) — let random = async bytes => crypto.getRandomValues(new Uint8Array(bytes))
- `index.browser.js` (~246 tok, medium)
- `index.cjs` (~718 tok, large) — let crypto = require('crypto')
- `index.d.ts` (~377 tok, medium)
- `index.js` (~246 tok, medium)
- `index.native.js` (~205 tok, medium)
- `package.json` (~59 tok, small) — Keys: type, main, module, react-native, browser
### `sdk/ts/mic-map/node_modules/nanoid/bin/`

- `nanoid.cjs` (~283 tok, medium) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/nanoid/`

- `index.browser.cjs` (~709 tok, large) — // This file replaces `index.js` in bundlers like webpack or Rollup,
- `index.browser.js` (~266 tok, medium)
- `index.cjs` (~858 tok, large) — let crypto = require('crypto')
- `index.d.cts` (~563 tok, large) — /**
- `index.d.ts` (~563 tok, large)
- `index.js` (~350 tok, medium)
- `LICENSE` (~274 tok, medium) — The MIT License (MIT)
- `nanoid.js` (~43 tok, tiny)
### `sdk/ts/mic-map/node_modules/nanoid/non-secure/`

- `index.cjs` (~279 tok, medium) — // This alphabet uses `A-Za-z0-9_-` symbols.
- `index.d.ts` (~246 tok, medium)
- `index.js` (~125 tok, small)
- `package.json` (~25 tok, tiny) — Keys: type, main, module, react-native
### `sdk/ts/mic-map/node_modules/nanoid/`

- `package.json` (~571 tok, large) — Keys: name, version, description, keywords, engines
- `README.md` (~388 tok, medium) — Nano ID
### `sdk/ts/mic-map/node_modules/nanoid/url-alphabet/`

- `index.cjs` (~70 tok, small) — // This alphabet uses `A-Za-z0-9_-` symbols.
- `index.js` (~28 tok, tiny)
- `package.json` (~25 tok, tiny) — Keys: type, main, module, react-native
### `sdk/ts/mic-map/node_modules/package-json-from-dist/dist/commonjs/`

- `index.d.ts` (~749 tok, large)
- `index.d.ts.map` (~86 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~1255 tok, large) — inside of node_modules. find the dist directly under package name.
- `index.js.map` (~1693 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/package-json-from-dist/dist/esm/`

- `index.d.ts` (~749 tok, large)
- `index.d.ts.map` (~86 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~1143 tok, large) — inside of node_modules. find the dist directly under package name.
- `index.js.map` (~1699 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/package-json-from-dist/`

- `LICENSE.md` (~441 tok, medium) — Blue Oak Model License
- `package.json` (~423 tok, medium) — Keys: name, version, description, main, exports
- `README.md` (~781 tok, large) — package-json-from-dist
### `sdk/ts/mic-map/node_modules/`

- `.package-lock.json` (~13764 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
### `sdk/ts/mic-map/node_modules/pathe/dist/`

- `index.cjs` (~165 tok, small) — 'use strict';
- `index.d.cts` (~485 tok, medium) — import path$1 from 'node:path';
- `index.d.mts` (~485 tok, medium) — import path$1 from 'node:path';
- `index.d.ts` (~485 tok, medium)
- `index.mjs` (~68 tok, small) — export { h as basename, p as default, d as delimiter, f as dirname, e as extname
### `sdk/ts/mic-map/node_modules/pathe/dist/shared/`

- `pathe.1f0a373c.cjs` (~1695 tok, huge) — 'use strict';
- `pathe.ff20891b.mjs` (~1635 tok, huge) — const _DRIVE_LETTER_START_RE = /^[A-Za-z]:\//;
### `sdk/ts/mic-map/node_modules/pathe/dist/`

- `utils.cjs` (~446 tok, medium) — 'use strict';
- `utils.d.cts` (~71 tok, small) — declare function normalizeAliases(_aliases: Record<string, string>): Record<stri
- `utils.d.mts` (~71 tok, small) — declare function normalizeAliases(_aliases: Record<string, string>): Record<stri
- `utils.d.ts` (~71 tok, small)
- `utils.mjs` (~432 tok, medium) — import { n as normalizeWindowsPath, j as join } from './shared/pathe.ff20891b.mj
### `sdk/ts/mic-map/node_modules/pathe/`

- `LICENSE` (~567 tok, large) — MIT License
- `package.json` (~328 tok, medium) — Keys: name, version, description, repository, license
- `README.md` (~697 tok, large) — 🛣️ pathe
- `utils.d.ts` (~8 tok, tiny)
### `sdk/ts/mic-map/node_modules/path-key/`

- `index.d.ts` (~258 tok, medium) — TODO: Remove this for the next major release, refactor the whole definition to:
- `index.js` (~104 tok, small) — TODO: Remove this for the next major release
- `license` (~278 tok, medium) — MIT License
- `package.json` (~163 tok, small) — Keys: name, version, description, license, repository
- `readme.md` (~337 tok, medium) — path-key [![Build Status](https://travis-ci.org/sindresorhus/path-key.svg?branch=master)](https://travis-ci.org/sindreso
### `sdk/ts/mic-map/node_modules/path-scurry/dist/commonjs/`

- `index.d.ts` (~9951 tok, huge)
- `index.d.ts.map` (~4541 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~16531 tok, huge) — TODO: test perf of fs/promises realpath vs realpathCB,
- `package.json` (~7 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/path-scurry/dist/esm/`

- `index.d.ts` (~9970 tok, huge)
- `index.d.ts.map` (~4541 tok, huge) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~16080 tok, huge) — TODO: test perf of fs/promises realpath vs realpathCB,
- `package.json` (~6 tok, tiny) — Keys: type
### `sdk/ts/mic-map/node_modules/path-scurry/`

- `LICENSE.md` (~388 tok, medium) — Blue Oak Model License
- `package.json` (~543 tok, large) — Keys: name, version, description, author, main
- `README.md` (~5505 tok, huge) — path-scurry
### `sdk/ts/mic-map/node_modules/pathval/`

- `index.js` (~1923 tok, huge) — The `in` operator does not work with primitives.
- `LICENSE` (~274 tok, medium) — MIT License
- `package.json` (~382 tok, medium) — Keys: name, description, homepage, version, keywords
- `README.md` (~1007 tok, large) — What is pathval?
### `sdk/ts/mic-map/node_modules/picocolors/`

- `LICENSE` (~197 tok, small) — ISC License
- `package.json` (~138 tok, small) — Keys: name, version, main, types, browser
- `picocolors.browser.js` (~150 tok, small)
- `picocolors.d.ts` (~35 tok, tiny)
- `picocolors.js` (~666 tok, large)
- `README.md` (~156 tok, small) — picocolors
- `types.d.ts` (~254 tok, medium)
### `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/`

- `CHANGELOG.md` (~1742 tok, huge) — Changelog
- `.editorconfig` (~75 tok, small) — # EditorConfig is awesome: http://EditorConfig.org
### `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/examples/`

- `is-default-value.js` (~192 tok, small) — This example shows how to understand if a default value is used or not.
- `limit-long-syntax.js` (~268 tok, medium) — This is an example of using tokens to add a custom behaviour.
- `negate.js` (~329 tok, medium) — This example is used in the documentation.
- `no-repeated-options.js` (~225 tok, medium) — This is an example of using tokens to add a custom behaviour.
- `ordered-options.mjs` (~349 tok, medium) — // This is an example of using tokens to add a custom behaviour.
- `simple-hard-coded.js` (~135 tok, small) — This example is used in the documentation.
### `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/`

- `index.js` (~3234 tok, huge) — Work out where to slice process.argv for user supplied arguments.
### `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/internal/`

- `errors.js` (~358 tok, medium)
- `primordials.js` (~2987 tok, huge) — This file subclasses and stores the JS builtins that come from the VM
- `util.js` (~59 tok, small) — This is a placeholder for util.js in node.js land.
- `validators.js` (~561 tok, large) — This file is a proxy of the original file located at:
### `sdk/ts/mic-map/node_modules/@pkgjs/parseargs/`

- `LICENSE` (~2840 tok, huge) —                                  Apache License
- `package.json` (~221 tok, medium) — Keys: name, version, description, engines, main
- `README.md` (~3411 tok, huge) — parseArgs
- `utils.js` (~1563 tok, huge) — These are internal utilities to make the parsing logic easier to read, and
### `sdk/ts/mic-map/node_modules/postcss/lib/`

- `at-rule.d.ts` (~839 tok, large)
- `at-rule.js` (~118 tok, small)
- `comment.d.ts` (~420 tok, medium)
- `comment.js` (~51 tok, small)
- `container.d.ts` (~3507 tok, huge)
- `container.js` (~2662 tok, huge)
- `css-syntax-error.d.ts` (~1612 tok, huge)
- `css-syntax-error.js` (~851 tok, large)
- `declaration.d.ts` (~959 tok, large)
- `declaration.js` (~124 tok, small)
- `document.d.ts` (~471 tok, medium)
- `document.js` (~164 tok, small) — type needs to be passed to super, otherwise child roots won't be normalized correctly
- `fromJSON.d.ts` (~40 tok, tiny)
- `fromJSON.js` (~377 tok, medium)
- `input.d.ts` (~1282 tok, large)
- `input.js` (~1700 tok, huge)
- `lazy-result.d.ts` (~1235 tok, large)
- `lazy-result.js` (~3468 tok, huge) — eslint-disable-next-line no-console
- `list.d.ts` (~356 tok, medium)
- `list.js` (~307 tok, medium)
- `map-generator.js` (~2508 tok, huge)
- `node.d.ts` (~3674 tok, huge)
- `node.js` (~2680 tok, huge) — Not all custom syntaxes support `offset` in `source.start` and `source.end`
- `no-work-result.d.ts` (~376 tok, medium)
- `no-work-result.js` (~655 tok, large)
- `parse.d.ts` (~34 tok, tiny)
- `parse.js` (~287 tok, medium)
- `parser.js` (~3712 tok, huge) — If the token is a word, e.g. `!important`, `red` or any other valid property's value.
- `postcss.d.mts` (~262 tok, medium) — export {
- `postcss.d.ts` (~2886 tok, huge)
- `postcss.js` (~725 tok, large) — eslint-disable-next-line no-console
- `postcss.mjs` (~245 tok, medium) — import postcss from './postcss.js'
- `previous-map.d.ts` (~440 tok, medium)
- `previous-map.js` (~1127 tok, large) — sourceMappingURLs from comments, strings, etc.
- `processor.d.ts` (~837 tok, large)
- `processor.js` (~435 tok, medium)
- `result.d.ts` (~1087 tok, large)
- `result.js` (~185 tok, small)
- `root.d.ts` (~564 tok, large)
- `root.js` (~310 tok, medium)
- `rule.d.ts` (~720 tok, large)
- `rule.js` (~143 tok, small)
- `stringifier.d.ts` (~340 tok, medium)
- `stringifier.js` (~2257 tok, huge) — Escapes sequences that could break out of an HTML <style> context.
- `stringify.d.ts` (~41 tok, tiny)
- `stringify.js` (~54 tok, small)
- `symbols.js` (~23 tok, tiny)
- `terminal-highlight.js` (~350 tok, medium)
- `tokenize.js` (~1675 tok, huge)
- `warning.d.ts` (~731 tok, large)
- `warning.js` (~185 tok, small)
- `warn-once.js` (~64 tok, small)
### `sdk/ts/mic-map/node_modules/postcss/`

- `LICENSE` (~274 tok, medium) — The MIT License (MIT)
- `package.json` (~625 tok, large) — Keys: name, version, description, keywords, homepage
- `README.md` (~289 tok, medium) — PostCSS
### `sdk/ts/mic-map/node_modules/rollup/dist/bin/`

- `rollup` (~20568 tok, huge) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/rollup/dist/es/`

- `getLogFilter.js` (~512 tok, large)
- `package.json` (~5 tok, tiny) — Keys: type
- `parseAst.js` (~74 tok, small)
- `rollup.js` (~111 tok, small)
### `sdk/ts/mic-map/node_modules/rollup/dist/es/shared/`

- `parseAst.js` (~21538 tok, huge) — This file is generated by scripts/generate-node-types.js.
### `sdk/ts/mic-map/node_modules/rollup/dist/`

- `getLogFilter.d.ts` (~43 tok, tiny)
- `getLogFilter.js` (~547 tok, large)
- `loadConfigFile.d.ts` (~118 tok, small)
- `loadConfigFile.js` (~177 tok, small)
- `native.js` (~1197 tok, large) — This is needed because report.getReport() crashes the process on Windows sometimes.
- `parseAst.d.ts` (~34 tok, tiny)
- `parseAst.js` (~127 tok, small)
- `rollup.d.ts` (~9037 tok, huge) — utils
- `rollup.js` (~1167 tok, large) — Will be overwritten by Rollup
### `sdk/ts/mic-map/node_modules/rollup/dist/shared/`

- `fsevents-importer.js` (~221 tok, medium) — A call to this function will be injected into the chokidar code
- `loadConfigFile.js` (~5732 tok, huge) — split out plugins joined by commas
- `parseAst.js` (~23814 tok, huge) — Needed if a plugin did not generate correct sourcemaps
- `watch-cli.js` (~4620 tok, huge) — Offset the date so it will return the correct value when getting the ISO string.
- `watch.js` (~2639 tok, huge) — unwatching and watching fixes an issue with chokidar where on certain systems,
### `sdk/ts/mic-map/node_modules/rollup/`

- `LICENSE.md` (~8808 tok, huge) — Rollup core license
### `sdk/ts/mic-map/node_modules/rollup/node_modules/@types/estree/`

- `flow.d.ts` (~1201 tok, large)
- `index.d.ts` (~4736 tok, huge) — This definition file follows a somewhat unusual format. ESTree allows
- `LICENSE` (~286 tok, medium) —     MIT License
- `package.json` (~208 tok, medium) — Keys: name, version, description, homepage, license
- `README.md` (~115 tok, small) — Installation
### `sdk/ts/mic-map/node_modules/rollup/`

- `package.json` (~3107 tok, huge) — Keys: name, version, description, main, module
- `README.md` (~2500 tok, huge) — Overview
### `sdk/ts/mic-map/node_modules/@rollup/rollup-linux-x64-gnu/`

- `package.json` (~120 tok, small) — Keys: name, version, os, cpu, files
- `README.md` (~24 tok, tiny) — `@rollup/rollup-linux-x64-gnu`
### `sdk/ts/mic-map/node_modules/semver/bin/`

- `semver.js` (~1240 tok, large) — Standalone semver comparison program.
### `sdk/ts/mic-map/node_modules/semver/classes/`

- `comparator.js` (~908 tok, large) — hoisted class for cyclic dependency
- `index.js` (~36 tok, tiny)
- `range.js` (~3745 tok, huge) — hoisted class for cyclic dependency
- `semver.js` (~2370 tok, huge) — this isn't actually relevant for versions, but keep it so that we
### `sdk/ts/mic-map/node_modules/semver/functions/`

- `clean.js` (~52 tok, small)
- `cmp.js` (~241 tok, medium)
- `coerce.js` (~501 tok, large) — Find the right-most coercible string that does not share
- `compare-build.js` (~71 tok, small)
- `compare.js` (~43 tok, tiny)
- `compare-loose.js` (~33 tok, tiny)
- `diff.js` (~356 tok, medium) — Going from prerelease -> no prerelease requires some special casing
- `eq.js` (~32 tok, tiny)
- `gte.js` (~32 tok, tiny)
- `gt.js` (~31 tok, tiny)
- `inc.js` (~120 tok, small)
- `lte.js` (~32 tok, tiny)
- `lt.js` (~31 tok, tiny)
- `major.js` (~34 tok, tiny)
- `minor.js` (~34 tok, tiny)
- `neq.js` (~32 tok, tiny)
- `parse.js` (~83 tok, small)
- `patch.js` (~34 tok, tiny)
- `prerelease.js` (~59 tok, small)
- `rcompare.js` (~33 tok, tiny)
- `rsort.js` (~41 tok, tiny)
- `satisfies.js` (~62 tok, small)
- `sort.js` (~41 tok, tiny)
- `truncate.js` (~256 tok, medium)
- `valid.js` (~44 tok, tiny)
### `sdk/ts/mic-map/node_modules/semver/`

- `index.js` (~673 tok, large) — just pre-load all the stuff that index.js lazily exports
### `sdk/ts/mic-map/node_modules/semver/internal/`

- `constants.js` (~219 tok, medium) — Note: this is the semver.org version of the spec that it implements
- `debug.js` (~60 tok, small)
- `identifiers.js` (~132 tok, small)
- `lrucache.js` (~201 tok, medium) — Remove the key from the map and add it to the end
- `parse-options.js` (~85 tok, small) — parse out just the options we care about
- `re.js` (~2035 tok, huge) — The actual regexps go on exports.re
### `sdk/ts/mic-map/node_modules/semver/`

- `LICENSE` (~192 tok, small) — The ISC License
- `package.json` (~416 tok, medium) — Keys: name, version, description, main, scripts
- `preload.js` (~21 tok, tiny) — XXX remove in v8 or beyond
- `range.bnf` (~176 tok, small) — range-set  ::= range ( logical-or range ) *
### `sdk/ts/mic-map/node_modules/semver/ranges/`

- `gtr.js` (~58 tok, small) — Determine if version is greater than all the versions possible in the range.
- `intersects.js` (~56 tok, small)
- `ltr.js` (~57 tok, small) — Determine if version is less than all the versions possible in the range
- `max-satisfying.js` (~149 tok, small) — satisfies(v, range, options)
- `min-satisfying.js` (~148 tok, small) — satisfies(v, range, options)
- `min-version.js` (~379 tok, medium) — Clone to avoid manipulating the comparator's semver object.
- `outside.js` (~551 tok, large) — If it satisfies the range it is not outside
- `simplify.js` (~339 tok, medium) — given a set of versions and a range, create a "simplified" range
- `subset.js` (~1881 tok, huge) — Complex range `r1 || r2 || ...` is a subset of `R1 || R2 || ...` iff:
- `to-comparators.js` (~71 tok, small) — Mostly just for testing and legacy API reasons
- `valid.js` (~82 tok, small) — Return '*' instead of '' so that truthiness works.
### `sdk/ts/mic-map/node_modules/semver/`

- `README.md` (~6418 tok, huge) — Install
### `sdk/ts/mic-map/node_modules/shebang-command/`

- `index.js` (~97 tok, small)
- `license` (~279 tok, medium) — MIT License
- `package.json` (~140 tok, small) — Keys: name, version, description, license, repository
- `readme.md` (~124 tok, small) — shebang-command [![Build Status](https://travis-ci.org/kevva/shebang-command.svg?branch=master)](https://travis-ci.org/k
### `sdk/ts/mic-map/node_modules/shebang-regex/`

- `index.d.ts` (~112 tok, small)
- `index.js` (~11 tok, tiny)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~146 tok, small) — Keys: name, version, description, license, repository
- `readme.md` (~163 tok, small) — shebang-regex [![Build Status](https://travis-ci.org/sindresorhus/shebang-regex.svg?branch=master)](https://travis-ci.or
### `sdk/ts/mic-map/node_modules/siginfo/`

- `index.js` (~114 tok, small)
- `LICENSE` (~186 tok, small) — Copyright (c) 2017, Emil Bay <github@tixz.dk>
- `package.json` (~169 tok, small) — Keys: name, version, description, main, dependencies
- `README.md` (~283 tok, medium) — `siginfo`
- `test.js` (~66 tok, small)
- `.travis.yml` (~381 tok, medium) — notifications:
### `sdk/ts/mic-map/node_modules/signal-exit/dist/cjs/`

- `browser.d.ts` (~99 tok, small)
- `browser.d.ts.map` (~88 tok, small) — {"version":3,"file":"browser.d.ts","sourceRoot":"","sources":["../../src/browser
- `browser.js` (~81 tok, small)
- `browser.js.map` (~176 tok, small) — {"version":3,"file":"browser.js","sourceRoot":"","sources":["../../src/browser.t
- `index.d.ts` (~436 tok, medium)
- `index.d.ts.map` (~115 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~2359 tok, huge) — Note: since nyc uses this module to output coverage, any lines
- `index.js.map` (~4386 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~7 tok, tiny) — Keys: type
- `signals.d.ts` (~271 tok, medium)
- `signals.d.ts.map` (~49 tok, tiny) — {"version":3,"file":"signals.d.ts","sourceRoot":"","sources":["../../src/signals
- `signals.js` (~390 tok, medium) — should detect profiler and enable/disable accordingly.
- `signals.js.map` (~523 tok, large) — {"version":3,"file":"signals.js","sourceRoot":"","sources":["../../src/signals.t
### `sdk/ts/mic-map/node_modules/signal-exit/dist/mjs/`

- `browser.d.ts` (~99 tok, small)
- `browser.d.ts.map` (~88 tok, small) — {"version":3,"file":"browser.d.ts","sourceRoot":"","sources":["../../src/browser
- `browser.js` (~35 tok, tiny)
- `browser.js.map` (~167 tok, small) — {"version":3,"file":"browser.js","sourceRoot":"","sources":["../../src/browser.t
- `index.d.ts` (~436 tok, medium)
- `index.d.ts.map` (~115 tok, small) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../../src/index.ts"
- `index.js` (~2273 tok, huge) — Note: since nyc uses this module to output coverage, any lines
- `index.js.map` (~4398 tok, huge) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../../src/index.ts"],
- `package.json` (~6 tok, tiny) — Keys: type
- `signals.d.ts` (~271 tok, medium)
- `signals.d.ts.map` (~49 tok, tiny) — {"version":3,"file":"signals.d.ts","sourceRoot":"","sources":["../../src/signals
- `signals.js` (~360 tok, medium) — should detect profiler and enable/disable accordingly.
- `signals.js.map` (~525 tok, large) — {"version":3,"file":"signals.js","sourceRoot":"","sources":["../../src/signals.t
### `sdk/ts/mic-map/node_modules/signal-exit/`

- `LICENSE.txt` (~198 tok, small) — The ISC License
- `package.json` (~644 tok, large) — Keys: name, version, description, main, module
- `README.md` (~607 tok, large) — signal-exit
### `sdk/ts/mic-map/node_modules/source-map-js/lib/`

- `array-set.js` (~800 tok, large)
- `base64.js` (~385 tok, medium) — 0 - 25: ABCDEFGHIJKLMNOPQRSTUVWXYZ
- `base64-vlq.js` (~1179 tok, large) — A single base 64 digit can contain 6 bits of data. For the base 64 variable
- `binary-search.js` (~1063 tok, large) — This function terminates when one of the following is true:
- `mapping-list.js` (~585 tok, large) — Optimized for most common case
- `quick-sort.js` (~1017 tok, large) — It turns out that some (most?) JavaScript engines don't self-host
- `source-map-consumer.d.ts` (~10 tok, tiny)
- `source-map-consumer.js` (~10395 tok, huge) — parsed mapping coordinates from the source map's "mappings" attribute. They
- `source-map-generator.d.ts` (~11 tok, tiny)
- `source-map-generator.js` (~3734 tok, huge) — Add the source content to the _sourcesContents map.
- `source-node.d.ts` (~9 tok, tiny)
- `source-node.js` (~3452 tok, huge) — Matches a Windows-style `\r\n` newline or a `\n` newline used by all other
- `util.js` (~3851 tok, huge) — Split the path into parts between `/` characters. This is much faster than
### `sdk/ts/mic-map/node_modules/source-map-js/`

- `LICENSE` (~382 tok, medium)
- `package.json` (~637 tok, large) — Keys: name, description, version, homepage, author
- `README.md` (~6510 tok, huge) — Source Map JS
- `source-map.d.ts` (~852 tok, large) — SourceMapConsumer.GREATEST_LOWER_BOUND or SourceMapConsumer.LEAST_UPPER_BOUND
- `source-map.js` (~102 tok, small)
### `sdk/ts/mic-map/node_modules/stackback/`

- `formatstack.js` (~590 tok, large) — Copyright 2012 the V8 project authors. All rights reserved.
- `index.js` (~365 tok, medium) — v8 builtin format stack trace
- `.npmignore` (~4 tok, tiny) — node_modules
- `package.json` (~122 tok, small) — Keys: name, version, description, main, scripts
- `README.md` (~475 tok, medium) — stackback
- `test.js` (~156 tok, small) — calling stackback on the same error twice should work
- `.travis.yml` (~12 tok, tiny) — language: node_js
### `sdk/ts/mic-map/node_modules/std-env/dist/`

- `index.cjs` (~1336 tok, large) — "use strict";var b=Object.defineProperty;var C=Object.getOwnPropertySymbols;var 
- `index.d.cts` (~991 tok, large) — type EnvObject = Record<string, string | undefined>;
- `index.d.mts` (~991 tok, large) — type EnvObject = Record<string, string | undefined>;
- `index.d.ts` (~991 tok, large)
- `index.mjs` (~928 tok, large) — const r=Object.create(null),i=e=>globalThis.process?.env||import.meta.env||globa
### `sdk/ts/mic-map/node_modules/std-env/`

- `LICENCE` (~270 tok, medium) — MIT License
- `package.json` (~334 tok, medium) — Keys: name, version, description, repository, license
- `README.md` (~676 tok, large) — std-env
### `sdk/ts/mic-map/node_modules/string-width-cjs/`

- `index.d.ts` (~198 tok, small) — TODO: remove this in the next major version, refactor the whole definition to:
- `index.js` (~231 tok, medium) — Ignore control characters
- `license` (~278 tok, medium) — MIT License
### `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/ansi-regex/`

- `index.d.ts` (~186 tok, small)
- `index.js` (~88 tok, small)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~211 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~642 tok, large) — ansi-regex
### `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/emoji-regex/es2015/`

- `index.js` (~2776 tok, huge) — https://mths.be/emoji
- `text.js` (~2777 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/emoji-regex/`

- `index.d.ts` (~107 tok, small)
- `index.js` (~2572 tok, huge) — https://mths.be/emoji
- `LICENSE-MIT.txt` (~270 tok, medium) — Copyright Mathias Bynens <https://mathiasbynens.be/>
- `package.json` (~320 tok, medium) — Keys: name, version, description, homepage, main
- `README.md` (~673 tok, large) — emoji-regex [![Build status](https://travis-ci.org/mathiasbynens/emoji-regex.svg?branch=master)](https://travis-ci.org/m
- `text.js` (~2572 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/string-width-cjs/node_modules/strip-ansi/`

- `index.d.ts` (~93 tok, small)
- `index.js` (~39 tok, tiny)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~200 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~400 tok, medium) — strip-ansi [![Build Status](https://travis-ci.org/chalk/strip-ansi.svg?branch=master)](https://travis-ci.org/chalk/strip
### `sdk/ts/mic-map/node_modules/string-width-cjs/`

- `package.json` (~236 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~349 tok, medium) — string-width
### `sdk/ts/mic-map/node_modules/string-width/`

- `index.d.ts` (~206 tok, medium)
- `index.js` (~266 tok, medium) — Ignore control characters
- `license` (~280 tok, medium) — MIT License
- `package.json` (~261 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~434 tok, medium) — string-width
### `sdk/ts/mic-map/node_modules/strip-ansi-cjs/`

- `index.d.ts` (~93 tok, small)
- `index.js` (~39 tok, tiny)
- `license` (~278 tok, medium) — MIT License
### `sdk/ts/mic-map/node_modules/strip-ansi-cjs/node_modules/ansi-regex/`

- `index.d.ts` (~186 tok, small)
- `index.js` (~88 tok, small)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~211 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~642 tok, large) — ansi-regex
### `sdk/ts/mic-map/node_modules/strip-ansi-cjs/`

- `package.json` (~200 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~400 tok, medium) — strip-ansi [![Build Status](https://travis-ci.org/chalk/strip-ansi.svg?branch=master)](https://travis-ci.org/chalk/strip
### `sdk/ts/mic-map/node_modules/strip-ansi/`

- `index.d.ts` (~88 tok, small)
- `index.js` (~157 tok, small) — Fast path: ANSI codes require ESC (7-bit) or CSI (8-bit) introducer
- `license` (~280 tok, medium) — MIT License
- `package.json` (~241 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~306 tok, medium) — strip-ansi
### `sdk/ts/mic-map/node_modules/supports-color/`

- `browser.js` (~17 tok, tiny)
- `index.js` (~687 tok, large) — Windows 10 build 10586 is the first Windows release that supports 256 colors.
- `license` (~278 tok, medium) — MIT License
- `package.json` (~205 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~574 tok, large) — supports-color [![Build Status](https://travis-ci.org/chalk/supports-color.svg?branch=master)](https://travis-ci.org/cha
### `sdk/ts/mic-map/node_modules/test-exclude/`

- `index.js` (~1299 tok, large) — Don't instrument files that are outside of the current working directory.
- `is-outside-dir.js` (~44 tok, tiny)
- `is-outside-dir-posix.js` (~37 tok, tiny)
- `is-outside-dir-win32.js` (~68 tok, small)
- `LICENSE.txt` (~183 tok, small) — Copyright (c) 2016, Contributors
- `package.json` (~227 tok, medium) — Keys: name, version, description, main, files
- `README.md` (~1009 tok, large) — test-exclude
### `sdk/ts/mic-map/node_modules/tinybench/dist/`

- `index.cjs` (~3817 tok, huge) — "use strict";
- `index.d.cts` (~2265 tok, huge) — /**
- `index.d.ts` (~2265 tok, huge)
- `index.js` (~3639 tok, huge) — node_modules/.pnpm/yocto-queue@1.0.0/node_modules/yocto-queue/index.js
### `sdk/ts/mic-map/node_modules/tinybench/`

- `LICENSE` (~267 tok, medium) — MIT License
- `package.json` (~135 tok, small) — Keys: name, version, type, packageManager, main
- `README.md` (~3352 tok, huge) — Tinybench 🔎
### `sdk/ts/mic-map/node_modules/tinyexec/dist/`

- `main.cjs` (~4062 tok, huge) — "use strict";
- `main.d.cts` (~662 tok, large) — import { ChildProcess, SpawnOptions } from 'node:child_process';
- `main.d.ts` (~662 tok, large)
- `main.js` (~4073 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
### `sdk/ts/mic-map/node_modules/tinyexec/`

- `LICENSE` (~267 tok, medium) — MIT License
- `package.json` (~401 tok, medium) — Keys: name, version, type, description, main
- `README.md` (~1399 tok, large) — tinyexec 📟
### `sdk/ts/mic-map/node_modules/tinypool/dist/`

- `common-Qw-RoVFD.js` (~270 tok, medium)
### `sdk/ts/mic-map/node_modules/tinypool/dist/entry/`

- `process.d.ts` (~3 tok, tiny)
- `process.js` (~512 tok, large)
- `utils.d.ts` (~63 tok, small)
- `utils.js` (~27 tok, tiny)
- `worker.d.ts` (~3 tok, tiny)
- `worker.js` (~684 tok, large)
### `sdk/ts/mic-map/node_modules/tinypool/dist/`

- `index.d.ts` (~1561 tok, huge)
- `index.js` (~6664 tok, huge)
- `utils-B--2TaWv.js` (~304 tok, medium)
- `utils-De75vAgL.js` (~54 tok, small)
### `sdk/ts/mic-map/node_modules/tinypool/`

- `LICENSE` (~303 tok, medium) — The MIT License (MIT)
- `package.json` (~238 tok, medium) — Keys: name, type, version, packageManager, description
- `README.md` (~357 tok, medium) — Tinypool - the node.js worker pool 🧵
### `sdk/ts/mic-map/node_modules/tinyrainbow/dist/`

- `browser.d.ts` (~60 tok, small)
- `browser.js` (~49 tok, tiny) — src/browser.ts
- `chunk-BVHSVHOK.js` (~558 tok, large) — src/index.ts
- `index-c1cfc5e9.d.ts` (~592 tok, large)
- `node.d.ts` (~60 tok, small)
- `node.js` (~58 tok, small) — src/node.ts
### `sdk/ts/mic-map/node_modules/tinyrainbow/`

- `LICENCE` (~267 tok, medium) — MIT License
- `package.json` (~209 tok, medium) — Keys: name, version, description, type, main
- `README.md` (~127 tok, small) — tinyrainbow
### `sdk/ts/mic-map/node_modules/tinyspy/dist/`

- `index.cjs` (~1176 tok, large) — "use strict";
- `index.d.cts` (~816 tok, large) — declare const S: unique symbol;
- `index.d.ts` (~816 tok, large)
- `index.js` (~976 tok, large) — src/utils.ts
### `sdk/ts/mic-map/node_modules/tinyspy/`

- `LICENCE` (~267 tok, medium) — MIT License
- `package.json` (~226 tok, medium) — Keys: name, type, version, packageManager, description
- `README.md` (~125 tok, small) — tinyspy
### `sdk/ts/mic-map/node_modules/typescript/bin/`

- `tsc` (~12 tok, tiny) — #!/usr/bin/env node
- `tsserver` (~13 tok, tiny) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/typescript/lib/`

- `lib.decorators.d.ts` (~3298 tok, huge)
- `lib.decorators.legacy.d.ts` (~332 tok, medium)
- `lib.dom.asynciterable.d.ts` (~472 tok, medium) — Window Async Iterable APIs
- `lib.dom.iterable.d.ts` (~7451 tok, huge) — Window Iterable APIs
- `lib.d.ts` (~248 tok, medium)
- `lib.es2015.collection.d.ts` (~1308 tok, large)
- `lib.es2015.core.d.ts` (~5717 tok, huge)
- `lib.es2015.d.ts` (~311 tok, medium)
- `lib.es2015.generator.d.ts` (~640 tok, large) — NOTE: 'next' is defined using a tuple to ensure we report the correct assignability errors in all places.
- `lib.es2015.iterable.d.ts` (~4550 tok, huge) — NOTE: 'next' is defined using a tuple to ensure we report the correct assignability errors in all places.
- `lib.es2015.promise.d.ts` (~800 tok, large) — see: lib.es2015.iterable.d.ts
- `lib.es2015.proxy.d.ts` (~1313 tok, large)
- `lib.es2015.reflect.d.ts` (~1624 tok, huge)
- `lib.es2015.symbol.d.ts` (~413 tok, medium)
- `lib.es2015.symbol.wellknown.d.ts` (~2753 tok, huge)
- `lib.es2016.array.include.d.ts` (~1301 tok, large)
- `lib.es2016.d.ts` (~242 tok, medium)
- `lib.es2016.full.d.ts` (~258 tok, medium)
- `lib.es2016.intl.d.ts` (~371 tok, medium)
- `lib.es2017.arraybuffer.d.ts` (~229 tok, medium)
- `lib.es2017.date.d.ts` (~479 tok, medium)
- `lib.es2017.d.ts` (~291 tok, medium)
- `lib.es2017.full.d.ts` (~258 tok, medium)
- `lib.es2017.intl.d.ts` (~363 tok, medium)
- `lib.es2017.object.d.ts` (~619 tok, large)
- `lib.es2017.sharedmemory.d.ts` (~1803 tok, huge)
- `lib.es2017.string.d.ts` (~595 tok, large)
- `lib.es2017.typedarrays.d.ts` (~386 tok, medium)
- `lib.es2018.asyncgenerator.d.ts` (~673 tok, large) — NOTE: 'next' is defined using a tuple to ensure we report the correct assignability errors in all places.
- `lib.es2018.asynciterable.d.ts` (~565 tok, large) — NOTE: 'next' is defined using a tuple to ensure we report the correct assignability errors in all places.
- `lib.es2018.d.ts` (~272 tok, medium)
- `lib.es2018.full.d.ts` (~269 tok, medium)
- `lib.es2018.intl.d.ts` (~756 tok, large) — http://cldr.unicode.org/index/cldr-spec/plural-rules#TOC-Determining-Plural-Categories
- `lib.es2018.promise.d.ts` (~339 tok, medium)
- `lib.es2018.regexp.d.ts` (~308 tok, medium)
- `lib.es2019.array.d.ts` (~791 tok, large)
- `lib.es2019.d.ts` (~268 tok, medium)
- `lib.es2019.full.d.ts` (~269 tok, medium)
- `lib.es2019.intl.d.ts` (~240 tok, medium)
- `lib.es2019.object.d.ts` (~369 tok, medium)
- `lib.es2019.string.d.ts` (~382 tok, medium)
- `lib.es2019.symbol.d.ts` (~252 tok, medium)
- `lib.es2020.bigint.d.ts` (~9547 tok, huge)
- `lib.es2020.date.d.ts` (~739 tok, large)
- `lib.es2020.d.ts` (~301 tok, medium)
- `lib.es2020.full.d.ts` (~269 tok, medium)
- `lib.es2020.intl.d.ts` (~5610 tok, huge)
- `lib.es2020.number.d.ts` (~397 tok, medium)
- `lib.es2020.promise.d.ts` (~451 tok, medium)
- `lib.es2020.sharedmemory.d.ts` (~1282 tok, large)
- `lib.es2020.string.d.ts` (~639 tok, large)
- `lib.es2020.symbol.wellknown.d.ts` (~402 tok, medium)
- `lib.es2021.d.ts` (~259 tok, medium)
- `lib.es2021.full.d.ts` (~269 tok, medium)
- `lib.es2021.intl.d.ts` (~2062 tok, huge)
- `lib.es2021.promise.d.ts` (~566 tok, large)
- `lib.es2021.string.d.ts` (~398 tok, medium)
- `lib.es2021.weakref.d.ts` (~793 tok, large)
- `lib.es2022.array.d.ts` (~1169 tok, large)
- `lib.es2022.d.ts` (~277 tok, medium)
- `lib.es2022.error.d.ts` (~595 tok, large)
- `lib.es2022.full.d.ts` (~269 tok, medium)
- `lib.es2022.intl.d.ts` (~1923 tok, huge)
- `lib.es2022.object.d.ts` (~272 tok, medium)
- `lib.es2022.regexp.d.ts` (~333 tok, medium)
- `lib.es2022.string.d.ts` (~289 tok, medium)
- `lib.es2023.array.d.ts` (~10059 tok, huge)
- `lib.es2023.collection.d.ts` (~225 tok, medium)
- `lib.es2023.d.ts` (~250 tok, medium)
- `lib.es2023.full.d.ts` (~269 tok, medium)
- `lib.es2023.intl.d.ts` (~662 tok, large)
- `lib.es2024.arraybuffer.d.ts` (~660 tok, large)
- `lib.es2024.collection.d.ts` (~306 tok, medium)
- `lib.es2024.d.ts` (~292 tok, medium)
- `lib.es2024.full.d.ts` (~269 tok, medium)
- `lib.es2024.object.d.ts` (~315 tok, medium)
- `lib.es2024.promise.d.ts` (~348 tok, medium)
- `lib.es2024.regexp.d.ts` (~269 tok, medium)
- `lib.es2024.sharedmemory.d.ts` (~770 tok, large)
- `lib.es2024.string.d.ts` (~299 tok, medium)
- `lib.es6.d.ts` (~258 tok, medium)
- `lib.esnext.array.d.ts` (~448 tok, medium)
- `lib.esnext.collection.d.ts` (~970 tok, large)
- `lib.esnext.decorators.d.ts` (~268 tok, medium)
- `lib.esnext.disposable.d.ts` (~1702 tok, huge)
- `lib.esnext.d.ts` (~321 tok, medium)
- `lib.esnext.error.d.ts` (~258 tok, medium)
- `lib.esnext.float16.d.ts` (~5132 tok, huge)
- `lib.esnext.full.d.ts` (~269 tok, medium)
- `lib.esnext.intl.d.ts` (~224 tok, medium) — Empty
- `lib.esnext.iterator.d.ts` (~2169 tok, huge) — NOTE: This is specified as what is essentially an unreachable module. All actual global declarations can be found
- `lib.esnext.promise.d.ts` (~415 tok, medium)
- `lib.esnext.sharedmemory.d.ts` (~267 tok, medium)
- `lib.scripthost.d.ts` (~2363 tok, huge) — Windows Script Host APIS
- `lib.webworker.asynciterable.d.ts` (~472 tok, medium) — Worker Async Iterable APIs
- `lib.webworker.importscripts.d.ts` (~261 tok, medium) — WorkerGlobalScope APIs
- `lib.webworker.iterable.d.ts` (~5692 tok, huge) — Worker Iterable APIs
- `tsc.js` (~67 tok, small) — This file is a shim which defers loading the real module until the compile cache is enabled.
- `_tsserver.js` (~6972 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
- `tsserver.js` (~68 tok, small) — This file is a shim which defers loading the real module until the compile cache is enabled.
- `tsserverlibrary.d.ts` (~217 tok, medium)
- `tsserverlibrary.js` (~253 tok, medium)
- `typesMap.json` (~4197 tok, huge) — Keys: typesMap, simpleMap
- `_typingsInstaller.js` (~2591 tok, huge) — If the importer is in node compatibility mode or this is not an ESM
- `typingsInstaller.js` (~70 tok, small) — This file is a shim which defers loading the real module until the compile cache is enabled.
- `watchGuard.js` (~579 tok, large) — If the importer is in node compatibility mode or this is not an ESM
### `sdk/ts/mic-map/node_modules/typescript/`

- `LICENSE.txt` (~2300 tok, huge) — Apache License
- `package.json` (~905 tok, large) — Keys: name, author, homepage, version, license
- `README.md` (~711 tok, large) — TypeScript
- `SECURITY.md` (~664 tok, large) — Security
- `ThirdPartyNoticeText.txt` (~9456 tok, huge) — /*!----------------- TypeScript ThirdPartyNotices ------------------------------
### `sdk/ts/mic-map/node_modules/@types/estree/`

- `flow.d.ts` (~1201 tok, large)
- `index.d.ts` (~4731 tok, huge) — This definition file follows a somewhat unusual format. ESTree allows
- `LICENSE` (~286 tok, medium) —     MIT License
- `package.json` (~208 tok, medium) — Keys: name, version, description, homepage, license
- `README.md` (~115 tok, small) — Installation
### `sdk/ts/mic-map/node_modules/@types/node/`

- `assert.d.ts` (~11318 tok, huge) — eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
### `sdk/ts/mic-map/node_modules/@types/node/assert/`

- `strict.d.ts` (~751 tok, large)
### `sdk/ts/mic-map/node_modules/@types/node/`

- `async_hooks.d.ts` (~6389 tok, huge)
- `buffer.buffer.d.ts` (~5710 tok, huge) — see buffer.d.ts for implementation shared with all TypeScript versions
- `buffer.d.ts` (~22061 tok, huge) — If lib.dom.d.ts or lib.webworker.d.ts is loaded, then use the global types.
- `child_process.d.ts` (~16681 tok, huge) — stdin
- `cluster.d.ts` (~6982 tok, huge) — the handle is a net.Socket or net.Server object, or undefined.
### `sdk/ts/mic-map/node_modules/@types/node/compatibility/`

- `disposable.d.ts` (~82 tok, small) — Polyfills for the explicit resource management types added in TypeScript 5.2.
- `indexable.d.ts` (~248 tok, medium) — Polyfill for ES2022's .at() method on string/array prototypes, added to TypeScript in 4.6.
- `index.d.ts` (~134 tok, small) — Declaration files in this directory contain types relating to TypeScript library features
- `iterators.d.ts` (~294 tok, medium) — Backwards-compatible iterator interfaces, augmented with iterator helper methods by lib.esnext.iterator in TypeScript 5.
### `sdk/ts/mic-map/node_modules/@types/node/`

- `console.d.ts` (~5313 tok, huge) — This needs to be global to avoid TS2403 in case lib.dom.d.ts is present in the same build
- `constants.d.ts` (~204 tok, medium)
- `dgram.d.ts` (~7029 tok, huge)
- `diagnostics_channel.d.ts` (~6306 tok, huge)
- `dns.d.ts` (~9257 tok, huge) — Supported getaddrinfo flags.
### `sdk/ts/mic-map/node_modules/@types/node/dns/`

- `promises.d.ts` (~5275 tok, huge) — Error codes
### `sdk/ts/mic-map/node_modules/@types/node/`

- `domain.d.ts` (~1953 tok, huge)
- `events.d.ts` (~11063 tok, huge) — Should just be `export { EventEmitter }`, but that doesn't work in TypeScript 3.4
### `sdk/ts/mic-map/node_modules/@types/node/fs/`

- `promises.d.ts` (~13924 tok, huge) — TODO: Add `EventEmitter` close
### `sdk/ts/mic-map/node_modules/@types/node/`

- `globals.d.ts` (~1529 tok, huge) — Default TReturn/TNext in v22 is `any`, for compatibility with the previously-used IterableIterator.
- `globals.typedarray.d.ts` (~457 tok, medium) — The following aliases are required to allow use of non-shared ArrayBufferViews in @types/node
- `http.d.ts` (~23219 tok, huge) — incoming headers will never contain number
- `https.d.ts` (~6464 tok, huge)
- `index.d.ts` (~1037 tok, large) — NOTE: These definitions support Node.js and TypeScript 5.7+.
- `inspector.d.ts` (~2755 tok, huge) — These methods are exposed by the V8 inspector console API (inspector/v8-console.h).
- `LICENSE` (~286 tok, medium) —     MIT License
- `module.d.ts` (~10071 tok, huge) — Global-scope aliases for backwards compatibility with @types/node <13.0.x
- `net.d.ts` (~12302 tok, huge) — TODO: remove empty ConnectOpts placeholder at next major @types/node version.
- `os.d.ts` (~4858 tok, huge)
- `package.json` (~1069 tok, large) — Keys: name, version, description, homepage, license
- `path.d.ts` (~2084 tok, huge)
- `perf_hooks.d.ts` (~9607 tok, huge) — TODO: PerformanceNodeEntry is missing
- `punycode.d.ts` (~1370 tok, large)
- `querystring.d.ts` (~1782 tok, huge)
- `readline.d.ts` (~6470 tok, huge)
### `sdk/ts/mic-map/node_modules/@types/node/readline/`

- `promises.d.ts` (~1610 tok, huge)
### `sdk/ts/mic-map/node_modules/@types/node/`

- `README.md` (~386 tok, medium) — Installation
- `repl.d.ts` (~4865 tok, huge)
- `sea.d.ts` (~1549 tok, huge)
- `sqlite.d.ts` (~9020 tok, huge)
### `sdk/ts/mic-map/node_modules/@types/node/stream/`

- `consumers.d.ts` (~405 tok, medium)
### `sdk/ts/mic-map/node_modules/@types/node/`

- `stream.d.ts` (~21689 tok, huge) — TODO: this interface never existed; remove in next major
### `sdk/ts/mic-map/node_modules/@types/node/stream/`

- `promises.d.ts` (~743 tok, large)
- `web.d.ts` (~7487 tok, huge) — stub module, pending copy&paste from .d.ts or manual impl
### `sdk/ts/mic-map/node_modules/@types/node/`

- `string_decoder.d.ts` (~703 tok, large)
- `test.d.ts` (~25361 tok, huge)
- `timers.d.ts` (~3659 tok, huge) — Legacy interface used in Node.js v9 and prior
### `sdk/ts/mic-map/node_modules/@types/node/timers/`

- `promises.d.ts` (~945 tok, large)
### `sdk/ts/mic-map/node_modules/@types/node/`

- `tls.d.ts` (~15655 tok, huge)
- `trace_events.d.ts` (~2232 tok, huge)
### `sdk/ts/mic-map/node_modules/@types/node/ts5.6/`

- `buffer.buffer.d.ts` (~5520 tok, huge) — see ../buffer.d.ts for implementation shared with all TypeScript versions
- `globals.typedarray.d.ts` (~290 tok, medium)
- `index.d.ts` (~1089 tok, large) — NOTE: These definitions support Node.js and TypeScript 4.9 through 5.6.
### `sdk/ts/mic-map/node_modules/@types/node/`

- `tty.d.ts` (~2514 tok, huge)
- `url.d.ts` (~10848 tok, huge) — Input to `url.format`
- `util.d.ts` (~24855 tok, huge) — https://nodejs.org/docs/latest/api/util.html#foreground-colors
- `v8.d.ts` (~9691 tok, huge)
- `vm.d.ts` (~11503 tok, huge)
- `wasi.d.ts` (~1986 tok, huge)
### `sdk/ts/mic-map/node_modules/@types/node/web-globals/`

- `abortcontroller.d.ts` (~287 tok, medium)
- `domexception.d.ts` (~642 tok, large)
- `events.d.ts` (~718 tok, large)
- `fetch.d.ts` (~675 tok, large)
- `navigator.d.ts` (~203 tok, medium) — lib.webworker has `WorkerNavigator` rather than `Navigator`, so conditionals use `onabort` instead of `onmessage`
- `storage.d.ts` (~181 tok, small) — These interfaces are absent from lib.webworker, so the conditionals use `onabort` rather than `onmessage`
### `sdk/ts/mic-map/node_modules/@types/node/`

- `worker_threads.d.ts` (~9432 tok, huge)
- `zlib.d.ts` (~6989 tok, huge) — Allowed flush values.
### `sdk/ts/mic-map/node_modules/undici-types/`

- `agent.d.ts` (~267 tok, medium)
- `api.d.ts` (~364 tok, medium)
- `balanced-pool.d.ts` (~241 tok, medium) — Override dispatcher APIs.
- `cache.d.ts` (~313 tok, medium)
- `client.d.ts` (~1241 tok, large) — Override dispatcher APIs.
- `connector.d.ts` (~258 tok, medium)
- `content-type.d.ts` (~141 tok, small)
- `cookies.d.ts` (~159 tok, small)
- `diagnostics-channel.d.ts` (~395 tok, medium)
- `dispatcher.d.ts` (~3556 tok, huge)
- `env-http-proxy-agent.d.ts` (~169 tok, small)
- `errors.d.ts` (~1070 tok, large)
- `eventsource.d.ts` (~412 tok, medium)
- `fetch.d.ts` (~1394 tok, large) — based on https://github.com/Ethan-Arrowood/undici-fetch/blob/249269714db874351589d2d364a0645d5160ae71/index.d.ts (MIT li
- `file.d.ts` (~427 tok, medium) — Based on https://github.com/octet-stream/form-data/blob/2d0f0dc371517444ce1f22cdde13f51995d0953a/lib/File.ts (MIT)
- `filereader.d.ts` (~359 tok, medium)
- `formdata.d.ts` (~1250 tok, large) — Based on https://github.com/octet-stream/form-data/blob/2d0f0dc371517444ce1f22cdde13f51995d0953a/lib/FormData.ts (MIT)
- `global-dispatcher.d.ts` (~69 tok, small)
- `global-origin.d.ts` (~44 tok, tiny)
- `handlers.d.ts` (~112 tok, small)
- `header.d.ts` (~34 tok, tiny)
- `index.d.ts` (~847 tok, large)
- `interceptors.d.ts` (~231 tok, medium)
- `LICENSE` (~273 tok, medium) — MIT License
- `mock-agent.d.ts` (~634 tok, large)
- `mock-client.d.ts` (~251 tok, medium)
- `mock-errors.d.ts` (~85 tok, small)
- `mock-interceptor.d.ts` (~976 tok, large)
- `mock-pool.d.ts` (~244 tok, medium)
- `package.json` (~299 tok, medium) — Keys: name, version, description, homepage, bugs
- `patch.d.ts` (~173 tok, small) — See https://github.com/nodejs/undici/issues/1740
- `pool.d.ts` (~334 tok, medium) — Override dispatcher APIs.
- `pool-stats.d.ts` (~168 tok, small)
- `proxy-agent.d.ts` (~195 tok, small)
- `readable.d.ts` (~435 tok, medium)
- `README.md` (~114 tok, small) — undici-types
- `retry-agent.d.ts` (~58 tok, small)
- `retry-handler.d.ts` (~746 tok, large)
- `util.d.ts` (~156 tok, small)
- `webidl.d.ts` (~1481 tok, large) — These types are not exported, and are only used internally
- `websocket.d.ts` (~960 tok, large)
### `sdk/ts/mic-map/node_modules/vite/bin/`

- `openChrome.applescript` (~673 tok, large) — (*
- `vite.js` (~418 tok, medium) — only available as dev dependency
### `sdk/ts/mic-map/node_modules/vite/`

- `client.d.ts` (~1186 tok, large) — CSS modules
### `sdk/ts/mic-map/node_modules/vite/dist/client/`

- `client.mjs` (~5946 tok, huge) — import '@vite/env';
- `env.mjs` (~161 tok, small) — const context = (() => {
### `sdk/ts/mic-map/node_modules/vite/dist/node/chunks/`

- `dep-BB45zftN.js` (~5803 tok, huge) — Base64 encode an import with conditions
- `dep-IQS-Za7F.js` (~3353 tok, huge) — Whitespaces
### `sdk/ts/mic-map/node_modules/vite/dist/node/`

- `cli.js` (~6974 tok, huge)
- `constants.js` (~734 tok, large) — moment still uses this...
- `index.js` (~1882 tok, huge) — eslint-disable-next-line regexp/no-unused-capturing-group
- `runtime.d.ts` (~679 tok, large)
- `runtime.js` (~11250 tok, huge) — export names (first arg) are irrelevant on the client side, they're
- `types.d-aGj9QkWt.d.ts` (~2427 tok, huge)
### `sdk/ts/mic-map/node_modules/vite/`

- `index.cjs` (~406 tok, medium) — warnCjsUsage()
- `index.d.cts` (~53 tok, small) — /**
### `sdk/ts/mic-map/node_modules/vite-node/dist/`

- `chunk-browser.cjs` (~533 tok, large) — 'use strict';
- `chunk-browser.mjs` (~529 tok, large) — // src/index.ts
- `chunk-hmr.cjs` (~2065 tok, huge) — 'use strict';
- `chunk-hmr.mjs` (~2020 tok, huge) — import { EventEmitter } from 'node:events';
- `cli.cjs` (~1287 tok, large) — 'use strict';
- `cli.d.ts` (~229 tok, medium)
- `client.cjs` (~3755 tok, huge) — 'use strict';
- `client.d.ts` (~37 tok, tiny)
- `client.mjs` (~3741 tok, huge) — import { createRequire } from 'node:module';
- `cli.mjs` (~1275 tok, large) — import { resolve } from 'node:path';
- `constants.cjs` (~159 tok, small) — 'use strict';
- `constants.d.ts` (~45 tok, tiny)
- `constants.mjs` (~139 tok, small) — const KNOWN_ASSET_TYPES = [
- `hmr.cjs` (~136 tok, small) — 'use strict';
- `hmr.d.ts` (~653 tok, large)
- `hmr.mjs` (~86 tok, small) — export { c as createHmrEmitter, a as createHotContext, g as getCache, h as handl
- `index.cjs` (~4 tok, tiny) — 'use strict';
- `index.d.ts` (~137 tok, small)
- `index.mjs` (~1 tok, tiny)
- `index-z0R8hVRu.d.ts` (~2407 tok, huge) — eslint-disable-next-line n/no-unsupported-features/node-builtins
- `server.cjs` (~4554 tok, huge) — 'use strict';
- `server.d.ts` (~665 tok, large)
- `server.mjs` (~4389 tok, huge) — import assert from 'node:assert';
- `source-map.cjs` (~7906 tok, huge) — 'use strict';
- `source-map.d.ts` (~152 tok, small)
- `source-map.mjs` (~7887 tok, huge) — import { isAbsolute, resolve as resolve$2, relative, dirname } from 'pathe';
- `trace-mapping.d-DLVdEqOp.d.ts` (~487 tok, medium)
- `types.cjs` (~4 tok, tiny) — 'use strict';
- `types.d.ts` (~137 tok, small)
- `types.mjs` (~1 tok, tiny)
- `utils.cjs` (~1604 tok, huge) — 'use strict';
- `utils.d.ts` (~384 tok, medium)
- `utils.mjs` (~1504 tok, huge) — import { existsSync, promises } from 'node:fs';
### `sdk/ts/mic-map/node_modules/vite-node/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~590 tok, large) — Keys: name, type, version, description, author
- `README.md` (~1306 tok, large) — Features
- `vite-node.mjs` (~12 tok, tiny) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/vite/`

- `package.json` (~1224 tok, large) — Keys: name, version, type, license, author
- `README.md` (~288 tok, medium) — vite ⚡
### `sdk/ts/mic-map/node_modules/vitest/`

- `browser.d.ts` (~9 tok, tiny)
- `config.d.ts` (~9 tok, tiny)
- `coverage.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/coverage-v8/dist/`

- `browser.d.ts` (~105 tok, small)
- `browser.js` (~321 tok, medium)
- `index.d.ts` (~116 tok, small)
- `index.js` (~301 tok, medium)
- `load-provider-Bl5rgjsL.js` (~54 tok, small)
- `provider.d.ts` (~234 tok, medium)
### `sdk/ts/mic-map/node_modules/@vitest/coverage-v8/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~532 tok, large) — Keys: name, type, version, description, author
### `sdk/ts/mic-map/node_modules/vitest/dist/`

- `browser.d.ts` (~661 tok, large)
- `browser.js` (~144 tok, small)
### `sdk/ts/mic-map/node_modules/vitest/dist/chunks/`

- `base.BZZh4cSm.js` (~283 tok, medium)
- `benchmark.Cdu9hjj4.js` (~325 tok, medium)
- `benchmark.geERunq4.d.ts` (~215 tok, medium)
- `cac.CB_9Zo9Q.js` (~12910 tok, huge) — skip dot names because only top level options are required
- `_commonjsHelpers.BFTU3MAI.js` (~100 tok, small)
- `config.Cy0C388Z.d.ts` (~1534 tok, huge)
- `console.BYGVloWk.js` (~1297 tok, large)
- `constants.fzPh7AOq.js` (~309 tok, medium) — Vite client
- `coverage.BoMDb1ip.js` (~576 tok, large)
- `creator.IIqd8RWT.js` (~4643 tok, huge)
- `date.W2xKR2qe.js` (~324 tok, medium)
- `environment.LoooBwUu.d.ts` (~1403 tok, large)
- `execute.2pr0rHgK.js` (~6249 tok, huge) — your mocked methods
- `git.B5SDxu-n.js` (~445 tok, medium)
- `globals.D8ZVAdXd.js` (~178 tok, small)
- `index.68735LiX.js` (~921 tok, large)
- `index.ckWaX2gY.js` (~392 tok, medium)
- `index.CqYx2Nsr.js` (~893 tok, large) — src/detect.ts
- `index.K90BXFOx.js` (~4015 tok, huge) — not specified in docs, but is available
- `index.nEwtF0bu.js` (~1004 tok, large)
- `inspector.70d6emsh.js` (~436 tok, medium)
- `mocker.cRtM890J.d.ts` (~137 tok, small)
- `node.AKq966Jp.js` (~125 tok, small)
- `RandomSequencer.CMRlh2v4.js` (~12456 tok, huge) — AST walker module for ESTree compatible trees
- `reporters.nr4dxCkA.d.ts` (~22807 tok, huge) — Write output but don't hide the cursor
- `rpc.C3q9uwRX.js` (~751 tok, large)
- `runBaseTests.3qpJUEJM.js` (~1221 tok, large)
- `run-once.2ogXb3JV.js` (~188 tok, small)
- `setup-common.Dj6BZI3u.js` (~565 tok, large)
- `spy.Cf_4R5Oe.js` (~134 tok, small)
- `suite.B2jumIFP.d.ts` (~101 tok, small)
- `utils.C8RiOc4B.js` (~601 tok, large) — Vitest
- `utils.Cn0zI1t3.js` (~455 tok, medium)
- `utils.DNoFbBUZ.js` (~1772 tok, huge)
- `vite.CzKp4x9w.d.ts` (~63 tok, small)
- `vm.Zr4qWzDJ.js` (~6836 tok, huge) — exposed for external use, Node.js does the opposite
- `worker.B9FxPCaC.d.ts` (~54 tok, small)
- `worker.tN5KGIih.d.ts` (~1351 tok, large)
### `sdk/ts/mic-map/node_modules/vitest/dist/`

- `cli.js` (~51 tok, small)
- `config.cjs` (~976 tok, large) — 'use strict';
- `config.d.ts` (~953 tok, large)
- `config.js` (~882 tok, large) — Vite client
- `coverage.d.ts` (~1666 tok, huge)
- `coverage.js` (~4002 tok, huge) — User's options
- `environments.d.ts` (~165 tok, small)
- `environments.js` (~29 tok, tiny)
- `execute.d.ts` (~1402 tok, large)
- `execute.js` (~84 tok, small)
- `index.d.ts` (~9000 tok, huge)
- `index.js` (~233 tok, medium)
- `mocker.d.ts` (~8 tok, tiny)
- `mocker.js` (~8 tok, tiny)
- `node.d.ts` (~1704 tok, huge)
- `node.js` (~619 tok, large)
- `path.js` (~62 tok, small)
- `reporters.d.ts` (~328 tok, medium)
- `reporters.js` (~254 tok, medium)
- `runners.d.ts` (~410 tok, medium)
- `runners.js` (~2240 tok, huge)
- `snapshot.d.ts` (~93 tok, small)
- `snapshot.js` (~51 tok, small)
- `spy.js` (~8 tok, tiny)
- `suite.d.ts` (~81 tok, small)
- `suite.js` (~82 tok, small)
- `utils.d.ts` (~25 tok, tiny)
- `utils.js` (~19 tok, tiny)
- `worker.js` (~1326 tok, large) — here we create a new one, workers can reassign this if they need to keep it non-isolated
- `workers.d.ts` (~463 tok, medium)
### `sdk/ts/mic-map/node_modules/vitest/dist/workers/`

- `forks.js` (~270 tok, medium)
### `sdk/ts/mic-map/node_modules/vitest/dist/`

- `workers.js` (~275 tok, medium)
### `sdk/ts/mic-map/node_modules/vitest/dist/workers/`

- `runVmTests.js` (~827 tok, large)
- `threads.js` (~197 tok, small)
- `vmForks.js` (~312 tok, medium)
- `vmThreads.js` (~238 tok, medium)
### `sdk/ts/mic-map/node_modules/vitest/`

- `environments.d.ts` (~9 tok, tiny)
- `execute.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/expect/dist/`

- `chai.d.cts` (~18324 tok, huge) — // Type definitions for chai 4.3
- `index.d.ts` (~6521 tok, huge)
- `index.js` (~15851 tok, huge) — seems redundant with received === ''
### `sdk/ts/mic-map/node_modules/@vitest/expect/`

- `index.d.ts` (~15 tok, tiny)
- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~281 tok, medium) — Keys: name, type, version, description, license
- `README.md` (~113 tok, small) — @vitest/expect
### `sdk/ts/mic-map/node_modules/vitest/`

- `globals.d.ts` (~222 tok, medium)
- `import-meta.d.ts` (~63 tok, small) — https://github.com/microsoft/TypeScript/issues/45096
- `importMeta.d.ts` (~21 tok, tiny)
- `index.cjs` (~103 tok, small) — throw new Error(
- `index.d.cts` (~8 tok, tiny) — export * from './dist/index.js'
- `jsdom.d.ts` (~22 tok, tiny)
- `LICENSE.md` (~19605 tok, huge) — Vitest core license
### `sdk/ts/mic-map/node_modules/@vitest/mocker/dist/`

- `auto-register.d.ts` (~4 tok, tiny)
- `auto-register.js` (~87 tok, small)
- `browser.d.ts` (~531 tok, large)
- `browser.js` (~874 tok, large)
- `chunk-interceptor-native.js` (~92 tok, small)
- `chunk-mocker.js` (~4298 tok, huge) — We need to await mock registration before importing the actual module
- `chunk-pathe.ff20891b.js` (~1218 tok, large)
- `chunk-registry.js` (~1275 tok, large)
- `chunk-utils.js` (~30 tok, tiny)
- `index.d.ts` (~265 tok, medium)
- `index.js` (~1358 tok, large)
- `mocker-pQgp1HFr.d.ts` (~852 tok, large)
- `node.d.ts` (~5412 tok, huge) — This definition file follows a somewhat unusual format. ESTree allows
- `node.js` (~11254 tok, huge) — TODO: make env updatable
- `redirect.d.ts` (~35 tok, tiny)
- `redirect.js` (~510 tok, large)
- `register.d.ts` (~110 tok, small)
- `register.js` (~321 tok, medium)
- `types-DZOqTgiN.d.ts` (~837 tok, large)
### `sdk/ts/mic-map/node_modules/vitest/`

- `mocker.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/mocker/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~475 tok, medium) — Keys: name, type, version, description, license
- `README.md` (~57 tok, small) — @vitest/mocker
### `sdk/ts/mic-map/node_modules/vitest/`

- `node.d.ts` (~8 tok, tiny)
- `package.json` (~1249 tok, large) — Keys: name, type, version, description, author
### `sdk/ts/mic-map/node_modules/@vitest/pretty-format/dist/`

- `index.d.ts` (~833 tok, large)
- `index.js` (~10015 tok, huge) — ATTENTION
### `sdk/ts/mic-map/node_modules/@vitest/pretty-format/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~259 tok, medium) — Keys: name, type, version, description, license
### `sdk/ts/mic-map/node_modules/vitest/`

- `README.md` (~66 tok, small) — vitest
- `reporters.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/runner/dist/`

- `chunk-tasks.js` (~1609 tok, huge)
- `index.d.ts` (~3303 tok, huge)
- `index.js` (~10077 tok, huge) — https://github.com/chaijs/chai/pull/1490
- `tasks-3ZnPj1LR.d.ts` (~4681 tok, huge)
- `types.d.ts` (~1450 tok, large)
- `types.js` (~1 tok, tiny)
- `utils.d.ts` (~483 tok, medium)
- `utils.js` (~110 tok, small)
### `sdk/ts/mic-map/node_modules/@vitest/runner/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~281 tok, medium) — Keys: name, type, version, description, license
- `README.md` (~41 tok, tiny) — @vitest/runner
### `sdk/ts/mic-map/node_modules/vitest/`

- `runners.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/runner/`

- `types.d.ts` (~8 tok, tiny)
- `utils.d.ts` (~8 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/snapshot/dist/`

- `environment-Ddx0EDtY.d.ts` (~186 tok, small)
- `environment.d.ts` (~184 tok, small)
- `environment.js` (~301 tok, medium)
- `index.d.ts` (~919 tok, large)
- `index.js` (~17338 tok, huge) — Matches the scheme of a URL, eg "http://"
- `manager.d.ts` (~210 tok, medium)
- `manager.js` (~486 tok, medium)
- `rawSnapshot-CPNkto81.d.ts` (~435 tok, medium)
### `sdk/ts/mic-map/node_modules/vitest/`

- `snapshot.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/snapshot/`

- `environment.d.ts` (~10 tok, tiny)
- `LICENSE` (~269 tok, medium) — MIT License
- `manager.d.ts` (~9 tok, tiny)
- `package.json` (~332 tok, medium) — Keys: name, type, version, description, license
- `README.md` (~643 tok, large) — @vitest/snapshot
### `sdk/ts/mic-map/node_modules/@vitest/spy/dist/`

- `index.d.ts` (~3882 tok, huge)
- `index.js` (~1042 tok, large)
### `sdk/ts/mic-map/node_modules/@vitest/spy/`

- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~228 tok, medium) — Keys: name, type, version, description, license
- `README.md` (~16 tok, tiny) — @vitest/spy
### `sdk/ts/mic-map/node_modules/vitest/`

- `suite.d.ts` (~8 tok, tiny)
- `suppress-warnings.cjs` (~195 tok, small) — // borrowed from tsx implementation:
### `sdk/ts/mic-map/node_modules/@vitest/utils/`

- `diff.d.ts` (~8 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/utils/dist/`

- `chunk-_commonjsHelpers.js` (~1078 tok, large) — min: true,
- `diff.d.ts` (~1039 tok, large)
- `diff.js` (~17216 tok, huge) — This diff-sequences package implements the linear space variation in
- `error.d.ts` (~86 tok, small)
- `error.js` (~967 tok, large)
- `helpers.d.ts` (~587 tok, large)
- `helpers.js` (~1565 tok, huge)
- `index.d.ts` (~609 tok, large)
- `index.js` (~5275 tok, huge) — Copyright 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Simon Lydell
- `source-map.d.ts` (~1035 tok, large)
- `source-map.js` (~8371 tok, huge) — Matches the scheme of a URL, eg "http://"
- `types-Bxe-2Udy.d.ts` (~274 tok, medium)
- `types.d.ts` (~402 tok, medium)
- `types.js` (~1 tok, tiny)
### `sdk/ts/mic-map/node_modules/vitest/`

- `utils.d.ts` (~8 tok, tiny)
### `sdk/ts/mic-map/node_modules/@vitest/utils/`

- `error.d.ts` (~8 tok, tiny)
- `helpers.d.ts` (~9 tok, tiny)
- `LICENSE` (~269 tok, medium) — MIT License
- `package.json` (~446 tok, medium) — Keys: name, type, version, description, license
### `sdk/ts/mic-map/node_modules/vitest/`

- `vitest.mjs` (~11 tok, tiny) — #!/usr/bin/env node
- `workers.d.ts` (~9 tok, tiny)
### `sdk/ts/mic-map/node_modules/vite/types/`

- `customEvent.d.ts` (~286 tok, medium) — eslint-disable-next-line n/no-unsupported-features/node-builtins
- `hmrPayload.d.ts` (~276 tok, medium)
- `hot.d.ts` (~251 tok, medium)
- `importGlob.d.ts` (~478 tok, medium)
- `import-meta.d.ts` (~63 tok, small) — https://github.com/microsoft/TypeScript/issues/45096
- `importMeta.d.ts` (~128 tok, small) — This file is an augmentation to the built-in ImportMeta interface
- `metadata.d.ts` (~49 tok, tiny)
- `package.json` (~28 tok, tiny) — Keys: //, version
### `sdk/ts/mic-map/node_modules/.vite/vitest/`

- `results.json` (~76 tok, small) — Keys: version, results
### `sdk/ts/mic-map/node_modules/which/bin/`

- `node-which` (~247 tok, medium) — #!/usr/bin/env node
### `sdk/ts/mic-map/node_modules/which/`

- `CHANGELOG.md` (~667 tok, large) — Changes
- `LICENSE` (~192 tok, small) — The ISC License
- `package.json` (~261 tok, medium) — Keys: author, name, description, version, repository
- `README.md` (~338 tok, medium) — which
- `which.js` (~791 tok, large) — If it has a slash, then we don't bother searching the pathenv.
### `sdk/ts/mic-map/node_modules/why-is-node-running/`

- `cli.js` (~111 tok, small)
- `example.js` (~61 tok, small)
### `sdk/ts/mic-map/node_modules/why-is-node-running/.github/`

- `FUNDING.yml` (~5 tok, tiny) — github: mafintosh
### `sdk/ts/mic-map/node_modules/why-is-node-running/`

- `include.js` (~14 tok, tiny)
- `index.js` (~519 tok, large)
- `LICENSE` (~270 tok, medium) — The MIT License (MIT)
- `package.json` (~214 tok, medium) — Keys: name, version, description, main, dependencies
- `README.md` (~634 tok, large) — why-is-node-running
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/`

- `index.js` (~1443 tok, large) — Calculate the length of words split on ' ', ignoring
- `license` (~280 tok, medium) — MIT License
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/ansi-regex/`

- `index.d.ts` (~186 tok, small)
- `index.js` (~88 tok, small)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~211 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~642 tok, large) — ansi-regex
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/ansi-styles/`

- `index.d.ts` (~1588 tok, huge)
- `index.js` (~1035 tok, large) — 21 isn't widely supported and 22 does the same thing
- `license` (~278 tok, medium) — MIT License
- `package.json` (~264 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~1082 tok, large) — ansi-styles [![Build Status](https://travis-ci.org/chalk/ansi-styles.svg?branch=master)](https://travis-ci.org/chalk/ans
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/emoji-regex/es2015/`

- `index.js` (~2776 tok, huge) — https://mths.be/emoji
- `text.js` (~2777 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/emoji-regex/`

- `index.d.ts` (~107 tok, small)
- `index.js` (~2572 tok, huge) — https://mths.be/emoji
- `LICENSE-MIT.txt` (~270 tok, medium) — Copyright Mathias Bynens <https://mathiasbynens.be/>
- `package.json` (~320 tok, medium) — Keys: name, version, description, homepage, main
- `README.md` (~673 tok, large) — emoji-regex [![Build status](https://travis-ci.org/mathiasbynens/emoji-regex.svg?branch=master)](https://travis-ci.org/m
- `text.js` (~2572 tok, huge) — https://mths.be/emoji
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/string-width/`

- `index.d.ts` (~198 tok, small) — TODO: remove this in the next major version, refactor the whole definition to:
- `index.js` (~231 tok, medium) — Ignore control characters
- `license` (~278 tok, medium) — MIT License
- `package.json` (~236 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~349 tok, medium) — string-width
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/node_modules/strip-ansi/`

- `index.d.ts` (~93 tok, small)
- `index.js` (~39 tok, tiny)
- `license` (~278 tok, medium) — MIT License
- `package.json` (~200 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~400 tok, medium) — strip-ansi [![Build Status](https://travis-ci.org/chalk/strip-ansi.svg?branch=master)](https://travis-ci.org/chalk/strip
### `sdk/ts/mic-map/node_modules/wrap-ansi-cjs/`

- `package.json` (~254 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~687 tok, large) — wrap-ansi [![Build Status](https://travis-ci.com/chalk/wrap-ansi.svg?branch=master)](https://travis-ci.com/chalk/wrap-an
### `sdk/ts/mic-map/node_modules/wrap-ansi/`

- `index.d.ts` (~317 tok, medium)
- `index.js` (~1445 tok, large) — Calculate the length of words split on ' ', ignoring
- `license` (~280 tok, medium) — MIT License
- `package.json` (~287 tok, medium) — Keys: name, version, description, license, repository
- `readme.md` (~617 tok, large) — wrap-ansi
### `sdk/ts/mic-map/`

- `package.json` (~210 tok, medium) — Keys: name, version, description, type, private
- `package-lock.json` (~19547 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
- `README.md` (~152 tok, small) — @mind/mic-map
### `sdk/ts/mic-map/scripts/`

- `regen_fixtures.sh` (~499 tok, medium) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/src/`

- `errors.ts` (~351 tok, medium) — Copyright 2026 STARGA Inc.
- `framing.ts` (~726 tok, large) — Copyright 2026 STARGA Inc.
- `index.ts` (~497 tok, medium) — Copyright 2026 STARGA Inc.
- `map.ts` (~2187 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2_emit.ts` (~586 tok, large) — Copyright 2026 STARGA Inc.
- `mic2_parse.ts` (~2242 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.ts` (~2959 tok, huge) — Copyright 2026 STARGA Inc.
- `types.ts` (~2016 tok, huge) — Copyright 2026 STARGA Inc.
- `varint.ts` (~617 tok, large) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/test/fixtures/`

- `map_examples.txt` (~76 tok, small) — # MAP protocol frame examples
- `residual_block.mic2.txt` (~20 tok, tiny) — mic@2
### `sdk/ts/mic-map/test/`

- `framing.test.ts` (~1257 tok, large) — Copyright 2026 STARGA Inc.
- `map.test.ts` (~2328 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2.test.ts` (~2637 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.test.ts` (~1621 tok, huge) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/`

- `tsconfig.json` (~152 tok, small) — Keys: compilerOptions, include, exclude
- `vitest.config.ts` (~72 tok, small)
### `skills/write-mind/`

- `SKILL.md` (~6002 tok, huge) — Write MIND Code
### `src/ast/`

- `mod.rs` (~7430 tok, huge) — Copyright 2025 STARGA Inc.
### `src/autodiff/`

- `engine.rs` (~3890 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~342 tok, medium) — Copyright 2025 STARGA Inc.
- `rules.rs` (~2392 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/`

- `mind-ai.rs` (~9805 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc.rs` (~15275 tok, huge) — Copyright 2025 STARGA Inc.
- `mind-native.rs` (~2351 tok, huge) — Copyright 2026 STARGA Inc.
### `src/build/`

- `cache.rs` (~5282 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~6077 tok, huge) — Copyright 2025 STARGA Inc.
### `src/cache/`

- `entry.rs` (~820 tok, large) — Copyright 2025-2026 STARGA Inc.
- `fingerprint.rs` (~629 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~964 tok, large) — Copyright 2025-2026 STARGA Inc.
- `store.rs` (~1112 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/check/`

- `gitignore.rs` (~2345 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~7034 tok, huge) — Copyright 2025 STARGA Inc.
- `reporter.rs` (~374 tok, medium) — Copyright 2025 STARGA Inc.
### `src/`

- `conformance.rs` (~1847 tok, huge) — The autodiff_pairwise conformance entry was removed 2026-05-20 — its
### `src/deps/`

- `mod.rs` (~8087 tok, huge) — Copyright 2025 STARGA Inc.
### `src/diagnostics/`

- `mod.rs` (~2230 tok, huge) — Copyright 2025 STARGA Inc.
### `src/distributed/`

- `allgather.rs` (~813 tok, large) — Copyright 2025-2026 STARGA Inc.
- `allreduce.rs` (~1011 tok, large) — Copyright 2025-2026 STARGA Inc.
- `invariants.rs` (~1416 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~580 tok, large) — Copyright 2025-2026 STARGA Inc.
- `pipeline.rs` (~1973 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `shard.rs` (~1640 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/doc/`

- `html.rs` (~956 tok, large) — Copyright 2025 STARGA Inc.
- `markdown.rs` (~2644 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~6874 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `abi_gate.rs` (~7970 tok, huge) — Runnable-artifact ABI gate (release-readiness P1.1).
- `autodiff.rs` (~14268 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~2397 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_interp.rs` (~3818 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~8046 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~9665 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~301 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~501 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_opt.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_run.rs` (~1535 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~8360 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `struct_resolver.rs` (~6499 tok, huge) — Copyright 2025 STARGA Inc.
- `value.rs` (~2003 tok, huge) — Copyright 2025 STARGA Inc.
### `src/exec/`

- `conv.rs` (~435 tok, medium) — Copyright 2025 STARGA Inc.
- `cpu.rs` (~3570 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~587 tok, large) — Copyright 2025 STARGA Inc.
### `src/ffi/`

- `header.rs` (~413 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1737 tok, huge) — Copyright 2025 STARGA Inc.
- `sys.rs` (~1769 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/fmt/`

- `cli.rs` (~3873 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~594 tok, large) — Copyright 2025 STARGA Inc.
- `printer.rs` (~15195 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/`

- `emit.rs` (~4693 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2377 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~8124 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v2/`

- `binary.rs` (~6314 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~2445 tok, huge) — Copyright 2025 STARGA Inc.
- `evidence.rs` (~9867 tok, huge) — Copyright 2025 STARGA Inc.
- `map_tests.rs` (~5702 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1310 tok, large) — Copyright 2025 STARGA Inc.
- `parse.rs` (~5746 tok, huge) — Copyright 2025 STARGA Inc.
- `types.rs` (~4764 tok, huge) — Copyright 2025 STARGA Inc.
- `varint.rs` (~1599 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v3/`

- `emit.rs` (~11038 tok, huge) — Copyright 2025 STARGA Inc.
- `evidence.rs` (~10856 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~13622 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~9936 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/`

- `evidence.rs` (~1913 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~12272 tok, huge) — Copyright 2025 STARGA Inc.
- `print.rs` (~3540 tok, huge) — Copyright 2025 STARGA Inc.
- `verify.rs` (~10698 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `lib.rs` (~1006 tok, large) — Copyright 2025 STARGA Inc.
- `linalg.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/`

- `mod.rs` (~1311 tok, large) — Copyright 2025 STARGA Inc.
- `rule.rs` (~2690 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/rules/`

- `mod.rs` (~454 tok, medium) — Copyright 2025 STARGA Inc.
- `naming_convention.rs` (~1855 tok, huge) — Copyright 2025 STARGA Inc.
- `q16_overflow.rs` (~2549 tok, huge) — Copyright 2025 STARGA Inc.
- `shadowing.rs` (~1792 tok, huge) — Copyright 2025 STARGA Inc.
- `trailing_whitespace.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `unused_import.rs` (~1692 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `main.rs` (~6507 tok, huge) — Copyright 2025 STARGA Inc.
### `src/mlir/`

- `c_export.rs` (~1685 tok, huge) — Copyright 2025 STARGA Inc.
- `gemm_tuning.rs` (~3405 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~325 tok, medium) — Copyright 2025 STARGA Inc.
### `src/ops/`

- `cerebras.rs` (~2713 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `core_v1.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~228 tok, medium) — Copyright 2025 STARGA Inc.
### `src/opt/`

- `fold.rs` (~1175 tok, large) — Copyright 2025 STARGA Inc.
- `ir_canonical.rs` (~4184 tok, huge) — Copyright 2025 STARGA Inc.
- `memory_layout.rs` (~4110 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~180 tok, small) — Copyright 2025 STARGA Inc.
### `src/package/`

- `manifest.rs` (~310 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1358 tok, large) — Copyright 2025 STARGA Inc.
### `src/parser/`

- `trivia.rs` (~3811 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `pipeline.rs` (~4477 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/`

- `mod.rs` (~20623 tok, huge) — Copyright 2025 STARGA Inc.
- `module_table.rs` (~3666 tok, huge) — Copyright 2025 STARGA Inc.
- `stdlib.rs` (~3553 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `python.rs` (~1082 tok, large) — Copyright 2025 STARGA Inc.
### `src/runtime/`

- `gpu.rs` (~288 tok, medium) — Experimental GPU backend contract for MIND.
### `src/`

- `runtime_interface.rs` (~573 tok, large) — Describes a tensor visible to the runtime.
### `src/runtime/`

- `mod.rs` (~92 tok, small) — Runtime abstractions for execution backends.
- `types.rs` (~1105 tok, large) — Shared runtime surface types for execution backends.
### `src/shapes/`

- `engine.rs` (~1882 tok, huge) — A rank-N tensor shape represented as a list of extents.
- `mod.rs` (~4170 tok, huge) — Copyright 2025 STARGA Inc.
### `src/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~391 tok, medium) — Copyright 2025 STARGA Inc.
### `src/test/`

- `mod.rs` (~5999 tok, huge) — Copyright 2025 STARGA Inc.
### `src/type_checker/`

- `resolve.rs` (~8655 tok, huge) — Copyright 2025 STARGA Inc.
### `src/types/`

- `infer.rs` (~448 tok, medium) — Copyright 2025 STARGA Inc.
- `intern.rs` (~1554 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1037 tok, large) — Copyright 2025 STARGA Inc.
- `value.rs` (~297 tok, medium) — Copyright 2025 STARGA Inc.
### `src/workspace/`

- `mod.rs` (~4906 tok, huge) — Copyright 2025 STARGA Inc.
### `std/`

- `arena.mind` (~1323 tok, large) — std.arena — bump-pointer region allocator.
- `async.mind` (~2460 tok, huge) — std/async.mind -- RFC 0011 Phase A: Scheduler injection + Sender/Receiver
- `blas.mind` (~2518 tok, huge) — std/blas.mind — RFC 0006 Track A: pure-MIND surface over the six
- `cli.mind` (~2781 tok, huge) — std/cli.mind — RFC 0013 Tier 1 Phase 1: argv-parsing surface.
- `fs.mind` (~4326 tok, huge) — std/fs.mind — Task #268: POSIX filesystem surface in pure MIND.
- `io_canon.mind` (~2624 tok, huge) — std.io_canon — canonical completion ordering for deterministic I/O.
- `io.mind` (~1688 tok, huge) — std/io.mind — RFC 0005 Phase 2: pure-MIND I/O surface.
- `iouring.mind` (~18593 tok, huge) — std.iouring — minimal io_uring binding (Linux). The physical-I/O reap source
- `json.mind` (~10194 tok, huge) — std/json.mind -- RFC 8259 / ECMA-404 subset parser (task #269, cargo-retirement track).
- `llvm.mind` (~11108 tok, huge) — std/llvm.mind — RFC 0010 Phase F: hand-written MIND extern "C" bindings
- `map.mind` (~1538 tok, huge) — std/map.mind — RFC 0005 Phase 2: pure-MIND insertion-ordered map.
- `mlir.mind` (~11056 tok, huge) — std/mlir.mind — RFC 0010 Phase E: hand-written MIND extern "C" bindings
- `net.mind` (~2381 tok, huge) — std/net.mind — Task #268: POSIX socket surface in pure MIND.
- `process.mind` (~3072 tok, huge) — std/process.mind — Task #268: subprocess + process environment in pure MIND.
- `reactor.mind` (~1420 tok, large) — std.reactor — deterministic per-connection request-id allocation.
- `regex.mind` (~9377 tok, huge) — std/regex.mind -- POSIX ERE subset NFA engine (task #269, cargo-retirement track).
- `ring.mind` (~1407 tok, large) — std.ring — fixed-capacity byte ring buffer (FIFO).
- `sha256.mind` (~3635 tok, huge) — std/sha256.mind — FIPS 180-4 SHA-256 in pure MIND.
- `string.mind` (~2326 tok, huge) — std/string.mind — RFC 0005 Phase 2: pure-MIND String.
- `time.mind` (~257 tok, medium) — std.time — wall-clock access for evidence / audit timestamps.
- `toml.mind` (~9906 tok, huge) — std/toml.mind -- TOML 1.0 subset parser (task #258, cargo-retirement track).
- `tui.mind` (~4815 tok, huge) — std/tui.mind — RFC 0013 Tier 1 (c): minimal pure-MIND TUI surface.
- `vec.mind` (~1100 tok, large) — std/vec.mind — RFC 0005 Phase 2: pure-MIND growable vector.
### `tests/`

- `array_ctor_push_get_run.rs` (~907 tok, large) — Copyright 2025 STARGA Inc.
- `array_surface_run.rs` (~875 tok, large) — Copyright 2025 STARGA Inc.
### `tests/autodiff/`

- `matmul_gradient.mind` (~167 tok, small) — Autodiff test: MatMul gradient computation
### `tests/`

- `autodiff_preview.rs` (~398 tok, medium) — Copyright 2025 STARGA Inc.
- `autodiff.rs` (~2672 tok, huge) — Gradient for x*x accumulates two paths: d/dx (x*x) = x + x.
### `tests/autodiff/`

- `simple_gradient.mind` (~80 tok, small) — Autodiff test: Simple scalar gradient
### `tests/backend/`

- `cpu_available.mind` (~52 tok, small) — Backend test: CPU backend availability
- `gpu_graceful_failure.mind` (~73 tok, small) — Backend test: GPU backend graceful failure
### `tests/`

- `bare_variant_ctor_run.rs` (~885 tok, large) — Copyright 2025 STARGA Inc.
- `blas_smoke.rs` (~5888 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_q16_smoke.rs` (~5639 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_smoke.rs` (~2130 tok, huge) — Copyright 2025 STARGA Inc.
- `bool_literal_value_run.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `bytes_buffer_run.rs` (~715 tok, large) — Copyright 2025 STARGA Inc.
- `bytes_fixed_into_vec_run.rs` (~1648 tok, huge) — Copyright 2025 STARGA Inc.
- `bytes_zero_run.rs` (~889 tok, large) — Copyright 2025 STARGA Inc.
- `cerebras_stencil_tile.rs` (~1929 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `char_literal_run.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `cli_buffers.rs` (~459 tok, medium) — Copyright 2025 STARGA Inc.
- `cli_build.rs` (~648 tok, large) — Copyright 2025 STARGA Inc.
- `cli_eval.rs` (~502 tok, large) — Copyright 2025 STARGA Inc.
- `cli_exec.rs` (~558 tok, large) — Copyright 2025 STARGA Inc.
- `cli_tensor.rs` (~469 tok, medium) — Copyright 2025 STARGA Inc.
- `collection_ctor_run.rs` (~514 tok, large) — Copyright 2025 STARGA Inc.
- `collection_mutation_expr_position_run.rs` (~997 tok, large) — Copyright 2025 STARGA Inc.
### `tests/common/`

- `mod.rs` (~668 tok, large) — Copyright 2025 STARGA Inc.
### `tests/`

- `compound_assign.rs` (~1034 tok, large) — Copyright 2025 STARGA Inc.
- `cond_truthiness.rs` (~891 tok, large) — Copyright 2025 STARGA Inc.
### `tests/conformance/cpu_baseline/`

- `autodiff_pairwise.runtime` (~1 tok, tiny) — 0
- `phase_10_5_const.mind` (~34 tok, tiny) — fn main() {
- `phase_10_5_logical.mind` (~32 tok, tiny) — fn main() {
- `phase_10_5_module.mind` (~31 tok, tiny) — module governance {
- `phase_10_5_struct.mind` (~29 tok, tiny) — fn main() {
- `simple_arith.ir` (~15 tok, tiny) — module {
- `simple_arith.mind` (~3 tok, tiny)
- `simple_arith.mlir` (~25 tok, tiny) — module {
- `simple_arith.runtime` (~1 tok, tiny) — 7
### `tests/conformance/gpu_profile/`

- `backend_unavailable.error` (~9 tok, tiny) — no backend available for target gpu
- `backend_unavailable.mind` (~2 tok, tiny)
### `tests/`

- `conformance.rs` (~129 tok, small)
- `CONFORMANCE_TESTS.md` (~1225 tok, large) — MIND Conformance Test Corpus
- `const_folding.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `continue_in_match_arm_run.rs` (~1221 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_exec.rs` (~620 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~3194 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_types.rs` (~366 tok, medium) — Copyright 2025 STARGA Inc.
- `cross_module_cdylib_compose.rs` (~4112 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module_enum_run.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `cross_module_field_access_run.rs` (~1504 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module.rs` (~847 tok, large) — Copyright 2025 STARGA Inc.
### `tests/cross_substrate_identity/dot-i16-4096/`

- `manifest.toml` (~384 tok, medium) — version = "1"
- `reference_hashes.toml` (~206 tok, medium) — avx2 = "af0fc3cf1b510f8f7306a5d7250ae25a52b35281a7cefff2a0ac94b0cd80a127"
### `tests/cross_substrate_identity/dot-l1-q16/`

- `manifest.toml` (~157 tok, small) — version = "1"
- `reference_hashes.toml` (~148 tok, small) — avx2 = "ce7e2a80515e123f5d4fbb77d841f0d6c56fcbc690bba2e2ff81e45765843b34"
### `tests/cross_substrate_identity/dot-l2-q16/`

- `manifest.toml` (~436 tok, medium) — version = "1"
- `reference_hashes.toml` (~319 tok, medium) — avx2 = "1d7f272b85e5f0fd7cf473086fb1da558a723134ff02ef30a4323eb757209823"
### `tests/cross_substrate_identity/gemm-i8-64x64x64/`

- `manifest.toml` (~437 tok, medium) — version = "1"
- `reference_hashes.toml` (~212 tok, medium) — avx2 = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7"
### `tests/cross_substrate_identity/gemm-q16-64x64x64/`

- `manifest.toml` (~391 tok, medium) — version = "1"
- `reference_hashes.toml` (~167 tok, small) — avx2 = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f"
### `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/`

- `manifest.toml` (~529 tok, large) — version = "1"
- `reference_hashes.toml` (~309 tok, medium) — avx2 = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f"
### `tests/cross_substrate_identity/gemv-i16-256x256/`

- `manifest.toml` (~375 tok, medium) — version = "1"
- `reference_hashes.toml` (~161 tok, small) — avx2 = "3238e8c7e1e9ee9937503700f63eda350fcd10e7db28d470c3dbc26592d0a936"
### `tests/cross_substrate_identity/gemv-q16-256x256/`

- `manifest.toml` (~310 tok, medium) — version = "1"
- `reference_hashes.toml` (~151 tok, small) — avx2 = "dfdf890874472ee369da524955995889c39bc6da770e4e2b1d0d69315e17611a"
### `tests/cross_substrate_identity/q16-arith-chain/`

- `manifest.toml` (~432 tok, medium) — version = "1"
- `reference_hashes.toml` (~298 tok, medium) — avx2 = "ce93cdeb0e650c1c8e0cd05687ed986bbdbac691b6a8742e155b8ffd65997d78"
### `tests/cross_substrate_identity/`

- `README.md` (~1219 tok, large) — cross_substrate_identity — the internal mind-bench reproducibility gate
### `tests/`

- `cross_substrate_identity.rs` (~12755 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/cross_substrate_identity/scalar-float-f64/`

- `manifest.toml` (~778 tok, large) — version = "1"
- `reference_hashes.toml` (~460 tok, medium) — avx2 = "7592a52a5e10a2f24469765f71ce1f9f8ebd9efb51904cf9a18f310d33b3c92d"
### `tests/cross_substrate_identity/struct-handle-roundtrip/`

- `manifest.toml` (~403 tok, medium) — version = "1"
- `reference_hashes.toml` (~285 tok, medium) — avx2 = "018a335a0e9fc397c6f41cba4fc2617f0cf8d1326c5dbf77d53e27feacaeb64c"
### `tests/cross_substrate_identity/`

- `xnode_driver.c` (~2833 tok, huge)
### `tests/`

- `diagnostics_parse.rs` (~359 tok, medium) — Copyright 2025 STARGA Inc.
- `diagnostics.rs` (~688 tok, large) — Copyright 2025 STARGA Inc.
- `digit_separator_run.rs` (~651 tok, large) — Copyright 2025 STARGA Inc.
- `dot_enum_variant_run.rs` (~806 tok, large) — Copyright 2025 STARGA Inc.
- `dot_variants.rs` (~284 tok, medium) — Copyright 2025 STARGA Inc.
- `emit_ir_for_loop.rs` (~369 tok, medium) — Regression test for #4: lowering a `for` loop to IR (the path `mindc --emit-ir`
- `enum_match_collision_run.rs` (~1033 tok, large) — Copyright 2025 STARGA Inc.
- `enum_match_run.rs` (~2361 tok, huge) — Copyright 2025 STARGA Inc.
- `enum_soundness.rs` (~1411 tok, large) — Copyright 2025 STARGA Inc.
- `enum_struct_variant_run.rs` (~1331 tok, large) — Copyright 2025 STARGA Inc.
- `exec_basic.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `expr_parser.rs` (~307 tok, medium) — Copyright 2025 STARGA Inc.
- `extern_c_phase_a.rs` (~2666 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_b.rs` (~5952 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_c.rs` (~3412 tok, huge) — Copyright 2025 STARGA Inc.
- `f64_activation_lowering.rs` (~972 tok, large) — Copyright 2025 STARGA Inc.
- `ffi_header.rs` (~221 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/fixtures/`

- `autodiff.mind` (~55 tok, small) — Minimal differentiable program for the --emit-grad-ir CLI test.
- `invalid_broadcast.mind` (~17 tok, tiny)
- `invalid.mind` (~6 tok, tiny)
- `simple.mind` (~3 tok, tiny)
- `test_phase_b_all_pass.mind` (~67 tok, small) — RFC 0008 Phase B test fixture — both tests pass.
- `test_phase_b_one_fail.mind` (~80 tok, small) — RFC 0008 Phase B test fixture — one pass, one fail.
### `tests/`

- `fmt_comment_placement.rs` (~2772 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_idempotence.rs` (~2782 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_ir_preservation.rs` (~2030 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_module_block_item_preserved.rs` (~926 tok, large) — Regression: `mindc fmt` must not drop item declarations nested inside a
- `fmt_stdlib_stability.rs` (~2693 tok, huge) — Copyright 2025 STARGA Inc.
- `for_each_run.rs` (~859 tok, large) — Copyright 2025 STARGA Inc.
- `g2_differential_mlir.rs` (~6202 tok, huge) — Copyright 2025 STARGA Inc.
- `gather_preview.rs` (~288 tok, medium) — Copyright 2025 STARGA Inc.
- `generics_lowering.rs` (~1384 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `genref_phase_jb.rs` (~3326 tok, huge) — Copyright 2025 STARGA Inc.
- `grad_wrt_resolve.rs` (~736 tok, large) — Copyright 2025 STARGA Inc.
- `if_expr.rs` (~429 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_grad.rs` (~289 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_preview.rs` (~376 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_types.rs` (~248 tok, medium) — Copyright 2025 STARGA Inc.
- `int_determinism.rs` (~1226 tok, large) — Copyright 2025 STARGA Inc.
- `intra_module_call_arity.rs` (~2603 tok, huge) — Copyright 2025 STARGA Inc.
- `int_suffix_literal.rs` (~930 tok, large) — Copyright 2025 STARGA Inc.
- `invariant_block_run.rs` (~720 tok, large) — Copyright 2025 STARGA Inc.
- `invariant_check_run.rs` (~695 tok, large) — Copyright 2025 STARGA Inc.
- `ir_core.rs` (~1265 tok, large) — Ensure the unused const is kept alive in the SSA namespace but removed from code.
- `ir_load_save.rs` (~1257 tok, large) — Copyright 2025 STARGA Inc.
- `ir_lower.rs` (~1331 tok, large) — Copyright 2025 STARGA Inc.
- `ir_negative_literals.rs` (~1722 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_stub.rs` (~219 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/ir_verification/`

- `ssa_single_assignment.mind` (~46 tok, tiny) — IR verification test: SSA property validation
- `undefined_operand.mind` (~62 tok, small) — IR verification test: Undefined operand detection
### `tests/lexical/`

- `invalid_keywords_as_identifiers.mind` (~45 tok, tiny) — Lexical test: Keywords cannot be used as identifiers
- `numeric_literals.mind` (~74 tok, small) — Lexical test: Numeric literal formats
- `valid_identifiers.mind` (~72 tok, small) — Lexical test: Valid identifier formats
### `tests/`

- `linalg_grad.rs` (~315 tok, medium) — Copyright 2025 STARGA Inc.
- `linalg_preview.rs` (~291 tok, medium) — Copyright 2025 STARGA Inc.
- `lint_infrastructure.rs` (~2540 tok, huge) — Copyright 2025 STARGA Inc.
- `loop_run.rs` (~764 tok, large) — Copyright 2025 STARGA Inc.
- `loud_fail_non_i64.rs` (~745 tok, large) — Release-readiness P1.1 — the runnable-artifact ABI gate.
- `map_get_inference_run.rs` (~713 tok, large) — Copyright 2025 STARGA Inc.
- `map_runtime_run.rs` (~843 tok, large) — Copyright 2025 STARGA Inc.
- `map_surface_run.rs` (~949 tok, large) — Copyright 2025 STARGA Inc.
- `match_arm_stmt_run.rs` (~846 tok, large) — Copyright 2025 STARGA Inc.
- `method_call.rs` (~397 tok, medium) — Copyright 2025 STARGA Inc.
- `mic3_break_continue_string_roundtrip.rs` (~1611 tok, huge) — Copyright 2026 STARGA Inc.
- `mic3_cli_emit.rs` (~2431 tok, huge) — Copyright 2025 STARGA Inc.
- `mic3_const_dense_tensor_roundtrip.rs` (~842 tok, large) — Copyright 2026 STARGA Inc.
- `micb_dos_reject.rs` (~2333 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_build_phase_a.rs` (~4566 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_phase_f.rs` (~5276 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_deps_phase_de.rs` (~6823 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_doc_phase1.rs` (~2565 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/`

- `clean.mind` (~11 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
### `tests/`

- `mindcraft_check_cli.rs` (~3370 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/`

- `drifted.mind` (~12 tok, tiny) — fn add(a: i64,  b: i64) -> i64 {
### `tests/`

- `mindcraft_check_fix.rs` (~1667 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/`

- `ignored.mind` (~9 tok, tiny) — fn ignored_fn() -> i64 {
### `tests/`

- `mindcraft_check_lsp_reporter.rs` (~1968 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/subdir/`

- `nested.mind` (~9 tok, tiny) — fn nested(x: i64) -> i64 {
### `tests/mindcraft/check/`

- `with_lint.mind` (~16 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
### `tests/mindcraft/fmt/`

- `01_indent_if_else.in.mind` (~46 tok, tiny) — fn classify(x: i64) -> i64 {
- `01_indent_if_else.out.mind` (~48 tok, tiny) — fn classify(x: i64) -> i64 {
- `02_struct_literal_multiline.in.mind` (~28 tok, tiny) — fn make_point(a: i64, b: i64) -> Point {
- `02_struct_literal_multiline.out.mind` (~28 tok, tiny) — fn make_point(a: i64, b: i64) -> Point {
- `03_fn_args_multiline.in.mind` (~34 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `03_fn_args_multiline.out.mind` (~35 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `04_trailing_comma_toggle.in.mind` (~33 tok, tiny) — fn make_config(w: i64, h: i64) -> Config {
- `04_trailing_comma_toggle.out.mind` (~33 tok, tiny) — fn make_config(w: i64, h: i64) -> Config {
- `05_internal_whitespace.in.mind` (~32 tok, tiny) — fn calc(a: i64, b: i64, c: i64) -> i64 {
- `05_internal_whitespace.out.mind` (~33 tok, tiny) — fn calc(a: i64, b: i64, c: i64) -> i64 {
- `06_comment_attachment.in.mind` (~48 tok, tiny) — Copyright 2025 STARGA Inc.
- `06_comment_attachment.out.mind` (~48 tok, tiny) — Copyright 2025 STARGA Inc.
- `07_string_literal_passthrough.in.mind` (~14 tok, tiny) — fn get_message() -> i64 {
- `07_string_literal_passthrough.out.mind` (~14 tok, tiny) — fn get_message() -> i64 {
### `tests/`

- `mindcraft_fmt_cli.rs` (~2925 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fix.rs` (~1445 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fixtures.rs` (~1300 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/naming_convention/`

- `negative.mind` (~61 tok, small) — Negative fixture: all names follow canonical conventions.
- `positive_bad_const.mind` (~39 tok, tiny) — Positive fixture: const name violates SCREAMING_SNAKE_CASE.
- `positive_bad_fn.mind` (~37 tok, tiny) — Positive fixture: function name violates lower_snake_case.
- `positive_bad_struct.mind` (~39 tok, tiny) — Positive fixture: struct name violates UpperCamelCase.
### `tests/`

- `mindcraft_lint_naming_convention.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/q16_overflow/`

- `edge_constant.mind` (~54 tok, small) — Edge case: i32 * literal constant still triggers if no >>16 shift.
- `negative.mind` (~67 tok, small) — Negative fixture: proper Q16.16 multiply with >>16 narrowing.
- `positive.mind` (~70 tok, small) — Positive fixture: bare i32 * i32 without >>16 narrowing.
### `tests/`

- `mindcraft_lint_q16_overflow.rs` (~867 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/shadowing/`

- `negative.mind` (~34 tok, tiny) — Negative fixture: two different names — no shadowing.
- `positive.mind` (~53 tok, small) — Positive fixture: two `let x` bindings in the same function body.
### `tests/`

- `mindcraft_lint_shadowing.rs` (~980 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/`

- `trailing_ws_clean.mind` (~10 tok, tiny) — fn foo() -> i64 {
- `trailing_ws_dirty.mind` (~11 tok, tiny) — fn foo() -> i64 {
### `tests/mindcraft/lint/unused_import/`

- `negative.mind` (~53 tok, small) — Negative fixture: `use std.vec` is declared AND the `vec` identifier
- `positive.mind` (~46 tok, tiny) — Positive fixture: `use std.vec` is declared but no symbol from vec
### `tests/`

- `mindcraft_lint_unused_import.rs` (~708 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_vec_check.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/`

- `STABILITY_SKIP_LIST.md` (~408 tok, medium) — Formatter Stability Skip List
### `tests/`

- `mindc.rs` (~1851 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_test_phase_b.rs` (~3525 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_workspace_phase_c.rs` (~4166 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_cross_substrate/staged/`

- `manifest.tsv` (~357 tok, medium) — scalar_arith_step000	f	3735928559	64	5e39820a2a8325417e39057f19ba9bceec01bd2068c
- `scalar_accum_step000.mind` (~154 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step001.mind` (~162 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step002.mind` (~169 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step003.mind` (~179 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step004.mind` (~182 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step005.mind` (~186 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step006.mind` (~186 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_arith_step000.mind` (~159 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step001.mind` (~167 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step002.mind` (~175 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step003.mind` (~184 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step004.mind` (~188 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step005.mind` (~192 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step006.mind` (~192 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
### `tests/`

- `mlir_broadcast.rs` (~1479 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~1412 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_exec.rs` (~833 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_indexing.rs` (~414 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_linalg.rs` (~863 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_reductions.rs` (~610 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~328 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_shape.rs` (~348 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_file_and_lower.rs` (~639 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~314 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~285 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_lowering.rs` (~1490 tok, large)
- `mlir_opt.rs` (~424 tok, medium) — Copyright 2025 STARGA Inc.
- `module_const_run.rs` (~774 tok, large) — Copyright 2025 STARGA Inc.
- `module_decl_run.rs` (~643 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_call_abi.rs` (~1374 tok, large) — Copyright 2025 STARGA Inc.
- `narrowing_check.rs` (~438 tok, medium) — Regression test for the silent i64->i32 narrowing miscompile found by MIND-Fuzz
- `narrow_local_mask_run.rs` (~792 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_reassign_mask_run.rs` (~1296 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_reassign_run.rs` (~733 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_unsigned_div_zero_run.rs` (~1217 tok, large) — Copyright 2025 STARGA Inc.
- `nested_block_surface_run.rs` (~1036 tok, large) — Copyright 2025 STARGA Inc.
- `nested_collection_run.rs` (~818 tok, large) — Copyright 2025 STARGA Inc.
- `ops_registry.rs` (~114 tok, small)
- `package_basic.rs` (~491 tok, medium) — Copyright 2025 STARGA Inc.
- `parse_match_and_ref.rs` (~2780 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_phase10_surface.rs` (~4815 tok, huge) — Parse-target tests for Phase 10.5 / 10.6 surface-syntax acceptance.
- `parser_trivia.rs` (~2706 tok, huge) — Copyright 2025 STARGA Inc.
- `parser_unsigned_i64_literals.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `phase_g_keystone_bootstrap.rs` (~6252 tok, huge) — Copyright 2025 STARGA Inc.
- `pipeline.rs` (~1476 tok, large) — Copyright 2025 STARGA Inc.
- `reap_threshold.rs` (~2047 tok, huge) — Copyright 2025 STARGA Inc.
- `reductions_grad.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `reductions_preview.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `region_phase_ja.rs` (~4228 tok, huge) — Copyright 2025 STARGA Inc.
- `relu_exec.rs` (~472 tok, medium) — Copyright 2025 STARGA Inc.
- `relu_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `repl_basic.rs` (~523 tok, large) — Copyright 2025 STARGA Inc.
- `resolve_fn_body.rs` (~979 tok, large) — Copyright 2025 STARGA Inc.
- `result_option_prelude_run.rs` (~910 tok, large) — Copyright 2025 STARGA Inc.
- `rfc0012_attribute_syntax.rs` (~1182 tok, large) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_a_shape_types.rs` (~6540 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_b_operators.rs` (~4514 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_c_annotations.rs` (~2394 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/runtime/`

- `elementwise_add.mind` (~68 tok, small) — Runtime test: Element-wise addition execution
- `reduction_sum.mind` (~67 tok, small) — Runtime test: Reduction sum operation
### `tests/`

- `scalar_cast_call_run.rs` (~719 tok, large) — Copyright 2025 STARGA Inc.
- `scalar_cast_unsigned_narrow_run.rs` (~1092 tok, large) — Copyright 2025 STARGA Inc.
### `tests/selfhost_gaps/`

- `call-arg-nesting_1.mind` (~25 tok, tiny)
- `call-arg-nesting_2.mind` (~35 tok, tiny)
- `call-arg-nesting_3.mind` (~41 tok, tiny)
- `call-arg-nesting_4.mind` (~48 tok, tiny)
- `call-arg-nesting_5.mind` (~20 tok, tiny)
- `call-arg-nesting_6.mind` (~28 tok, tiny)
- `call-arg-nesting_7.mind` (~28 tok, tiny)
- `call-arg-nesting_8.mind` (~50 tok, small)
- `call-arg-nesting_9.mind` (~10 tok, tiny)
- `deep-combos_1.mind` (~42 tok, tiny)
- `deep-combos_2.mind` (~42 tok, tiny)
- `deep-combos_3.mind` (~35 tok, tiny)
- `deep-combos_4.mind` (~35 tok, tiny)
- `deep-combos_5.mind` (~69 tok, small)
- `deep-combos_6.mind` (~128 tok, small)
- `deep-combos_7.mind` (~87 tok, small)
- `deep-combos_8.mind` (~14 tok, tiny)
- `fallthrough-shadow_1.mind` (~28 tok, tiny)
- `fallthrough-shadow_2.mind` (~42 tok, tiny)
- `fallthrough-shadow_3.mind` (~55 tok, small)
- `fallthrough-shadow_4.mind` (~29 tok, tiny)
- `fallthrough-shadow_5.mind` (~57 tok, small)
- `fallthrough-shadow_6.mind` (~38 tok, tiny)
- `fallthrough-shadow_7.mind` (~33 tok, tiny)
- `fallthrough-shadow_8.mind` (~25 tok, tiny)
- `field-read_1.mind` (~63 tok, small)
- `field-read_2.mind` (~41 tok, tiny)
- `field-read_3.mind` (~27 tok, tiny)
- `field-read_4.mind` (~29 tok, tiny)
- `field-read_5.mind` (~61 tok, small)
- `GAPS.md` (~2429 tok, huge) — Self-host nfn driver — gap inventory (fuzz-discovered)
- `let-ifexpr-seq_1.mind` (~23 tok, tiny)
- `let-ifexpr-seq_2.mind` (~34 tok, tiny)
- `let-ifexpr-seq_3.mind` (~23 tok, tiny)
- `let-ifexpr-seq_4.mind` (~21 tok, tiny)
- `let-ifexpr-seq_5.mind` (~19 tok, tiny)
- `let-ifexpr-seq_6.mind` (~27 tok, tiny)
- `let-ifexpr-seq_7.mind` (~46 tok, tiny)
- `mixed-prefix_10.mind` (~32 tok, tiny)
- `mixed-prefix_11.mind` (~27 tok, tiny)
- `mixed-prefix_12.mind` (~40 tok, tiny)
- `mixed-prefix_1.mind` (~28 tok, tiny)
- `mixed-prefix_2.mind` (~38 tok, tiny)
- `mixed-prefix_3.mind` (~65 tok, small)
- `mixed-prefix_4.mind` (~59 tok, small)
- `mixed-prefix_5.mind` (~41 tok, tiny)
- `mixed-prefix_6.mind` (~38 tok, tiny)
- `mixed-prefix_7.mind` (~29 tok, tiny)
- `mixed-prefix_8.mind` (~49 tok, tiny)
- `mixed-prefix_9.mind` (~50 tok, small)
- `operator-edges_1.mind` (~24 tok, tiny)
- `operator-edges_2.mind` (~26 tok, tiny)
- `operator-edges_3.mind` (~26 tok, tiny)
- `operator-edges_4.mind` (~8 tok, tiny)
- `operator-edges_5.mind` (~16 tok, tiny)
- `operator-edges_6.mind` (~14 tok, tiny)
- `struct-lit_1.mind` (~63 tok, small)
- `struct-lit_2.mind` (~56 tok, small)
- `struct-lit_3.mind` (~60 tok, small)
- `value-ifexpr_1.mind` (~98 tok, small) — MISMATCH: a `let`-block in a NESTED (else-if) branch of a value if-expr.
- `value-ifexpr_2.mind` (~79 tok, small) — MISMATCH: same-named `let` in two SIBLING branches of a value if-expr.
- `value-ifexpr_3.mind` (~71 tok, small) — MISMATCH: a `let` inside a NESTED if-expr that sits in the THEN-side of an
- `value-ifexpr_4.mind` (~75 tok, small) — MISMATCH: let-block then-side whose trailing value is a nested if-expr that
- `value-ifexpr_5.mind` (~83 tok, small) — FAIL_CLOSED (in-subset): value if-expr whose else-branch is
- `value-ifexpr_6.mind` (~77 tok, small) — FAIL_CLOSED (in-subset): `let outer; if .. { use outer } else { use outer }`
- `value-ifexpr_7.mind` (~78 tok, small) — FAIL_CLOSED (in-subset): struct-lit construction as a value if-expr branch.
- `value-ifexpr_8.mind` (~64 tok, small) — FAIL_CLOSED (in-subset): field-read `recv.field` as a value if-expr branch.
### `tests/`

- `set_surface_run.rs` (~931 tok, large) — Copyright 2025 STARGA Inc.
- `sha256_smoke.rs` (~1555 tok, huge) — Copyright 2025 STARGA Inc.
- `shape_integration.rs` (~416 tok, medium)
- `shape_ops_preview.rs` (~302 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/shapes/`

- `broadcast_compatible.mind` (~77 tok, small) — Shape test: Compatible broadcasting
- `broadcast_incompatible.mind` (~76 tok, small) — Shape test: Incompatible broadcasting
### `tests/`

- `shapes_engine.rs` (~699 tok, large) — Rank-0 scalar represented as an empty shape.
### `tests/shapes/`

- `matmul_shapes.mind` (~107 tok, small) — Shape test: MatMul shape inference
### `tests/`

- `shapes.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `smoke.rs` (~259 tok, medium) — Copyright 2025 STARGA Inc.
- `sparse_tensor_types.rs` (~1960 tok, huge) — Copyright 2025 STARGA Inc.
- `statement_mutation_run.rs` (~957 tok, large) — Copyright 2025 STARGA Inc.
- `std_import_standalone_run.rs` (~1151 tok, large) — Copyright 2025 STARGA Inc.
- `stdlib_tensor.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `std_llvm_bindings_smoke.rs` (~2606 tok, huge) — Copyright 2025 STARGA Inc.
- `std_mlir_bindings_smoke.rs` (~4774 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_arena.rs` (~1342 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_array_literals.rs` (~1334 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_async.rs` (~4296 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bitwise_binops.rs` (~2391 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bool_return.rs` (~1247 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_break_continue.rs` (~1314 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_call_lowering.rs` (~834 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cdylib_link.rs` (~2078 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_cli_equals_form.rs` (~871 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli.rs` (~1005 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli_subcommand.rs` (~796 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_field_access.rs` (~2912 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_field_access_step2.rs` (~3401 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_fndef_lowering.rs` (~1412 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_i32_intrinsics.rs` (~808 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_if_statement.rs` (~3282 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_intrinsics.rs` (~2350 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_ansi.rs` (~790 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_io_canon.rs` (~3266 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_module.rs` (~1540 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_iouring.rs` (~3066 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_json.rs` (~4343 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_logical_ops.rs` (~1187 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_map_module.rs` (~2015 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_method_call.rs` (~2939 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_net_fs_process.rs` (~7308 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_phase_c_stdlib_bundle.rs` (~1463 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_phase_d_env_override.rs` (~1597 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_promotion_compose.rs` (~5551 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_reactor.rs` (~1357 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_regex.rs` (~4835 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_ring.rs` (~1310 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_self_emit_shared.rs` (~1795 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_itoa.rs` (~931 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_string_module.rs` (~2284 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_push_str.rs` (~861 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_struct_lowering.rs` (~2712 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_toml.rs` (~4146 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_tui.rs` (~2256 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import_phase_b.rs` (~2198 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import.rs` (~1753 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_module.rs` (~1805 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_zeroed.rs` (~1077 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_while_statement.rs` (~3201 tok, huge) — Copyright 2025 STARGA Inc.
- `stride_gather_grad.rs` (~312 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_types.rs` (~250 tok, medium) — Copyright 2025 STARGA Inc.
- `string_escape_decode_run.rs` (~1209 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_parse.rs` (~483 tok, medium) — Copyright 2025 STARGA Inc.
- `string_from_bytes_run.rs` (~709 tok, large) — Copyright 2025 STARGA Inc.
- `string_runtime_shim_run.rs` (~1027 tok, large) — Copyright 2025 STARGA Inc.
- `string_split_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc.
- `struct_array_field_run.rs` (~847 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_collection_run.rs` (~854 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_in_loop_run.rs` (~1089 tok, large) — Copyright 2025 STARGA Inc.
- `struct_narrow_field.rs` (~821 tok, large) — Copyright 2025 STARGA Inc.
- `target_cerebras.rs` (~340 tok, medium) — Cerebras backend target — first-class surface tests.
- `tensor_broadcast.rs` (~1004 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_buffers.rs` (~517 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_eval.rs` (~457 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_param_fail_loud_run.rs` (~1572 tok, huge) — Copyright 2025 STARGA Inc.
- `tensor_stdlib.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_symbolic.rs` (~550 tok, large) — Copyright 2025 STARGA Inc.
- `transpose_preview.rs` (~269 tok, medium) — Copyright 2025 STARGA Inc.
- `tuple_destructure_run.rs` (~1162 tok, large) — Copyright 2025 STARGA Inc.
- `type_ann_check.rs` (~330 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_parse.rs` (~584 tok, large) — Copyright 2025 STARGA Inc.
- `typecheck_binary.rs` (~327 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_env.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/type_checker/`

- `basic_type_inference.mind` (~66 tok, small) — Type checker test: Basic type inference
- `dtype_mismatch.mind` (~74 tok, small) — Type checker test: Dtype mismatch detection
### `tests/`

- `typed_literal_match_pattern_run.rs` (~764 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `type_error_spans.rs` (~1064 tok, large) — Copyright 2025 STARGA Inc.
- `type_infer.rs` (~344 tok, medium) — Copyright 2025 STARGA Inc.
- `type_struct_run.rs` (~628 tok, large) — Copyright 2025 STARGA Inc.
- `value_if_comparison.rs` (~813 tok, large) — Copyright 2025 STARGA Inc.
- `value_if_f64_let.rs` (~1148 tok, large) — Copyright 2025 STARGA Inc.
- `vars_assign.rs` (~260 tok, medium) — Copyright 2025 STARGA Inc.
- `verify_audit.rs` (~2010 tok, huge) — Audit coverage tests for the IR verifier (C1: SSA verification, conv2d stride/axis validation).
- `verify_cli.rs` (~2377 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_ssa.rs` (~6332 tok, huge) — Copyright 2025 STARGA Inc.
### `tools/`

- `add_copyright_headers.py` (~1132 tok, large) — # Copyright 2025 STARGA Inc.
- `bench_gate.py` (~2344 tok, huge) — # Copyright 2025 STARGA Inc.
- `cargo-deny-sanitize.sh` (~572 tok, large) — Run cargo-deny but sanitize advisory entries that older cargo-deny versions
### `tools/mindfuzz/`

- `ci_batch.py` (~1186 tok, large) — MIND-Fuzz deterministic CI batch -> cross-substrate candidate staging.
- `fuzz_loop.py` (~3529 tok, huge) — MIND-Fuzz loop -- LLM-mutation differential testing for the MIND compiler.
- `.gitignore` (~7 tok, tiny) — tools/mindfuzz/__pycache__/
- `mutate.py` (~3631 tok, huge) — MIND-Fuzz mutation engine.
- `mutations.txt` (~1765 tok, huge) — # MIND-Fuzz mutation instructions (adapted from arXiv:2501.00655 Table 1).
- `oracles.py` (~4514 tok, huge) — MIND-Fuzz differential oracles.
- `README.md` (~1360 tok, large) — MIND-Fuzz
### `tools/mindfuzz/seeds/`

- `cast_edge.mind` (~241 tok, medium) — MIND-Fuzz seed: type-conversion + edge-value scalar entry (reference-checkable).
- `control_flow.mind` (~312 tok, medium) — MIND-Fuzz seed: control-flow-heavy program (loops, nested if, early return,
- `dot_q16.mind` (~221 tok, medium) — MIND-Fuzz seed: Q16.16 dot / L1 / gemv kernels.
- `multi_fn.mind` (~243 tok, medium) — MIND-Fuzz seed: multi-function program with a reference-checkable scalar entry.
- `scalar_accum.mind` (~154 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_arith.mind` (~159 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
### `tools/mindfuzz/violations/`

- `.gitkeep` (~0 tok, tiny)
### `tools/pytorch_bridge/`

- `ai_proof.py` (~640 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `.gitignore` (~4 tok, tiny) — __pycache__/
- `__init__.py` (~384 tok, medium) — # Copyright 2025-2026 STARGA Inc.
- `ir.py` (~920 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `jax.py` (~1007 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `pytorch.py` (~1718 tok, huge) — # Copyright 2025-2026 STARGA Inc.
### `tools/pytorch_bridge/tests/`

- `__init__.py` (~0 tok, tiny)
- `test_bridge.py` (~1244 tok, large) — # Copyright 2025-2026 STARGA Inc.
### `tools/`

- `run_bench_gate.sh` (~530 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `.wrangler/cache/`

- `wrangler-account.json` (~21 tok, tiny) — Keys: account

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*
