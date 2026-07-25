# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 3212 | **Est. tokens:** ~7,737,080
**Generated:** 2026-07-25 23:05 UTC

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
| `./` | 33 | ~25,641 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~644 |
| `audits/` | 6 | ~607 |
| `bench/` | 2 | ~1,772 |
| `benches/` | 27 | ~80,382 |
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
| `config/` | 1 | ~1,450 |
| `docs/` | 32 | ~70,235 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,315 |
| `docs/design/` | 3 | ~8,181 |
| `docs/mindcraft/` | 3 | ~7,086 |
| `docs/rfcs/` | 31 | ~143,431 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 28 | ~49,880 |
| `examples/bimap_currency/` | 3 | ~780 |
| `examples/bimap_pairs/` | 2 | ~801 |
| `examples/c/` | 2 | ~400 |
| `examples/columnar/` | 4 | ~7,585 |
| `examples/compliance/` | 3 | ~5,294 |
| `examples/distribution-crossisa/` | 6 | ~6,336 |
| `examples/emit_ir/` | 5 | ~13,648 |
| `examples/grammar_mask/` | 2 | ~4,636 |
| `examples/halbach_q16/` | 2 | ~7,856 |
| `examples/lexer/` | 6 | ~8,888 |
| `examples/mindc_mind/` | 139 | ~320,924 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~915 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 1 | ~102 |
| `examples/native/` | 4 | ~527 |
| `examples/parser/` | 5 | ~17,923 |
| `examples/typecheck/` | 5 | ~14,553 |
| `examples/zoo/` | 6 | ~12,518 |
| `experiments/global-vs-local/` | 7 | ~6,492 |
| `.githooks/` | 1 | ~255 |
| `.github/` | 3 | ~149 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 9 | ~15,354 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 1 | ~18,670 |
| `scripts/` | 9 | ~11,565 |
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
| `src/` | 7 | ~18,310 |
| `src/ast/` | 1 | ~8,927 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 2 | ~34,966 |
| `src/build/` | 2 | ~16,388 |
| `src/cache/` | 4 | ~3,682 |
| `src/check/` | 3 | ~10,829 |
| `src/deps/` | 1 | ~9,345 |
| `src/diagnostics/` | 1 | ~3,719 |
| `src/distributed/` | 6 | ~7,433 |
| `src/doc/` | 3 | ~10,987 |
| `src/eval/` | 12 | ~65,572 |
| `src/eval/stdlib/` | 2 | ~8,529 |
| `src/exec/` | 3 | ~4,592 |
| `src/ffi/` | 3 | ~3,919 |
| `src/fmt/` | 3 | ~20,612 |
| `src/ir/` | 5 | ~46,820 |
| `src/ir/compact/` | 3 | ~15,267 |
| `src/ir/compact/v2/` | 8 | ~38,037 |
| `src/ir/compact/v3/` | 6 | ~48,342 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~9,864 |
| `src/mlir/` | 3 | ~5,905 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 7 | ~36,960 |
| `src/package/` | 2 | ~1,877 |
| `src/parser/` | 2 | ~20,355 |
| `src/phf/` | 1 | ~4,955 |
| `src/project/` | 2 | ~8,218 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 1 | ~5,979 |
| `src/type_checker/` | 1 | ~12,882 |
| `src/types/` | 4 | ~3,336 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 41 | ~194,284 |
| `tests/` | 292 | ~502,307 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 1 | ~668 |
| `tests/conformance/cpu_baseline/` | 9 | ~171 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4,113 |
| `tests/cross_substrate_identity/bimap-phf/` | 2 | ~1,418 |
| `tests/cross_substrate_identity/collatz/` | 2 | ~962 |
| `tests/cross_substrate_identity/dot-f32-v-4093/` | 2 | ~1,222 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~648 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~363 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~813 |
| `tests/cross_substrate_identity/galperin-pi/` | 2 | ~1,004 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~707 |
| `tests/cross_substrate_identity/gemm-i8-mt-64x64x64/` | 2 | ~872 |
| `tests/cross_substrate_identity/gemm-i8-vnni-64x64x64/` | 2 | ~921 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~616 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~896 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~594 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~519 |
| `tests/cross_substrate_identity/grammar-mask/` | 2 | ~916 |
| `tests/cross_substrate_identity/lorenz-q16/` | 2 | ~1,243 |
| `tests/cross_substrate_identity/matmul-f32-v-64x64/` | 2 | ~1,115 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~788 |
| `tests/cross_substrate_identity/scalar-cast-conv/` | 2 | ~1,644 |
| `tests/cross_substrate_identity/scalar-cast-conv-narrow/` | 2 | ~1,790 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1,310 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~746 |
| `tests/cross_substrate_identity/u64-ops/` | 2 | ~1,054 |
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
| `tests/mindfuzz_cross_substrate/staged/` | 16 | ~2,989 |
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

## Files

### `./`

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
- `Cargo.toml` (~2110 tok, huge) — [package]
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~1348 tok, large) — Contributing to MIND
- `deny.toml` (~89 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~130 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~599 tok, large) — # Rust
- `incompatible` (~0 tok, tiny)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `Mind.toml` (~108 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~5837 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~1256 tok, large) — Security Policy
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `STATUS.md` (~4409 tok, huge) — MIND Compiler Status
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
- `bench_hkdf.rs` (~4424 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hpack.rs` (~3927 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_http2_frame.rs` (~5035 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_keccak.rs` (~2576 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_mlkem768.rs` (~3700 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_rsa_pss.rs` (~3317 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_sha256.rs` (~2594 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_tls13_record.rs` (~6121 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x25519.rs` (~2469 tok, huge) — Copyright 2025-2026 STARGA Inc.
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
- `parser_throughput.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `shapes.rs` (~1208 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~707 tok, large) — Mirror mindc's allocator so this compile-speed bench measures the same heap
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

- `matmul_det_bench.mind` (~1079 tok, large) — bench/matmul_det_bench.mind — first pure-MIND runtime benchmark for the
- `RESULTS-int8-2026-06-08.md` (~693 tok, large) — MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `config/`

- `capabilities.toml` (~1450 tok, large) — [ir]
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

- `determinism.md` (~3956 tok, huge) — The Determinism Contract
- `errors.md` (~701 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
- `gpu.md` (~387 tok, medium) — GPU backend profile
- `INDEPENDENCE_ROADMAP.md` (~15229 tok, huge) — MIND Rust-Independence Roadmap
- `install.md` (~1012 tok, large) — Installing mindc
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1485 tok, large) — IR stability contract
- `migration-roadmap.md` (~1896 tok, huge) — MIND Migration Roadmap — Any Language → Pure Executing MIND
### `docs/mindcraft/`

- `fmt.md` (~2302 tok, huge) — `mindc fmt` — Canonical Formatter Reference
- `phase2-implementation-plan.md` (~2209 tok, huge) — Mindcraft Phase 2 — Implementation Plan
- `rfc0010-phase-ghi-migration-plan.md` (~2575 tok, huge) — RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)
### `docs/`

- `mlir-lowering.md` (~286 tok, medium) — MLIR lowering pipeline (public)
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `optimization-frontier.md` (~11347 tok, huge) — MIND Optimization Frontier
- `performance.md` (~880 tok, large) — Performance Guide
- `README.md` (~162 tok, small) — MIND Documentation
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
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
- `0007-mindcraft.md` (~4499 tok, huge) — RFC 0007: Mindcraft — the pure-MIND format / lint / check toolchain
- `0008-mindc-build.md` (~10964 tok, huge) — RFC 0008: mindc build + mindc test — retiring cargo from the build path
- `0009-federation-package-layer.md` (~6976 tok, huge) — RFC 0009: Federation-First MIND Package Layer
- `000-template.md` (~1 tok, tiny)
- `0010-memory-safety-and-c-abi.md` (~7359 tok, huge) — RFC 0010: Memory Safety Model + C ABI in Pure MIND
- `0011-async-and-structured-concurrency.md` (~4891 tok, huge) — RFC 0011: Async + Structured Concurrency Model
- `0012-tensor-native-syntax.md` (~11307 tok, huge) — RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer
- `0013-cli-agent-harness-stack.md` (~6781 tok, huge) — RFC 0013: CLI Agent Harness Stack
- `0014-per-substrate-mlir-lowering-contracts.md` (~5412 tok, huge) — RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts
- `0015-cross-substrate-bit-identity.md` (~5174 tok, huge) — RFC 0015: Cross-Substrate Bit-Identity Proof Obligation
- `0016-evidence-chain-emission.md` (~6944 tok, huge) — RFC 0016: Compile-Time Evidence-Chain Emission
- `0017-mindc-verify.md` (~3745 tok, huge) — RFC 0017: `mindc verify` — Artifact Verification Surface
- `0018-bare-metal-substrate.md` (~3799 tok, huge) — RFC 0018: Bare-Metal Substrate Lowering Tier
- `0019-deterministic-agent-substrate.md` (~4131 tok, huge) — RFC 0019: Deterministic Agent Substrate
- `0020-mind-bench-reproducibility-harness.md` (~4083 tok, huge) — RFC 0020: mind-bench Public Reproducibility Harness
- `0021-canonical-ir-unification.md` (~4388 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2120 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `0024-loop-collapse.md` (~7516 tok, huge) — RFC 0024: Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`)
- `DRAFT-deterministic-format-frontend.md` (~10507 tok, huge) — RFC DRAFT: Deterministic Multi-Format Ingest Front-End (JSON / TOON / CSV / TSV / NDJSON / TOML)
- `DRAFT-deterministic-json-frontend.md` (~5175 tok, huge) — RFC DRAFT: Deterministic Streaming SIMD JSON Structural Front-End
- `odc-language-primitives.md` (~422 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

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
- `VERIFICATION_APPARATUS.md` (~7783 tok, huge) — Self-Host Port Verification Apparatus & SOTA Roadmap
- `versioning.md` (~804 tok, large) — MIND Core Stability & Versioning
- `version-matrix.md` (~1796 tok, huge) — MIND Ecosystem — Version Matrix
- `whitepaper.md` (~2788 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `anthropobrot.mind` (~3257 tok, huge) — Anthropobrot: depth-selected orbit-density multisets of the Fatou-Julia iteral.
- `autodiff_demo.mind` (~1715 tok, huge) — Autodiff Demonstration
### `examples/bimap_currency/`

- `.gitignore` (~2 tok, tiny) — target/
- `main.mind` (~716 tok, large) — Single-source bijective map over a "nice set" — ONE declaration, both
- `Mind.toml` (~62 tok, small) — [package]
### `examples/bimap_pairs/`

- `main.mind` (~732 tok, large) — Single-source bijective const pair-tables — ONE declaration, both directions
- `Mind.toml` (~69 tok, small) — [package]
### `examples/c/`

- `min.c` (~82 tok, small)
- `mind.h` (~318 tok, medium) — Copyright 2025 STARGA Inc.
### `examples/`

- `cnn_classifier.mind` (~1060 tok, large) — CNN Classifier Example
- `collatz.mind` (~495 tok, medium) — Deterministic integer Collatz (3n+1) iterator — the integer sibling of the
### `examples/columnar/`

- `structural_scan_json.mind` (~1703 tok, huge) — examples/columnar/structural_scan_json.mind
- `structural_scan_test.py` (~1541 tok, huge) — Runnable verification for examples/columnar/structural_scan_json.mind.
- `tiled_fold.mind` (~1784 tok, huge) — examples/columnar/tiled_fold.mind
- `tiled_fold_test.py` (~2557 tok, huge) — Runnable verification for examples/columnar/tiled_fold.mind.
### `examples/compliance/`

- `auditable_model.mind` (~1932 tok, huge) — auditable_model.mind -- Compliance-Ready MLP with Provenance Metadata
- `audit_report.mind` (~2289 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `README.md` (~1073 tok, large) — Compliance Example
### `examples/`

- `cos_dottie.mind` (~781 tok, large) — Cosine-map iteration toward the Dottie fixed point (x* ≈ 0.7390851332151607),
### `examples/distribution-crossisa/`

- `afterkelly.cpp` (~2278 tok, huge) — Command line arguments. ____________________________________________
- `data1.txt` (~212 tok, medium) — 45.96
- `data2.txt` (~223 tok, medium) — 107.50
- `distribution.cpp` (~1217 tok, large)
- `distribution_interp_f64.mind` (~1231 tok, large) — Deterministic IEEE-754 float64 piecewise-LINEAR density interpolation kernel,
- `README.md` (~1175 tok, large) — Cross-ISA determinism: a piecewise-linear density kernel
### `examples/`

- `dottie_collapse.mind` (~1144 tok, large) — Salov loop-collapse — Q16.16 fixed-point ITERATION collapse (Slice S3).
### `examples/emit_ir/`

- `bootstrap_smoke.py` (~2890 tok, huge)
- `EXPECTED.md` (~1942 tok, huge) — Phase 6.4 — Expected IR Text
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `main.mind` (~6419 tok, huge) — examples/emit_ir/main.mind — RFC 0005 Phase 6.4 self-host MLIR text emitter.
- `README.md` (~2214 tok, huge) — RFC 0005 Phase 6.4 — Self-Host MLIR Text Emitter
### `examples/`

- `fft_q16.mind` (~1248 tok, large) — Deterministic Q16.16 fixed-point radix-2 DIT FFT, N=256 (complex).
- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
- `galperin_pi.mind` (~1486 tok, large) — Galperin's billiard-π: count elastic collisions of two balls + a wall to
- `gauss_collapse.mind` (~672 tok, large) — Salov loop-collapse — closed-form affine sums (Slice S1).
- `geometric_collapse.mind` (~807 tok, large) — Salov loop-collapse — geometric powering closed forms (Slice S2).
### `examples/grammar_mask/`

- `main.mind` (~4577 tok, huge) — examples/grammar_mask/main.mind — structured / grammar-constrained decoding,
- `Mind.toml` (~59 tok, small) — [package]
### `examples/halbach_q16/`

- `main.mind` (~7793 tok, huge) — examples/halbach_q16/main.mind — standalone, SELF-VERIFYING project build of
### `examples/`

- `halbach_q16.mind` (~3965 tok, huge) — Deterministic Q16.16 2D Halbach-vs-uniform magnet-array field model.
### `examples/halbach_q16/`

- `Mind.toml` (~63 tok, small) — [package]
### `examples/`

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
- `mandelbrot.mind` (~1019 tok, large) — Deterministic IEEE-754 float64 Mandelbrot escape-count renderer.
- `mandelbrot_strict.mind` (~982 tok, large) — Strict-f64 Mandelbrot escape-count checksum — a determinism-wedge demo.
### `examples/mindc_mind/`

- `bootstrap_smoke.py` (~2329 tok, huge)
- `check_driver.mind` (~3938 tok, huge) — ===========================================================================
- `closure_netverify.py` (~1414 tok, large) — # Canonical independent net-verify harness for CLOSURES / FN-VALUES / UNRESOLVED
- `collect_field_strings_smoke.py` (~1161 tok, large)
- `cutover_coverage_measure.py` (~2238 tok, huge)
- `div_shift_cmp_edge_smoke.py` (~1846 tok, huge)
- `enum_netverify.py` (~1840 tok, huge) — # Canonical independent net-verify harness for C-LIKE ENUMS in the native-ELF backend.
- `EXPECTED.md` (~773 tok, large) — Phase 6.5 Stage 5 — Expected IR Text (APEX)
- `fast_keystone.sh` (~3154 tok, huge) — fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
- `field_store_netverify.py` (~1404 tok, large) — # Canonical independent value harness for struct field STORES (`p.x = v`) in the
- `FIXED_POINT_REPORT.md` (~1770 tok, huge) — Phase 6.5 — Bootstrap Fixed-Point Report
- `fixed_point_smoke.py` (~3275 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `full_strtab_smoke.py` (~1663 tok, huge)
- `gap_corpus_smoke.py` (~1922 tok, huge)
- `general_float_netverify.py` (~1779 tok, huge) — general_float_netverify.py — GENERAL-path f64 value battery (B0 gate lift).
- `.gitignore` (~5 tok, tiny) — __pycache__/
- `lockstep_lint.py` (~3922 tok, huge) — lockstep_lint.py -- native-ELF walker lockstep linter for the pure-MIND self-host compiler.
- `match_struct_smoke.py` (~1311 tok, large)
- `method_callee_smoke.py` (~1350 tok, large)
- `method_calls_smoke.py` (~1294 tok, large)
- `mic3_flip_smoke.py` (~1150 tok, large)
- `mic3_oracle_smoke.py` (~764 tok, large) — mic@3 self-host convergence — Phase 0 gate: the Rust oracle.
- `mic3_primitives_smoke.py` (~22501 tok, huge) — mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.
- `mod_operator_smoke.py` (~2100 tok, huge)
- `multi_let_smoke.py` (~1499 tok, large)
- `now_ns_smoke.py` (~678 tok, large) — # Copyright 2025 STARGA Inc.
- `option_netverify.py` (~2312 tok, huge) — # Canonical independent net-verify harness for SINGLE-PAYLOAD ENUMS (the
- `oracle_parity_lint.py` (~3627 tok, huge)
- `param_types_smoke.py` (~1273 tok, large)
- `_ref_add.note` (~16 tok, tiny) — 757f339973d282495a5d15f72a761b6baf3a6b38dc08deb95400e03318bc5de0
- `_ref_if_ret.note` (~16 tok, tiny) — d042c5006591bd69365074c242559b25d031bb550f0d282255a57f3563c1ff45
- `_ref_main.note` (~16 tok, tiny) — 52d6b1210f0b294848c30b4f7a96c4b203e2d560126f31efde31de931da34b50
- `ref_netverify.py` (~1464 tok, large) — # Canonical independent net-verify harness for i64 references in the native-ELF backend.
- `_ref_recursion.note` (~16 tok, tiny) — 320e76629d23b18074bd73d2a0849074be06099452bb530ed19e0657f32c6fc5
- `_ref_struct_field.note` (~16 tok, tiny) — 062dd03998de380436f501819bf8ee1e05901d427a2f0ed3614310d020aa0e1b
- `_ref_value_if.note` (~16 tok, tiny) — 5abd28b3622a896e51617f2bbe0f6976231cb607664c0471c4a3dbebfc5e72d5
- `self_host_andor_smoke.py` (~2065 tok, huge) — Permanent battery for the self-host `&&` / `||` short-circuit operators.
- `self_host_arena_growth_smoke.py` (~1382 tok, large)
- `self_host_args_from_os_smoke.py` (~1356 tok, large)
- `selfhost_argv_driver.mind` (~1187 tok, large) — ===========================================================================
- `self_host_argv_smoke.py` (~1136 tok, large)
- `self_host_body_smoke.py` (~3055 tok, huge)
- `self_host_carry_cap_smoke.py` (~1088 tok, large) — Cap-guard smoke for the self-host loop-carry / loop-frame scratch tables.
- `self_host_check_driver_smoke.py` (~2045 tok, huge)
- `selfhost_driver.mind` (~623 tok, large) — ===========================================================================
- `self_host_dtype_tag_smoke.py` (~780 tok, large) — RI-B1 per-SSA dtype-tag gate (parser <-> nb_fp_* encoder connecting construct).
- `self_host_else_if_smoke.py` (~1715 tok, huge)
- `self_host_failclosed_smoke.py` (~7852 tok, huge) — self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
- `self_host_float_lit_exact_smoke.py` (~1069 tok, large) — CPU-as-oracle smoke for the C1 float-literal exactness guard.
- `self_host_for_smoke.py` (~1891 tok, huge) — Permanent battery for the self-host range-`for` loop.
- `self_host_if_region_carry_smoke.py` (~4552 tok, huge) — Native-ELF smoke: i64 loop-carry through BRANCHED regions (Sub-step C).
- `self_host_lockstep_smoke.py` (~2161 tok, huge) — SUB-STEP A lockstep smoke: the loop-carry frame COUNT and the loop-carry EMIT are
- `self_host_loop_smoke.py` (~3420 tok, huge)
- `self_host_match_smoke.py` (~1893 tok, huge)
- `self_host_mlir_smoke.py` (~1736 tok, huge)
- `self_host_narrow_param_smoke.py` (~11195 tok, huge) — Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.
- `self_host_native_autowrap_smoke.py` (~2442 tok, huge) — Roadmap C2 declared-width AUTO-WRAP driver — narrow-int (i8/i16/i32) `let` and
- `self_host_native_avx2_dot_f32_smoke.py` (~1559 tok, huge) — RI-B2-S13 (#108) — native-ELF PACKED-f32 SIMD via 256-bit AVX2 (VEX/YMM) STRICT-FP DOT.
- `self_host_native_blas_dot_i16_smoke.py` (~2879 tok, huge)
- `self_host_native_blas_dot_q16_smoke.py` (~1774 tok, huge)
- `self_host_native_cast_conv_smoke.py` (~1267 tok, large) — RI-B2 scalar-cast-conv rung (#108) — native-ELF int<->float `as`-cast chain.
- `self_host_native_dot_f32_smoke.py` (~1334 tok, large) — RI-B2-S8 STEP C (#108) — native-ELF scalar STRICT-FP f32 DOT-PRODUCT.
- `self_host_native_dot_l1_q16_smoke.py` (~1088 tok, large) — RI-B2 L1-Q16 rung (#108) — native-ELF Q16.16 L1 distance.
- `self_host_native_elf_smoke.py` (~10783 tok, huge)
- `self_host_native_fp_binop_smoke.py` (~1095 tok, large) — RI-B1 nb_expr FLOAT-op-FLOAT arithmetic routing gate (zero MLIR/LLVM).
- `self_host_native_fp_call_smoke.py` (~5471 tok, huge) — RI-D2 S-C1: FLOAT call-RETURN dtype through the native-ELF general nb_expr lowering.
- `self_host_native_fp_expr_smoke.py` (~1020 tok, large) — RI-B1 nb_expr float-scalar routing gate (zero MLIR/LLVM).
- `self_host_native_fp_field_smoke.py` (~1832 tok, huge) — RI-D2 S-D FLOAT struct-FIELD READ dtype through native-ELF general lowering (zero MLIR).
- `self_host_native_fp_let_smoke.py` (~1208 tok, large) — RI-B1 (#107 follow-up) FLOAT dtype propagation ACROSS a LET binding (zero MLIR/LLVM).
- `self_host_native_fp_param_smoke.py` (~1403 tok, large) — RI-D2 S-B: FLOAT fn-param dtype classification + SysV SSE-spill ABI (zero MLIR/LLVM).
- `self_host_native_fp_smoke.py` (~1142 tok, large) — RI-B1 native-ELF scalar-f64 gate (zero MLIR/LLVM).
- `self_host_native_gemm_i8_smoke.py` (~1051 tok, large) — RI-B2-S7 (#108) — native-ELF scalar int8 GEMM (matrix x matrix), byte-identity rung.
- `self_host_native_gemm_q16_smoke.py` (~1053 tok, large) — RI-B2-S7 (#108) — native-ELF scalar GEMM Q16.16 (matrix x matrix), byte-identity rung.
- `self_host_native_gemv_i16_smoke.py` (~1044 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV int16 (matrix x vector), byte-identity rung.
- `self_host_native_gemv_q16_smoke.py` (~1037 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV Q16.16 (matrix x vector), byte-identity rung.
- `self_host_native_genf32_smoke.py` (~1066 tok, large) — RI-B2-S8 STEP B (#108) — isolate the LCG f32 rounding BEFORE the dot.
- `self_host_native_intdot_i16_smoke.py` (~1018 tok, large) — RI-B2-S2 (#108) — native-ELF scalar int16 DOT-PRODUCT, FIRST byte-identity rung.
- `self_host_native_intdot_q16_smoke.py` (~1008 tok, large) — RI-B2-S4 (#108) — native-ELF scalar Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_intdot_smoke.py` (~1165 tok, large) — RI-B2-S1 (#108) scalar i64 DOT-PRODUCT reduction native-ELF (zero MLIR/LLVM).
- `self_host_native_matmul_f32_v_smoke.py` (~1611 tok, huge) — RI-B2-S9 (#108) — native-ELF scalar STRICT-FP f32 GEMV (matmul-f32-v).
- `self_host_native_narrow_add_i8_smoke.py` (~2260 tok, huge) — C2 — native-ELF NARROW-INT (i8) WRAP ARITHMETIC, zero MLIR/LLVM.
- `self_host_native_narrow_arith_batch_smoke.py` (~2554 tok, huge) — C2 — native-ELF NARROW-INT WRAP ARITHMETIC batch: {sub,mul}xi8 + {add,mul}xi16.
- `self_host_native_narrowint_smoke.py` (~1656 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 truncating
- `self_host_native_narrow_paramret_smoke.py` (~1949 tok, huge) — Byte-behavior smoke for narrow-int (i8/i16/i32) PARAM + RETURN auto-wrap in the
- `self_host_native_narrowwrap_smoke.py` (~1673 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 two's-complement
- `self_host_native_scalar_f32_smoke.py` (~1342 tok, large) — Phase C1-remainder f32 rung — native-ELF scalar SINGLE-precision chain.
- `self_host_native_scalar_f64_smoke.py` (~1201 tok, large) — RI-B2 f64 rung (#108) — native-ELF scalar STRICT-FP f64 CHAIN.
- `self_host_native_simd_dot_f32_smoke.py` (~1502 tok, huge) — RI-B2-S11 (#108) — native-ELF PACKED-f32 SIMD (SSE, 128-bit) STRICT-FP DOT-PRODUCT.
- `self_host_native_simd_dot_i16_smoke.py` (~1130 tok, large) — RI-B2-S12 (#108) — native-ELF PACKED-int16 SIMD DOT-PRODUCT, byte-identity rung.
- `self_host_native_simd_dot_q16_smoke.py` (~1486 tok, large) — RI-B2-S10 (#108) — native-ELF PACKED-SIMD Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_tensor_batchsum_smoke.py` (~2372 tok, huge) — C4-T6 — native-ELF 3-D BATCHED SUM (i64), zero MLIR/LLVM. The FIRST N-D
- `self_host_native_tensor_bcastadd_smoke.py` (~1700 tok, huge) — C4-T5 — native-ELF tensor ROW-VECTOR BROADCAST ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_colsum_smoke.py` (~2094 tok, huge) — C4-T5 — native-ELF tensor COLUMN REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_dot_smoke.py` (~1197 tok, large) — C4-T2 — native-ELF tensor DOT PRODUCT (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewadd_f64_smoke.py` (~1794 tok, huge) — C4-T4 — native-ELF float64 TENSOR element-wise-add + STRICT-SEQUENTIAL reduce.
- `self_host_native_tensor_ewadd_smoke.py` (~1118 tok, large) — C4-T1 — native-ELF tensor ELEMENT-WISE ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewmul_smoke.py` (~2047 tok, huge) — C4-T5 — native-ELF tensor ELEMENT-WISE MULTIPLY (i64), zero MLIR/LLVM.
