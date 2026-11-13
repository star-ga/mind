# NSF/1 frozen diagnostic catalogue

Author: STARGA Inc.

The independent source catalogue was prelisted before any candidate execution.
Its original expectations remain in `independent-CASES.json` and
`independent-SCOPE_CASES.json`; these are planned outcomes, not results.
The private operational capture is not part of this directory. The sources
are generic fixtures with no private consumer code.

`SUPPLEMENT.json` adds author boundary cases and source-extracted MIND observers.
An observer failure must return normally; a crash, timeout or failed build does
not count as mutation detection. Observer stubs test an explicit ownership/state
protocol and do not substitute for full source-to-image execution.
