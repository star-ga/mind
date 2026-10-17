#!/usr/bin/env python3
"""Runner for the self-host LOOP advancement harness controls.

The stable command remains python3 tests/self_host_loop_advance_test.py.
Cases live in focused modules so each file stays readable and below the
repository's 800-line harness ceiling.
"""
from __future__ import annotations

import sys

import self_host_loop_advance_contract_test as contract
import self_host_loop_advance_publication_test as publication
from _self_host_loop_advance_support import FAILURES


def main() -> int:
    cases = []
    for module in (contract, publication):
        cases.extend(
            value for name, value in vars(module).items()
            if name.startswith("case_") and callable(value)
        )
    for fn in sorted(cases, key=lambda function: function.__name__):
        fn()
    if FAILURES:
        print()
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print()
    print("RESULT: PASS — --advance publishes only on full agreement, "
          "--advance --hops 2 publishes stage2 only once stage2 == stage3 and "
          "the oracle agrees with it, --reseed records the broken seed chain, "
          "every ordinary refusal preserves the fixture, failed rollback is reported "
          "with recoverable copies, and ordinary verification is unweakened. "
          "(Harness controls: synthetic images, stub oracle — not "
          "compiler evidence.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
