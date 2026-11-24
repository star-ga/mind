# Bounded formal-mathematics examples

Seven small MIND programs. Each one computes exact integer or rational values
over a fixed, finite input range and checks them against known values inside
the program. Each program is a bounded, finite computation: a check over the
inputs it lists, not a proof of a statement over all integers. No program uses
floating point, a table of precomputed results or an import.

A program exits with status 0 only when all of its checks pass. It exits with
a nonzero status at the first check that fails.

| Program | What it computes | Values it checks |
| --- | --- | --- |
| [implication_truth_table.mind](implication_truth_table.mind) | The truth table of `p -> q` over all 4 assignments of two Boolean variables. It checks that modus ponens (`p` and `p -> q` give `q`) has no counterexample, and that the invalid rule "`p` gives `q`" has exactly one (`p` true, `q` false). | Exit 10 if modus ponens has a counterexample, exit 11 if the invalid rule does not have exactly one. It writes no output. |
| [prime_counting.mind](prime_counting.mind) | Primality by trial division, the prime count `pi(n)` for `n` up to 1000, and the next prime after `n`. | `pi(10)=4`, `pi(100)=25`, `pi(1000)=168`, next prime after 2 is 3, after 97 is 101 |
| [divisor_functions.mind](divisor_functions.mind) | The divisor sum `sigma(n)`, Euler's totient `phi(n)` for `n` in 1..1000, and the gcd. | `sigma(12)=28`, `sigma(28)=56`, `sigma(30)=72`, `phi(9)=6`, `phi(30)=8`, `phi(36)=12`, `gcd(54,24)=6` |
| [mobius_finite.mind](mobius_finite.mind) | The Mobius function `mu(n)` for `n` in 1..10000 and the Mertens sum `M(N)` for `N` in 0..100. | `mu(30)=-1`, `mu(210)=1`, `mu(4)=mu(36)=0`, `M(10)=-1` |
| [harmonic_exact.mind](harmonic_exact.mind) | The harmonic number `H(n)` for `n` in 0..10 as a reduced fraction of two `i64` values. | `H(5)=137/60`, `H(8)=761/280`, `H(10)=7381/2520` |
| [farey_finite.mind](farey_finite.mind) | The Farey sequence `F(n)` for `n` in 1..8, by the next-term recurrence. At each step it checks that neighbors `a/b < c/d` have `bc - ad = 1` and that `c/d` is in lowest terms. | `|F(1)|=2`, `|F(2)|=3`, `|F(3)|=5`, `|F(5)|=11`, `|F(8)|=23` |
| [goldbach_bounded.mind](goldbach_bounded.mind) | For each even number from 4 to 100, the smallest prime `p` such that `n - p` is also prime. | 49 even numbers each have a pair; the smallest `p` is 2 for 4, 5 for 28 and 3 for 100. This does not prove Goldbach's conjecture. |

Each program except `implication_truth_table.mind` writes two lines when all
of its checks pass, for example:

```text
EXAMPLE=PRIME_COUNTING
SCOPE=FINITE_EXAMPLE
```

## Run the examples

From the repository root, with a `mindc` binary:

```sh
MINDC=/path/to/mindc bash examples/formal_verification/run_all.sh
```

The script runs `mindc check` on each program, builds it with the native
backend into a temporary directory, runs it, and requires exit status 0, the
expected output and an empty stderr. It prints `EXAMPLES_PASSED=7` and exits 0
when all seven programs pass, and exits nonzero at the first failure.

To build and run one program by hand:

```sh
mindc check examples/formal_verification/prime_counting.mind
mindc build examples/formal_verification/prime_counting.mind --emit binary --out /tmp/prime_counting
/tmp/prime_counting
```

`--emit binary` is necessary from the repository root, because the `Mind.toml`
there selects a library output.
