//! Same-run paired A/B of two `mindc` binaries: the end-to-end gate's measurement.
//!
//! An absolute floor cannot decide an end-to-end regression: a build host drifts
//! 18-31% between sessions with load, three times the +10% the gate exists to
//! catch, and a hosted runner's number means nothing against a build host's. So
//! the gate times the base `mindc` (`MINDC_BASE`) and the candidate (`MINDC`) in
//! ONE run, interleaved per sample, and `tools/bench_gate.py --pairs` decides on
//! the per-pair ratios. Both arms share whatever noise the host has.
//!
//! Protocol, per row: 2 untimed spawns per arm, then N pairs (warm 40, cold 20,
//! native 60). Pair i runs base then candidate when i is even and candidate then
//! base when i is odd (ABBA: cancels position and linear drift). One sample is
//! the wall clock around one `Command::status()`; a cold sample's fresh
//! `XDG_CACHE_HOME` is created before and deleted after the timed region. Raw
//! pairs go to `bench-e2e-pairs.tsv` (or `MIND_E2E_PAIRS`); the candidate arm is
//! also printed as bencher lines (median, +/- sd) so the record-only floor and
//! ordering steps reuse this run. No statistic is computed here: the decision
//! rule lives in `tools/bench_gate.py` and nowhere else.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::{
    NATIVE_SHAPES, T2_SHAPES, Tools, build_native_cmd, emit_shared_cmd, manifest_dir, read, run_ok,
    stage_references, tmp,
};

/// Minimum pairs per row; `tools/bench_gate.py --pairs` rejects fewer (exit 4).
const WARM_PAIRS: usize = 40;
const COLD_PAIRS: usize = 20;
const NATIVE_PAIRS: usize = 60;
/// Candidate-only samples per same-session reference row (recorded, not gated).
const REFERENCE_SAMPLES: usize = 20;
/// Untimed spawns per arm before a row's first timed pair.
const UNTIMED: usize = 2;

/// Index 0 is the base arm, index 1 the candidate, everywhere in this module.
const BASE: usize = 0;
const CAND: usize = 1;

struct Run {
    mindc: [PathBuf; 2],
    header: String,
    rows: String,
}

impl Run {
    /// Time `n` ABBA pairs of `spawn(arm)` for `row` and record them. `spawn`
    /// returns the command plus anything that must outlive it (a cold cache).
    fn pairs(&mut self, row: &str, n: usize, spawn: impl Fn(usize) -> (Command, Option<TempDir>)) {
        for _ in 0..UNTIMED {
            for arm in [BASE, CAND] {
                let (mut cmd, keep) = spawn(arm);
                assert!(run_ok(&mut cmd), "{row}: untimed spawn failed");
                drop(keep);
            }
        }
        let mut ns = [Vec::with_capacity(n), Vec::with_capacity(n)];
        for i in 0..n {
            let (order, label) = if i % 2 == 0 {
                ([BASE, CAND], "BC")
            } else {
                ([CAND, BASE], "CB")
            };
            for arm in order {
                let (mut cmd, keep) = spawn(arm);
                ns[arm].push(timed(&mut cmd, row));
                drop(keep); // the cold cache is deleted outside the timed region
            }
            let _ = writeln!(
                self.rows,
                "{row}\t{i}\t{label}\t{}\t{}",
                ns[BASE][i], ns[CAND][i]
            );
        }
        bencher_line(row, &ns[CAND]);
        println!(
            "e2e_pairs {row}: n={n} median base {:.1} ms, cand {:.1} ms",
            median(&ns[BASE]) / 1e6,
            median(&ns[CAND]) / 1e6
        );
    }

    /// Record (never block on) whether base and candidate emit the same bytes: a
    /// codegen change may legitimately alter them.
    fn identity(&mut self, what: &str, same: bool) {
        println!(
            "e2e_identity {what} cand==base {}",
            if same { "yes" } else { "no" }
        );
        let _ = writeln!(self.header, "# identical {what}={}", u8::from(same));
    }
}

/// Wall clock around exactly one `Command::status()`; a failed compile inside the
/// timed loop is not a timing result.
fn timed(cmd: &mut Command, row: &str) -> u128 {
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let t = Instant::now();
    let status = cmd.status();
    let ns = t.elapsed().as_nanos();
    assert!(
        status.map(|s| s.success()).unwrap_or(false),
        "{row}: mindc failed inside the timed loop"
    );
    ns
}

fn median(v: &[u128]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    let m = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[m - 1] as f64 + s[m] as f64) / 2.0
    } else {
        s[m] as f64
    }
}

/// The criterion bencher shape `tools/bench_gate.py` already parses.
fn bencher_line(row: &str, ns: &[u128]) {
    let n = ns.len() as f64;
    let mean = ns.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = ns.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    println!(
        "test {row} ... bench: {:.0} ns/iter (+/- {:.0})",
        median(ns),
        var.sqrt()
    );
}

fn sha256_hex(path: &Path) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("compiler::e2e paired: cannot read {}: {e}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(crate) fn run(tools: &Tools, base: &Path) {
    assert!(
        base.is_file(),
        "compiler::e2e paired: MINDC_BASE={} is not a file. A paired run without its base \
         is a wiring defect, never a criterion fallback.",
        base.display()
    );
    assert!(
        tools.t2 && tools.native,
        "compiler::e2e paired: mlir-opt/mlir-translate/clang or the native corpus is absent; \
         every gated row must run (3 warm, 3 cold, 4 native)."
    );
    let (cand_sha, base_sha) = (sha256_hex(&tools.mindc), sha256_hex(base));
    let null_control = u8::from(cand_sha == base_sha);
    println!("e2e_pairs cand={} sha256={cand_sha}", tools.mindc.display());
    println!("e2e_pairs base={} sha256={base_sha}", base.display());
    println!("null_control={null_control}");
    let mut run = Run {
        mindc: [base.to_path_buf(), tools.mindc.clone()],
        header: format!(
            "# mind e2e paired A/B v1\n# cand_sha256={cand_sha}\n# base_sha256={base_sha}\n\
             # null_control={null_control}\n"
        ),
        rows: String::from("row\ti\torder\tt_base_ns\tt_cand_ns\n"),
    };

    // Interleaved by shape, as the criterion rows are: every MIND row is followed
    // by its clang and rustc equivalents under the same host load.
    for (shape, source) in T2_SHAPES {
        t2_shape(&mut run, shape, source);
        references(tools, shape);
    }
    for shape in NATIVE_SHAPES {
        native_shape(&mut run, shape);
        references(tools, shape);
    }

    let path = std::env::var_os("MIND_E2E_PAIRS")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("bench-e2e-pairs.tsv"));
    std::fs::write(&path, format!("{}{}", run.header, run.rows))
        .unwrap_or_else(|e| panic!("compiler::e2e paired: cannot write {}: {e}", path.display()));
    println!("e2e_pairs written to {}", path.display());
}

/// Untimed pre-check for one arm: a cold build, a cache-filling and a cache-hit
/// warm build. Returns (cold, fill, hit) bytes, or `None` when the arm cannot emit.
fn t2_precheck(mindc: &Path, src: &Path, dir: &Path, warm_xdg: &Path) -> Option<[Vec<u8>; 3]> {
    let cold_xdg = tmp();
    let (cold_out, warm_out) = (dir.join("cold.so"), dir.join("warm.so"));
    if !run_ok(&mut emit_shared_cmd(
        mindc,
        src,
        &cold_out,
        cold_xdg.path(),
        true,
    )) || !run_ok(&mut emit_shared_cmd(mindc, src, &warm_out, warm_xdg, false))
    {
        return None;
    }
    let fill = read(&warm_out);
    if !run_ok(&mut emit_shared_cmd(mindc, src, &warm_out, warm_xdg, false)) {
        return None;
    }
    Some([read(&cold_out), fill, read(&warm_out)])
}

fn t2_shape(run: &mut Run, shape: &str, source: &str) {
    let work = tmp();
    let src = work.path().join(format!("{shape}.mind"));
    std::fs::write(&src, source).expect("compiler::e2e: cannot stage the T2 source");
    // One primed cache and one output per arm: the compilers never share either.
    let dirs = [work.path().join("base"), work.path().join("cand")];
    let xdg = [dirs[BASE].join("xdg-warm"), dirs[CAND].join("xdg-warm")];
    let mut cold = [Vec::new(), Vec::new()];
    for arm in [BASE, CAND] {
        std::fs::create_dir_all(&xdg[arm]).expect("compiler::e2e: cannot create the warm cache");
        let Some([c, fill, hit]) = t2_precheck(&run.mindc[arm], &src, &dirs[arm], &xdg[arm]) else {
            panic!(
                "compiler::e2e paired: the {} mindc did not emit the {shape} shared object",
                ["base", "cand"][arm]
            );
        };
        let same = !c.is_empty() && c == fill && c == hit;
        if arm == CAND {
            assert!(
                same,
                "compiler::e2e_compile_shared: {shape}: cold, cache-filling and cache-hit \
                 builds emitted different bytes. A cache must never change output."
            );
        } else if !same {
            println!("e2e_identity e2e_compile_shared/{shape} base cold!=warm (recorded)");
        }
        cold[arm] = c;
    }
    run.identity(
        &format!("e2e_compile_shared/{shape}"),
        cold[BASE] == cold[CAND],
    );

    let mindc = run.mindc.clone();
    let out = [dirs[BASE].join("timed.so"), dirs[CAND].join("timed.so")];
    run.pairs(
        &format!("e2e_compile_shared/warm/{shape}"),
        WARM_PAIRS,
        |arm| {
            (
                emit_shared_cmd(&mindc[arm], &src, &out[arm], &xdg[arm], false),
                None,
            )
        },
    );
    run.pairs(
        &format!("e2e_compile_shared/cold/{shape}"),
        COLD_PAIRS,
        |arm| {
            let fresh = tmp(); // created before the timed region
            let cmd = emit_shared_cmd(&mindc[arm], &src, &out[arm], fresh.path(), true);
            (cmd, Some(fresh))
        },
    );
}

fn native_shape(run: &mut Run, shape: &str) {
    let corpus = manifest_dir().join("examples/mindc_mind/testdata/backend_native_bridge");
    let src = corpus.join(format!("{shape}.mind"));
    let oracle = read(&corpus.join(format!("{shape}.elf")));
    let work = tmp();
    let out = [work.path().join("base.elf"), work.path().join("cand.elf")];
    let mut bytes = [Vec::new(), Vec::new()];
    for arm in [BASE, CAND] {
        assert!(
            run_ok(&mut build_native_cmd(&run.mindc[arm], &src, &out[arm])),
            "compiler::e2e paired: the {} mindc did not build native {shape}",
            ["base", "cand"][arm]
        );
        bytes[arm] = read(&out[arm]);
    }
    assert!(
        !oracle.is_empty() && bytes[CAND] == oracle,
        "compiler::e2e_native: {shape} is not byte-identical to its committed oracle"
    );
    if bytes[BASE] != oracle {
        println!("e2e_identity e2e_native/{shape} base!=oracle (recorded)");
    }
    run.identity(&format!("e2e_native/{shape}"), bytes[BASE] == bytes[CAND]);

    let mindc = run.mindc.clone();
    run.pairs(&format!("e2e_native/{shape}"), NATIVE_PAIRS, |arm| {
        (build_native_cmd(&mindc[arm], &src, &out[arm]), None)
    });
}

/// Same-session clang/rustc rows for `shape`, candidate-session only: recorded as
/// bencher lines for the ordering step, never part of the paired decision.
fn references(tools: &Tools, shape: &str) {
    let work = tmp();
    for r in stage_references(tools, shape, work.path()) {
        let out = work.path().join(format!("{shape}_{}.so", r.row));
        let row = format!("e2e_reference/{}/{shape}", r.row);
        for _ in 0..UNTIMED {
            assert!(
                run_ok(&mut r.command(&out)),
                "{row}: untimed compile failed"
            );
        }
        let ns: Vec<u128> = (0..REFERENCE_SAMPLES)
            .map(|_| timed(&mut r.command(&out), &row))
            .collect();
        bencher_line(&row, &ns);
    }
}
