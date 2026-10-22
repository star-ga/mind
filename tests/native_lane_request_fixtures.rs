// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The native-wave2 and native-wave3 checkpoint programs and the native
//! admission request programs, ported onto native-lane-v1.
//!
//! The two checkpoints added admission rules to the historical frozen profile,
//! which native-lane-v1 replaced (docs/compiler/NATIVE_LANE_PROFILE.md). Their
//! programs, and the request programs (memory intrinsics, integer `/ % << >>`,
//! the byte-compare loop, value-producing `if` inside `while`), live in
//! `tests/fixtures/native_lane/request_corpus.txt`, one record per program with
//! its expectation on the lane:
//!
//! * `exit=N` / `parity` (GREEN): `mindc build --backend native` builds it, the
//!   MLIR backend builds it, and both artifacts exit alike with identical
//!   stdout (`exit=N` also pins the value).
//! * `refused=TEXT`: the native build exits 3, writes no artifact and names
//!   TEXT. A wave refusal that the lane enforces under its own kind is pinned
//!   here; a refusal that a later lane lifts is edited by that lane.
//! * `defect=WHY`: the lane builds it today and the result differs from MLIR.
//!   `open_defects_are_refused_or_match_mlir` is ignored until a compiler
//!   change makes each one a named refusal or a parity case.
//!
//! The RED leg (`red=3`) runs the same records through a `mindc` whose
//! embedded seed predates native-lane-v1, named by `MIND_LANE_PRESEED_MINDC`:
//! every such record must be refused with exit 3, so the GREEN rows are
//! programs the lane newly builds, not programs any native compiler built.
//!
//! Default-profile builds use the committed seed, or `MINDC_NATIVE_ELF` when it
//! is set (a freshly derived stage1). Bounded-profile records always use the
//! committed bounded seed.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports",
    target_os = "linux",
    target_arch = "x86_64"
))]

mod common;
mod native_bridge_support;

use common::gate;
use native_bridge_support::{MINDC, Project, repo_path};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const TARGET: &str = "native_lane_request_fixtures";
const CORPUS: &str = include_str!("fixtures/native_lane/request_corpus.txt");
const PRESEED_VAR: &str = "MIND_LANE_PRESEED_MINDC";
const DEFAULT_SEED: &str = "examples/mindc_mind/testdata/selfhost_loop/stage1.elf";
const BOUNDED_SEED: &str = "examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf";
/// A miscompiled loop must turn a case red, not hang the suite.
const RUN_DEADLINE: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Expect {
    Exit(i32),
    Parity,
    Refused(String),
    Defect(String),
}

#[derive(Debug, Clone)]
struct Case {
    id: String,
    expect: Expect,
    red: bool,
    bounded: bool,
    origin: String,
    source: String,
}

fn parse_head(head: &str) -> Case {
    let fields: Vec<&str> = head.splitn(5, " | ").collect();
    assert_eq!(fields.len(), 5, "malformed corpus record head: {head}");
    let expect = match fields[1].split_once('=') {
        Some(("exit", n)) => Expect::Exit(n.parse().expect("exit code")),
        Some(("refused", text)) if !text.is_empty() => Expect::Refused(text.to_string()),
        Some(("defect", why)) if !why.is_empty() => Expect::Defect(why.to_string()),
        None if fields[1] == "parity" => Expect::Parity,
        _ => panic!("unknown expectation in record head: {head}"),
    };
    let red = match fields[2] {
        "red=3" => true,
        "red=-" => false,
        other => panic!("unknown red marker `{other}` in: {head}"),
    };
    let bounded = match fields[3] {
        "bounded" => true,
        "default" => false,
        other => panic!("unknown profile `{other}` in: {head}"),
    };
    Case {
        id: fields[0].to_string(),
        expect,
        red,
        bounded,
        origin: fields[4].to_string(),
        source: String::new(),
    }
}

fn parse(text: &str) -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        if let Some(head) = line.strip_prefix("=== ") {
            cases.push(parse_head(head));
        } else if let Some(case) = cases.last_mut() {
            case.source.push_str(line);
            case.source.push('\n');
        } else {
            assert!(
                line.is_empty() || line.starts_with('#'),
                "text before the first record must be a comment: {line}"
            );
        }
    }
    cases
}

fn corpus() -> Vec<Case> {
    parse(CORPUS)
}

fn in_group(case: &Case, prefix: &str) -> bool {
    case.origin.starts_with(prefix)
}

fn native_seed(case: &Case) -> PathBuf {
    if case.bounded {
        return repo_path(BOUNDED_SEED);
    }
    std::env::var_os("MINDC_NATIVE_ELF")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_path(DEFAULT_SEED))
}

/// Which `mindc` builds the native artifact and with which seed.
enum Compiler<'a> {
    /// This crate's `mindc` with the tree's seed and std (the lane).
    Lane,
    /// A pre-lane `mindc` with its own embedded seed and std blob.
    Preseed(&'a Path),
}

struct Built {
    code: i32,
    stderr: String,
    artifact: Option<PathBuf>,
}

fn project(case: &Case) -> Project {
    let p = Project::new(&case.id);
    p.write("src/main.mind", &case.source);
    p
}

fn native_build(p: &Project, case: &Case, compiler: &Compiler) -> Built {
    let out = p.root().join("native.out");
    let mut cmd = match compiler {
        Compiler::Lane => {
            let mut c = Command::new(MINDC);
            c.env("MINDC_STD_DIR", repo_path("std"))
                .env("MINDC_NATIVE_ELF", native_seed(case));
            c
        }
        Compiler::Preseed(mindc) => {
            let mut c = Command::new(mindc);
            c.env_remove("MINDC_STD_DIR").env_remove("MINDC_NATIVE_ELF");
            c
        }
    };
    cmd.current_dir(p.root())
        .args(["build", "src/main.mind", "--release", "--backend", "native"])
        .arg("--emit=binary")
        .env_remove("MINDC_SO");
    if case.bounded {
        cmd.args(["--native-profile", "bounded"]);
    }
    let res = cmd.arg("--out").arg(&out).output().expect("spawn mindc");
    Built {
        code: res.status.code().unwrap_or(-1),
        stderr: String::from_utf8_lossy(&res.stderr).into_owned(),
        artifact: out.exists().then_some(out),
    }
}

fn mlir_build(p: &Project) -> (Output, PathBuf) {
    let out = p.root().join("mlir.out");
    let res = Command::new(MINDC)
        .current_dir(p.root())
        .args(["build", "src/main.mind", "--out"])
        .arg(&out)
        .env("MINDC_STD_DIR", repo_path("std"))
        .env_remove("MINDC_NATIVE_ELF")
        .env_remove("MINDC_SO")
        .output()
        .expect("spawn mindc (mlir)");
    (res, out)
}

/// Run an artifact with no stdin and piped stdout; (exit code or 128 + signal, stdout).
fn run(artifact: &Path, id: &str) -> (i32, Vec<u8>) {
    use std::os::unix::process::ExitStatusExt;
    let mut child = Command::new(artifact)
        .current_dir(artifact.parent().expect("artifact dir"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("{id}: run {}: {e}", artifact.display()));
    let mut stdout = child.stdout.take().expect("piped stdout");
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .read_to_end(&mut bytes)
            .expect("read artifact stdout");
        bytes
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait artifact") {
            break status;
        }
        if start.elapsed() > RUN_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{id}: {} ran past {RUN_DEADLINE:?}", artifact.display());
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let code = status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0));
    (code, reader.join().expect("stdout reader"))
}

/// What one record does on both backends. `None` only on a routed toolchain skip.
struct Pair {
    native: (i32, Vec<u8>),
    mlir: (i32, Vec<u8>),
}

fn build_and_run_both(case: &Case) -> Option<Pair> {
    let p = project(case);
    let native = native_build(&p, case, &Compiler::Lane);
    let artifact = native.artifact.clone().unwrap_or_else(|| {
        panic!(
            "{}: native-lane-v1 must build this program (exit {}):\n{}",
            case.id, native.code, native.stderr
        )
    });
    assert_eq!(native.code, 0, "{}: {}", case.id, native.stderr);
    assert!(
        !native.stderr.contains("not natively compiled"),
        "{}: the native build wrote a launcher stub, not an executable\n{}",
        case.id,
        native.stderr
    );
    let (mlir, mlir_artifact) = mlir_build(&p);
    if !gate::compiled(TARGET, &mlir) {
        return None;
    }
    Some(Pair {
        native: run(&artifact, &case.id),
        mlir: run(&mlir_artifact, &case.id),
    })
}

fn assert_parity(case: &Case) -> Option<()> {
    let pair = build_and_run_both(case)?;
    assert_eq!(
        pair.native, pair.mlir,
        "{} ({}): native (exit, stdout) != MLIR (exit, stdout)",
        case.id, case.origin
    );
    if let Expect::Exit(want) = case.expect {
        assert_eq!(pair.native.0, want, "{}: exit code", case.id);
    }
    Some(())
}

fn assert_refused(case: &Case, built: &Built, text: &str) {
    assert!(
        !built.stderr.contains("panicked"),
        "{}: the native build panicked:\n{}",
        case.id,
        built.stderr
    );
    assert_eq!(
        built.code, 3,
        "{}: expected a refusal (exit 3):\n{}",
        case.id, built.stderr
    );
    assert!(
        built.artifact.is_none(),
        "{}: a refused build wrote an artifact",
        case.id
    );
    assert!(
        built.stderr.contains(text),
        "{}: expected the refusal to name `{text}`:\n{}",
        case.id,
        built.stderr
    );
}

/// Every GREEN record whose origin starts with `prefix`; returns how many ran.
fn green_group(prefix: &str) -> usize {
    let cases: Vec<Case> = corpus()
        .into_iter()
        .filter(|c| in_group(c, prefix))
        .filter(|c| matches!(c.expect, Expect::Exit(_) | Expect::Parity))
        .collect();
    assert!(!cases.is_empty(), "no GREEN record has origin `{prefix}`");
    for case in &cases {
        if assert_parity(case).is_none() {
            return 0;
        }
    }
    cases.len()
}

#[test]
fn wave2_programs_match_mlir_on_the_lane() {
    let ran = green_group("native-wave2");
    assert!(ran == 0 || ran == 18, "wave2 GREEN records ran: {ran}");
}

#[test]
fn wave3_programs_match_mlir_on_the_lane() {
    let ran = green_group("native-wave3");
    assert!(ran == 0 || ran == 12, "wave3 GREEN records ran: {ran}");
}

#[test]
fn request_programs_match_mlir_on_the_lane() {
    let a = green_group("native intrinsic admission request");
    let e = green_group("value-if inside while request");
    assert!(a == 0 || a == 6, "admission request GREEN records ran: {a}");
    assert!(e == 0 || e == 7, "value-if request GREEN records ran: {e}");
}

#[test]
fn the_scalar_control_matches_mlir_on_the_lane() {
    let ran = green_group("control");
    assert!(ran == 0 || ran == 1, "control GREEN records ran: {ran}");
}

#[test]
fn ported_refusals_are_named_and_write_nothing() {
    let mut ran = 0;
    for case in corpus() {
        let Expect::Refused(text) = &case.expect else {
            continue;
        };
        let p = project(&case);
        assert_refused(&case, &native_build(&p, &case, &Compiler::Lane), text);
        ran += 1;
    }
    assert_eq!(ran, 115, "refusal records checked");
}

#[test]
fn the_corpus_is_complete_and_well_formed() {
    let cases = corpus();
    let mut ids: Vec<&str> = cases.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), cases.len(), "duplicate record id");
    let count = |f: &dyn Fn(&Case) -> bool| cases.iter().filter(|c| f(c)).count();
    assert_eq!(count(&|c| matches!(c.expect, Expect::Exit(_))), 41);
    assert_eq!(count(&|c| c.expect == Expect::Parity), 3);
    assert_eq!(count(&|c| matches!(c.expect, Expect::Refused(_))), 115);
    assert_eq!(count(&|c| matches!(c.expect, Expect::Defect(_))), 13);
    assert_eq!(count(&|c| c.bounded), 3);
    assert_eq!(count(&|c| c.red), 138);
    for case in &cases {
        assert!(
            case.source.contains("fn main("),
            "{}: no entry point",
            case.id
        );
        assert!(!case.origin.trim().is_empty(), "{}: no origin", case.id);
    }
    // The parser must see a malformed head rather than skip it.
    let bad = std::panic::catch_unwind(|| parse("=== x | exit=1 | red=3 | default\n"));
    assert!(bad.is_err(), "a four-field head must be refused");
}

/// The RED leg. Run it with
/// `MIND_LANE_PRESEED_MINDC=<pre-lane mindc> cargo test ... -- --ignored red_leg`.
#[test]
#[ignore = "RED leg: needs a pre-lane mindc named by MIND_LANE_PRESEED_MINDC"]
fn red_leg_a_pre_lane_compiler_refuses_every_red_record() {
    let preseed = PathBuf::from(
        std::env::var_os(PRESEED_VAR)
            .expect("MIND_LANE_PRESEED_MINDC must name a mindc whose seed predates the lane"),
    );
    let cases = corpus();
    // Positive control: the pre-lane compiler builds and runs a scalar program,
    // so the refusals below are about the programs, not a broken compiler.
    let control = cases
        .iter()
        .find(|c| c.id == "ctl_return_7")
        .expect("control record");
    let p = project(control);
    let built = native_build(&p, control, &Compiler::Preseed(&preseed));
    let artifact = built.artifact.unwrap_or_else(|| {
        panic!(
            "the pre-lane compiler cannot build the control:\n{}",
            built.stderr
        )
    });
    assert_eq!(run(&artifact, &control.id).0, 7, "control exit");
    let mut green = 0;
    let mut refused = 0;
    for case in cases.iter().filter(|c| c.red) {
        let p = project(case);
        let built = native_build(&p, case, &Compiler::Preseed(&preseed));
        assert_refused(case, &built, "error[backend-native]");
        assert!(
            !built.stderr.contains("E6011"),
            "{}: the pre-lane compiler crashed rather than refused:\n{}",
            case.id,
            built.stderr
        );
        match case.expect {
            Expect::Exit(_) | Expect::Parity => green += 1,
            _ => refused += 1,
        }
    }
    println!(
        "{TARGET} RED leg: {green} GREEN records and {refused} other records refused (exit 3)"
    );
    assert_eq!((green, refused), (37, 101), "RED records checked");
}

/// Each open defect must end as a named refusal or as parity with MLIR.
#[test]
#[ignore = "open defects: native-lane-v1 builds these and the result differs from MLIR"]
fn open_defects_are_refused_or_match_mlir() {
    let mut open = Vec::new();
    for case in corpus() {
        let Expect::Defect(why) = &case.expect else {
            continue;
        };
        let p = project(&case);
        let built = native_build(&p, &case, &Compiler::Lane);
        if built.code == 3 && built.artifact.is_none() {
            continue;
        }
        let Some(pair) = build_and_run_both(&case) else {
            return;
        };
        if pair.native != pair.mlir {
            open.push(format!(
                "{}: native exit {} vs MLIR exit {} ({why})",
                case.id, pair.native.0, pair.mlir.0
            ));
        }
    }
    assert!(
        open.is_empty(),
        "silent native != MLIR:\n{}",
        open.join("\n")
    );
}
