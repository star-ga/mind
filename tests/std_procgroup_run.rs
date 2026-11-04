// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/procgroup.mind`: start many child processes, wait for all of them.
//!
//! Section A (`std-surface`): the module parses, lowers, exports its public
//! surface and resolves through the bundled stdlib. With `cross-module-imports`,
//! the flat `mindc <file>` path refuses an artifact for a program that calls a
//! world-reading procgroup function unless `--allow-nondeterministic` is given,
//! and attests such a program `nondeterministic`.
//!
//! Section B (`mlir-build`, Linux): `tests/std_procgroup_run.mind` is built
//! with `mindc build` (the MLIR path) and run once. It is its own child: the
//! parent spawns the same executable through `std.procgroup` and prints
//! `k <key>` / `v <value>` pairs; each test below checks one case of it:
//! 25 children with distinct results, failing children reported, a straggler
//! killed at the deadline, pipes drained while children run, arguments and a
//! missing program, output truncation, the task index, refused inputs, a
//! spawn whose pipe cannot be made, stdin and descriptor isolation, and
//! children the kernel reaped. A second run of the same executable makes
//! fork() fail (RLIMIT_NPROC 0) and checks that spawn is counted.

#![cfg(feature = "std-surface")]

mod common;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const PROCGROUP_SRC: &str = include_str!("../std/procgroup.mind");

const PUBLIC_FNS: &[&str] = &[
    "procgroup_new",
    "procgroup_spawn",
    "procgroup_wait_all",
    "procgroup_free",
    "procgroup_len",
    "procgroup_failures",
    "procgroup_state",
    "procgroup_pid",
    "procgroup_exit_code",
    "procgroup_signal",
    "procgroup_stdout_addr",
    "procgroup_stdout_len",
    "procgroup_truncated",
    "procgroup_task_index",
    "procgroup_self_exe",
    "procgroup_cstr",
    "procgroup_decimal",
    "procgroup_state_ok",
    "procgroup_state_exit_nonzero",
    "procgroup_state_signaled",
    "procgroup_state_timed_out",
    "procgroup_state_spawn_failed",
    "procgroup_state_running",
    "procgroup_state_lost",
    "procgroup_max_children",
    "procgroup_max_output",
];

#[test]
fn procgroup_mind_parses_and_lowers_every_public_fn() {
    let module = parser::parse(PROCGROUP_SRC).expect("std/procgroup.mind must parse");
    let ir = lower_to_ir(&module).expect("std/procgroup.mind must lower");
    for want in PUBLIC_FNS {
        assert!(
            ir.instrs
                .iter()
                .any(|i| matches!(i, Instr::FnDef { name, .. } if name == want)),
            "std/procgroup.mind: missing FnDef `{want}`"
        );
    }
}

#[cfg(feature = "cross-module-imports")]
mod bundle {
    use super::*;
    use libmind::project::module_table::{build_module_table, collect_module_exports};
    use libmind::project::stdlib::parsed_stdlib_modules;

    #[test]
    fn procgroup_exports_its_public_fns() {
        let module = parser::parse(PROCGROUP_SRC).expect("parse");
        let ex = collect_module_exports("std.procgroup", &module);
        for want in PUBLIC_FNS {
            assert!(
                ex.exported.iter().any(|s| s == want),
                "std.procgroup must export `{want}`; got {:?}",
                ex.exported
            );
        }
    }

    #[test]
    fn bundled_stdlib_resolves_std_procgroup() {
        let mods = parsed_stdlib_modules();
        let refs: Vec<(String, &libmind::ast::Module)> =
            mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        assert!(
            table.resolves(&["std".into(), "procgroup".into()], "procgroup_wait_all"),
            "the bundled stdlib must expose std.procgroup::procgroup_wait_all"
        );
    }

    /// docs/std-procgroup.md's example, verbatim, so the docs show what is tested.
    fn doc_example() -> &'static str {
        const DOC: &str = include_str!("../docs/std-procgroup.md");
        let start = DOC
            .find("```mind\n")
            .expect("the doc has a ```mind example")
            + 8;
        let len = DOC[start..]
            .find("\n```")
            .expect("the example block is closed");
        &DOC[start..=start + len]
    }

    /// `mindc <tag>.mind <args>` from a scratch dir; (exit ok, stderr, stdout).
    fn flat(tag: &str, src: &str, args: &[&str]) -> (bool, String, String) {
        let dir = crate::common::scratch_dir("std_procgroup_flat");
        let path = dir.join(format!("{tag}.mind"));
        std::fs::write(&path, src).expect("write the program");
        let out = std::process::Command::new(crate::common::mindc_bin())
            .arg(&path)
            .args(args)
            .current_dir(&dir)
            .output()
            .expect("run mindc");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    }

    fn one_call(body: &str) -> String {
        format!("import std.procgroup;\n\nfn main() -> i64 {{\n{body}}}\n")
    }

    /// A program reaching any of the three world-reading functions (fork and
    /// the environment, the clock deadline, the environment) is refused an
    /// artifact without the flag, and the refusal names that function.
    #[test]
    fn a_world_reading_procgroup_call_needs_allow_nondeterministic() {
        let cases = [
            (
                "spawn",
                one_call(
                    "    let g: i64 = procgroup_new(1);\n    return procgroup_spawn(g, procgroup_self_exe(), 0, 0);\n",
                ),
                "procgroup_spawn",
            ),
            (
                "wait_all",
                one_call(
                    "    let g: i64 = procgroup_new(1);\n    return procgroup_wait_all(g, 1000);\n",
                ),
                "procgroup_wait_all",
            ),
            (
                "task_index",
                one_call("    return procgroup_task_index();\n"),
                "procgroup_task_index",
            ),
            ("doc", doc_example().to_string(), "procgroup_task_index"),
        ];
        for (tag, src, offender) in &cases {
            let so = format!("{tag}.so");
            let (ok, stderr, _) = flat(tag, src, &["--emit-shared", &so]);
            assert!(
                !ok,
                "{tag}: --emit-shared succeeded without the flag\n{stderr}"
            );
            assert!(
                stderr.contains(&format!("error[determinism]: `{offender}()`")),
                "{tag}: the refusal must name `{offender}()`\n{stderr}"
            );
            let (_, stderr, _) = flat(
                tag,
                src,
                &["--emit-shared", &so, "--allow-nondeterministic"],
            );
            assert!(
                !stderr.contains("error[determinism]"),
                "{tag}: the flag must authorise the build\n{stderr}"
            );
        }
    }

    /// With the flag the doc example is attested `nondeterministic`; a program
    /// that only creates, counts and frees a group is not tainted at all.
    #[test]
    fn evidence_attests_procgroup_programs_nondeterministic() {
        let pure = one_call(
            "    let g: i64 = procgroup_new(1);\n    let n: i64 = procgroup_len(g);\n    let _ = procgroup_free(g);\n    return n;\n",
        );
        for (tag, src, args, want) in [
            (
                "doc_ev",
                doc_example().to_string(),
                &["--allow-nondeterministic"][..],
                "nondeterministic",
            ),
            ("pure_ev", pure, &[][..], "deterministic"),
        ] {
            let ev = format!("{tag}.evidence");
            let mut argv = vec!["--emit-evidence", ev.as_str()];
            argv.extend_from_slice(args);
            let (ok, stderr, _) = flat(tag, &src, &argv);
            assert!(ok, "{tag}: --emit-evidence failed\n{stderr}");
            let dir = crate::common::scratch_dir("std_procgroup_flat");
            let out = std::process::Command::new(crate::common::mindc_bin())
                .args([
                    "verify",
                    dir.join(&ev).to_str().expect("utf-8 path"),
                    "--json",
                ])
                .output()
                .expect("run mindc verify");
            let report = String::from_utf8_lossy(&out.stdout);
            assert!(
                report.contains(&format!("\"determinism\":\"{want}\"")),
                "{tag}: evidence must attest {want}\n{report}"
            );
        }
    }

    /// The substrate list is the link side, which only `mlir-build` compiles.
    #[cfg(feature = "mlir-build")]
    #[test]
    fn std_procgroup_is_a_self_contained_substrate_module() {
        use libmind::project::substrate_link::{SUBSTRATE_MODULES, substrate_closure};
        assert!(
            SUBSTRATE_MODULES.contains(&"std.procgroup"),
            "std.procgroup must be linked into executables that import it"
        );
        let closure = substrate_closure(["import std.procgroup;\n"]);
        assert_eq!(
            closure.iter().copied().collect::<Vec<_>>(),
            vec!["std.procgroup"],
            "std.procgroup is self-contained: importing it links only itself"
        );
    }
}

#[cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "cross-module-imports"
))]
mod run {
    use std::collections::BTreeMap;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::{Command, Stdio};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    const TARGET: &str = "std_procgroup_run";
    const PROGRAM: &str = include_str!("std_procgroup_run.mind");
    /// The whole program, every case included, finishes in a few seconds; a
    /// child the deadline failed to kill sleeps forever, so this bound is
    /// what turns that defect into a failure instead of a hung test.
    const RUN_LIMIT: Duration = Duration::from_secs(120);

    struct Run {
        kv: BTreeMap<i64, i64>,
        stdout: String,
        elapsed: Duration,
    }

    impl Run {
        fn get(&self, key: i64) -> i64 {
            *self.kv.get(&key).unwrap_or_else(|| {
                panic!(
                    "key {key} missing from the program output:\n{}",
                    self.stdout
                )
            })
        }
    }

    fn build(dir: &Path) -> Option<std::path::PathBuf> {
        std::fs::write(
            dir.join("Mind.toml"),
            "[package]\nname = \"pg\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"pg\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
        )
        .expect("write Mind.toml");
        std::fs::write(dir.join("main.mind"), PROGRAM).expect("write main.mind");
        let out = Command::new(crate::common::mindc_bin())
            .arg("build")
            .current_dir(dir)
            .output()
            .expect("run mindc build");
        if !crate::common::gate::compiled(TARGET, &out) {
            return None;
        }
        Some(dir.join("target").join("debug").join("pg"))
    }

    /// Run the program once. An `Err` is stored and reported by every test, so
    /// a hung or failed run costs one `RUN_LIMIT`, not one per test.
    fn execute(exe: &Path) -> Result<Run, String> {
        let mut cmd = Command::new(exe);
        cmd.env_remove("PG_CASE");
        execute_cmd(cmd, exe)
    }

    fn execute_cmd(mut cmd: Command, exe: &Path) -> Result<Run, String> {
        let dir = exe.parent().expect("artifact dir");
        let out_path = dir.join("run.stdout");
        let err_path = dir.join("run.stderr");
        let start = Instant::now();
        // stdin stays an open pipe until the program exits: a child that
        // inherited it instead of /dev/null would block reading it. The own
        // process group lets a timeout kill the children with the parent.
        let mut child = cmd
            .env_remove("MIND_PROCGROUP_TASK")
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(std::fs::File::create(&out_path).expect("stdout file"))
            .stderr(std::fs::File::create(&err_path).expect("stderr file"))
            .spawn()
            .expect("run the built executable");
        let status = loop {
            if let Some(status) = child.try_wait().expect("poll the executable") {
                break status;
            }
            if start.elapsed() > RUN_LIMIT {
                let _ = Command::new("kill")
                    .args(["-KILL", "--", &format!("-{}", child.id())])
                    .status();
                let _ = child.wait();
                return Err(format!("the program did not finish within {RUN_LIMIT:?}"));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let elapsed = start.elapsed();
        let stdout = std::fs::read_to_string(&out_path).expect("read stdout");
        let stderr = std::fs::read_to_string(&err_path).expect("read stderr");
        if !status.success() || !stdout.lines().any(|l| l == "done 1") {
            return Err(format!(
                "the parent must exit 0 after its last line: {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ));
        }
        Ok(Run {
            kv: parse_pairs(&stdout),
            stdout,
            elapsed,
        })
    }

    /// `k <key>` then `v <value>`; any other line (a child's bytes the parent
    /// echoed, for instance) is ignored.
    fn parse_pairs(stdout: &str) -> BTreeMap<i64, i64> {
        let mut kv = BTreeMap::new();
        let mut key: Option<i64> = None;
        for line in stdout.lines() {
            if let Some(k) = line.strip_prefix("k ") {
                key = Some(k.trim().parse().expect("numeric key"));
            } else if let Some(v) = line.strip_prefix("v ") {
                let k = key
                    .take()
                    .unwrap_or_else(|| panic!("value without a key: {line}"));
                let v: i64 = v.trim().parse().expect("numeric value");
                assert!(kv.insert(k, v).is_none(), "key {k} printed twice");
            }
        }
        kv
    }

    fn built() -> Option<&'static std::path::PathBuf> {
        static EXE: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
        EXE.get_or_init(|| build(&crate::common::scratch_dir(TARGET)))
            .as_ref()
    }

    fn shared() -> Option<&'static Run> {
        static RUN: OnceLock<Option<Result<Run, String>>> = OnceLock::new();
        let run = RUN.get_or_init(|| built().map(|exe| execute(exe)));
        match run {
            None => None,
            Some(Ok(run)) => Some(run),
            Some(Err(why)) => panic!("{why}"),
        }
    }

    /// The effective uid from /proc/self/status (field 2 of `Uid:`).
    fn effective_uid() -> Option<u32> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("Uid:"))?;
        line.split_whitespace().nth(2)?.parse().ok()
    }

    /// The fork-failure case, alone (`PG_CASE=9`). The kernel never applies
    /// RLIMIT_NPROC to root, so as root the run drops to uid/gid 65534, from a
    /// copy of the executable in a directory that uid can read.
    fn fork_failure_run() -> Option<Run> {
        use std::os::unix::fs::PermissionsExt;
        let exe = built()?;
        let dir = crate::common::scratch_dir("std_procgroup_run_nproc");
        let copy = dir.join("pg");
        std::fs::copy(exe, &copy).expect("copy the executable");
        for p in [&dir, &copy] {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755))
                .expect("make the copy readable");
        }
        let mut cmd = Command::new(&copy);
        cmd.env("PG_CASE", "9");
        if effective_uid() == Some(0) {
            cmd.uid(65534).gid(65534);
        }
        Some(execute_cmd(cmd, &copy).unwrap_or_else(|why| panic!("{why}")))
    }

    const OK: i64 = 0;
    const EXIT_NONZERO: i64 = 1;
    const SIGNALED: i64 = 2;
    const TIMED_OUT: i64 = 3;
    const SPAWN_FAILED: i64 = 4;
    const LOST: i64 = 6;

    #[test]
    fn twenty_five_children_each_return_a_distinct_result() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(1901), 25, "25 slots taken");
        assert_eq!(run.get(1902), 0, "no child failed");
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..25 {
            assert_eq!(run.get(1000 + i), OK, "slot {i} state");
            assert_eq!(run.get(1200 + i), 0, "slot {i} exit code");
            assert_eq!(run.get(1100 + i), i * i + 1000, "slot {i} result");
            seen.insert(run.get(1100 + i));
        }
        assert_eq!(seen.len(), 25, "every child wrote its own result");
        assert_eq!(run.get(1903), 1, "a reaped group frees");
    }

    #[test]
    fn a_failing_child_is_reported_and_never_dropped() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(2900), 2, "wait_all counts both failures");
        assert_eq!(run.get(2901), 2, "procgroup_failures agrees");
        assert_eq!(run.get(2003), EXIT_NONZERO, "slot 3 exited non-zero");
        assert_eq!(run.get(2103), 3, "slot 3 exit code");
        assert_eq!(run.get(2303), 33, "slot 3 output is kept");
        assert_eq!(run.get(2005), SIGNALED, "slot 5 aborted");
        assert_eq!(run.get(2205), 6, "slot 5 died of SIGABRT");
        assert_eq!(run.get(2105), -1, "a signalled child has no exit code");
        for i in [0, 1, 2, 4, 6, 7] {
            assert_eq!(run.get(2000 + i), OK, "slot {i} state");
            assert_eq!(run.get(2300 + i), i, "slot {i} output");
        }
    }

    #[test]
    fn the_deadline_kills_a_straggler_and_keeps_the_rest() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(
            run.get(3903),
            0,
            "a group with running children is not freed"
        );
        assert_eq!(run.get(3900), 1, "only the straggler failed");
        assert_eq!(run.get(3002), TIMED_OUT, "slot 2 timed out");
        assert_eq!(run.get(3102), 9, "slot 2 was killed with SIGKILL");
        assert_eq!(
            run.get(3202),
            72,
            "output written before the deadline is kept"
        );
        for i in [0, 1, 3] {
            assert_eq!(run.get(3000 + i), OK, "slot {i} state");
            assert_eq!(run.get(3200 + i), 70 + i, "slot {i} output");
        }
        assert_eq!(run.get(3901), 1, "a second wait returns the same count");
        assert_eq!(
            run.get(3902),
            1,
            "nothing is left running after the deadline"
        );
        assert!(
            run.elapsed < Duration::from_secs(60),
            "the program took {:?}; the 1.5 s deadline did not bound the wait",
            run.elapsed
        );
    }

    #[test]
    fn pipes_are_drained_while_the_children_run() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(4900), 0, "no child blocked on a full pipe");
        for i in 0..4 {
            assert_eq!(run.get(4000 + i), OK, "slot {i} state");
            assert_eq!(run.get(4100 + i), 300_000 + i, "slot {i} byte count");
            assert_eq!(run.get(4200 + i), 0, "slot {i} not truncated");
        }
    }

    #[test]
    fn arguments_reach_the_child_and_a_missing_program_is_reported() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(5000), 0, "echo takes slot 0");
        assert_eq!(run.get(5001), 1, "the missing program takes slot 1");
        assert_eq!(run.get(5002), -1, "a full group refuses the spawn");
        assert_eq!(run.get(5009), 2, "the refused spawn took no slot");
        assert_eq!(run.get(5003), 1, "the missing program is the one failure");
        assert_eq!(run.get(5004), OK, "echo succeeded");
        assert_eq!(run.get(5005), 17, "echo wrote `procgroup-arg 42\\n`");
        assert_eq!(run.get(5006), 42, "the decimal argument arrived");
        assert_eq!(run.get(5007), EXIT_NONZERO, "the exec failure is a failure");
        assert_eq!(run.get(5008), 127, "the exec failure exits 127");
        assert!(
            run.stdout.lines().any(|l| l == "procgroup-arg 42"),
            "the captured bytes are exactly what echo wrote:\n{}",
            run.stdout
        );
    }

    #[test]
    fn output_past_the_cap_is_dropped_and_flagged() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(6900), 0, "the writer still exits 0");
        assert_eq!(run.get(6000), OK);
        assert_eq!(run.get(6001), 16_777_216, "exactly the cap is kept");
        assert_eq!(run.get(6002), 1, "the drop is flagged");
    }

    #[test]
    fn each_child_sees_its_own_index_and_the_parent_keeps_its_own() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(1900), -1, "a parent started directly reads -1");
        assert_eq!(
            run.get(7900),
            41,
            "the parent's own index survives spawning"
        );
        assert_eq!(run.get(7901), 0);
        assert_eq!(run.get(7000), 0, "child 0 reads index 0");
        assert_eq!(run.get(7001), 1, "child 1 reads index 1");
        assert_eq!(
            run.get(7903),
            -1,
            "a value that is not a number is no index"
        );
        assert_eq!(run.get(7904), -1, "an empty value is no index");
        assert_eq!(run.get(7905), -1, "ten digits are no index");
        assert_eq!(run.get(7906), 123_456_789, "nine digits are read");
        assert_eq!(run.get(7902), -1, "unset again");
    }

    #[test]
    fn inputs_outside_the_contract_are_refused() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(8000), 0, "capacity 0");
        assert_eq!(run.get(8001), 0, "capacity past the maximum");
        assert_eq!(run.get(8002), -1, "spawn on a 0 handle");
        assert_eq!(run.get(8003), -1, "wait on a 0 handle");
        assert_eq!(run.get(8004), -1, "a negative argument count");
        assert_eq!(run.get(8005), 0, "the refused spawn took no slot");
        assert_eq!(run.get(8006), 0, "an empty group waits for nothing");
        assert_eq!(run.get(8007), -1, "no slot 0 in an empty group");
        assert_eq!(run.get(8008), 0, "no address for a missing slot");
        assert_eq!(run.get(8009), 0, "a negative length");
        assert_eq!(run.get(8010), 0, "a negative decimal");
    }

    #[test]
    fn reads_outside_a_group_return_the_missing_value() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(8011), -1, "len of a 0 handle");
        assert_eq!(run.get(8012), -1, "failures of a 0 handle");
        assert_eq!(run.get(8013), 0, "free of a 0 handle");
        assert_eq!(run.get(8014), -1, "state of a 0 handle");
        assert_eq!(run.get(8015), 0, "stdout address of a 0 handle");
        assert_eq!(run.get(8016), 0, "the two-slot group finished");
        assert_eq!(run.get(8024), OK, "slot 1 exists");
        // Slot -1 would read the bytes before the slot array.
        assert_eq!(run.get(8017), -1, "state of slot -1");
        assert_eq!(run.get(8018), -1, "pid of slot -1");
        assert_eq!(run.get(8019), -1, "exit code of slot -1");
        assert_eq!(run.get(8020), -1, "signal of slot -1");
        assert_eq!(run.get(8021), 0, "stdout address of slot -1");
        assert_eq!(run.get(8022), -1, "stdout length of slot -1");
        assert_eq!(run.get(8023), -1, "truncation flag of slot -1");
    }

    #[test]
    fn a_spawn_that_cannot_make_its_pipe_is_counted() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(9004), 0, "the descriptor limit was lowered");
        assert_eq!(run.get(9000), 0, "the failed spawn still takes slot 0");
        assert_eq!(run.get(9001), SPAWN_FAILED, "and records why");
        assert_eq!(run.get(9002), 1, "wait_all counts it");
        assert_eq!(run.get(9003), -1, "no child was started");
    }

    /// A failed fork must not be stored as a running child with pid -1:
    /// waitpid(-1) would then reap slot 1's child under slot 0's name.
    #[test]
    fn a_spawn_whose_fork_fails_is_counted_and_shadows_no_sibling() {
        let Some(run) = fork_failure_run() else {
            return;
        };
        assert_eq!(run.get(12000), 0, "the process limit was lowered");
        assert_eq!(run.get(12001), 0, "the failed spawn still takes slot 0");
        assert_eq!(run.get(12002), 0, "the process limit was restored");
        assert_eq!(run.get(12003), SPAWN_FAILED, "and records why");
        assert_eq!(run.get(12004), -1, "no child was started");
        assert_eq!(run.get(12005), 1, "the next spawn takes slot 1");
        assert_eq!(run.get(12006), 1, "wait_all counts the failed fork");
        assert_eq!(run.get(12007), 1, "procgroup_failures agrees");
        assert_eq!(run.get(12008), SPAWN_FAILED, "slot 0 is still spawn_failed");
        assert_eq!(run.get(12009), OK, "slot 1 is reaped as itself");
        assert_eq!(run.get(12010), 1, "slot 1 output is its own");
        assert_eq!(run.get(12011), 1, "nothing is left running");
    }

    #[test]
    fn a_child_reads_eof_on_stdin_and_holds_no_sibling_pipe() {
        let Some(run) = shared() else {
            return;
        };
        // The parent's own stdin is a pipe the harness keeps open, so a child
        // that inherited it would block in read() until the deadline.
        assert_eq!(run.get(10900), 0, "every child finished");
        let first = run.get(10000);
        assert!(
            (1000..2000).contains(&first),
            "child 0 must read EOF (value 1000 + open fds), got {first}"
        );
        for i in 1..6 {
            assert_eq!(
                run.get(10000 + i),
                first,
                "child {i} holds the same descriptors as child 0"
            );
        }
    }

    #[test]
    fn children_reaped_by_the_kernel_are_reported_lost() {
        let Some(run) = shared() else {
            return;
        };
        assert_eq!(run.get(11900), 2, "both count as failures");
        assert_eq!(run.get(11000), LOST, "slot 0 lost");
        assert_eq!(run.get(11001), LOST, "slot 1 lost");
        assert_eq!(run.get(11002), 0, "slot 0 output is kept");
        assert_eq!(run.get(11003), 1, "slot 1 output is kept");
    }
}
