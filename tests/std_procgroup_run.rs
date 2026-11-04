// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/procgroup.mind`: start many child processes, wait for all of them.
//!
//! Section A (`std-surface`): the module parses, lowers, exports its public
//! surface and resolves through the bundled stdlib.
//!
//! Section B (`mlir-build`, Linux): `tests/std_procgroup_run.mind` is built
//! with `mindc build` (the MLIR path) and run once. It is its own child: the
//! parent spawns the same executable through `std.procgroup` and prints
//! `k <key>` / `v <value>` pairs; each test below checks one case of it:
//! 25 children with distinct results, failing children reported, a straggler
//! killed at the deadline, pipes drained while children run, arguments and a
//! missing program, output truncation, the task index, refused inputs, a
//! spawn whose pipe cannot be made, stdin and descriptor isolation, and
//! children the kernel reaped.

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
        let dir = exe.parent().expect("artifact dir");
        let out_path = dir.join("run.stdout");
        let err_path = dir.join("run.stderr");
        let start = Instant::now();
        // stdin stays an open pipe until the program exits: a child that
        // inherited it instead of /dev/null would block reading it. The own
        // process group lets a timeout kill the children with the parent.
        let mut child = Command::new(exe)
            .env_remove("MIND_PROCGROUP_TASK")
            .env_remove("PG_CASE")
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

    fn shared() -> Option<&'static Run> {
        static RUN: OnceLock<Option<Result<Run, String>>> = OnceLock::new();
        let run = RUN.get_or_init(|| {
            let dir = crate::common::scratch_dir(TARGET);
            build(&dir).map(|exe| execute(&exe))
        });
        match run {
            None => None,
            Some(Ok(run)) => Some(run),
            Some(Err(why)) => panic!("{why}"),
        }
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
