// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

mod common;

use common::xsi_gate::{self, CasePolicy, EvidenceKind, LegPolicy, RequiredIsa, VnniDecision};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

static SCRATCH_NONCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = SCRATCH_NONCE.fetch_add(1, Ordering::Relaxed);
        let path =
            crate::common::scratch_dir("cross_substrate_receipts").join(format!("{label}-{nonce}"));
        std::fs::create_dir(&path).expect("create receipt-test scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn required(id: impl Into<String>) -> CasePolicy {
    CasePolicy {
        id: id.into(),
        evidence: EvidenceKind::NativeRuntime,
        legs: LegPolicy::Required,
    }
}

fn expected(policies: impl IntoIterator<Item = CasePolicy>) -> BTreeMap<String, CasePolicy> {
    policies
        .into_iter()
        .map(|policy| (policy.id.clone(), policy))
        .collect()
}

fn vnni_policy() -> CasePolicy {
    CasePolicy {
        id: xsi_gate::VNNI_CASE.to_string(),
        evidence: EvidenceKind::NativeRuntime,
        legs: LegPolicy::DeferredWhenIsaMissing(RequiredIsa::Avx512Vnni),
    }
}

fn write_vnni_receipt(dir: &Path, outcome: &str, required_isa: &str) {
    std::fs::write(
        dir.join(format!("{}.receipt", xsi_gate::VNNI_CASE)),
        format!(
            "version=1\ncase={}\noutcome={outcome}\nevidence=native-runtime\n\
             substrate=avx2\nrequired_isa={required_isa}\n",
            xsi_gate::VNNI_CASE
        ),
    )
    .unwrap();
}

#[test]
fn manifest_inventory_names_every_case_and_only_the_narrow_deferral() {
    let inventory = xsi_gate::load_inventory(&xsi_gate::workload_root()).expect("load inventory");
    assert_eq!(inventory.len(), 35);
    let compile_only: Vec<_> = inventory
        .values()
        .filter(|p| p.evidence == EvidenceKind::CompilerConstruction)
        .map(|p| p.id.as_str())
        .collect();
    assert_eq!(compile_only, [xsi_gate::COMPILE_ONLY_CASE]);
    let deferred: Vec<_> = inventory
        .values()
        .filter(|p| matches!(p.legs, LegPolicy::DeferredWhenIsaMissing(_)))
        .map(|p| p.id.as_str())
        .collect();
    assert_eq!(deferred, [xsi_gate::VNNI_CASE]);
}

#[test]
fn concurrent_case_records_publish_as_whole_independent_files() {
    let scratch = Scratch::new("concurrent");
    let policies: Vec<_> = (0..32).map(|i| required(format!("case-{i}"))).collect();
    let barrier = Arc::new(Barrier::new(policies.len()));
    std::thread::scope(|scope| {
        for policy in &policies {
            let barrier = Arc::clone(&barrier);
            let dir = scratch.0.clone();
            scope.spawn(move || {
                barrier.wait();
                xsi_gate::record_measured_to(policy, &dir, "avx2")
                    .expect("publish independent receipt");
            });
        }
    });
    let summary = xsi_gate::validate_inventory(&expected(policies), &scratch.0, "avx2")
        .expect("validate concurrent receipts");
    assert_eq!(summary.measured_native, 32);
    assert_eq!(summary.total(), 32);
}

#[test]
fn duplicate_case_publication_is_refused_atomically() {
    let scratch = Scratch::new("duplicate");
    let policy = required("same-case");
    let barrier = Arc::new(Barrier::new(2));
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let dir = scratch.0.clone();
                let policy = policy.clone();
                scope.spawn(move || {
                    barrier.wait();
                    xsi_gate::record_measured_to(&policy, &dir, "avx2")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("writer thread"))
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    xsi_gate::validate_inventory(&expected([policy]), &scratch.0, "avx2")
        .expect("the one published receipt remains intact");
}

#[test]
fn missing_and_unexpected_receipts_fail_the_inventory() {
    let scratch = Scratch::new("sets");
    let one = required("case-one");
    let two = required("case-two");
    let extra = required("case-extra");
    xsi_gate::record_measured_to(&one, &scratch.0, "neon").unwrap();
    xsi_gate::record_measured_to(&extra, &scratch.0, "neon").unwrap();
    let errors = xsi_gate::validate_inventory(&expected([one, two]), &scratch.0, "neon")
        .expect_err("missing and unexpected cases must fail");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("missing receipt for `case-two`"))
    );
    assert!(
        errors
            .iter()
            .any(|e| e.contains("unexpected receipt for `case-extra`"))
    );
}

#[test]
fn a_deferred_receipt_cannot_launder_a_required_case() {
    let scratch = Scratch::new("bogus-defer");
    std::fs::write(
        scratch.0.join("required-case.receipt"),
        "version=1\ncase=required-case\noutcome=deferred\nevidence=native-runtime\n\
         substrate=avx2\nrequired_isa=avx512vnni\n",
    )
    .unwrap();
    let errors =
        xsi_gate::validate_inventory(&expected([required("required-case")]), &scratch.0, "avx2")
            .expect_err("required case cannot defer");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("not permitted by its manifest"))
    );
}

#[test]
fn vnni_policy_runs_only_on_supported_opted_in_hardware() {
    let policy = vnni_policy();
    assert_eq!(
        xsi_gate::vnni_decision(&policy, false, None).unwrap(),
        VnniDecision::DeferUnsupported(RequiredIsa::Avx512Vnni)
    );
    assert_eq!(
        xsi_gate::vnni_decision(&policy, true, Some("1")).unwrap(),
        VnniDecision::Run
    );
    assert!(xsi_gate::vnni_decision(&policy, true, None).is_err());
    assert!(xsi_gate::vnni_decision(&policy, true, Some("0")).is_err());
    assert!(xsi_gate::vnni_decision(&required(xsi_gate::VNNI_CASE), false, None).is_err());
}

#[test]
fn vnni_receipt_inventory_binds_outcome_to_detected_capability() {
    let policy = vnni_policy();

    let deferred = Scratch::new("vnni-deferred-supported");
    write_vnni_receipt(&deferred.0, "deferred", "avx512vnni");
    let errors = xsi_gate::validate_inventory_with_vnni(
        &expected([policy.clone()]),
        &deferred.0,
        "avx2",
        true,
    )
    .expect_err("supported VNNI hardware must not accept a deferred receipt");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("detector reports available"))
    );

    let measured = Scratch::new("vnni-measured-unsupported");
    write_vnni_receipt(&measured.0, "measured", "none");
    let errors = xsi_gate::validate_inventory_with_vnni(
        &expected([policy.clone()]),
        &measured.0,
        "avx2",
        false,
    )
    .expect_err("unsupported VNNI hardware must not accept a measured receipt");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("detector reports unavailable"))
    );

    let valid_deferred = Scratch::new("vnni-deferred-unsupported");
    write_vnni_receipt(&valid_deferred.0, "deferred", "avx512vnni");
    let summary = xsi_gate::validate_inventory_with_vnni(
        &expected([policy.clone()]),
        &valid_deferred.0,
        "avx2",
        false,
    )
    .expect("unsupported VNNI hardware may defer");
    assert_eq!(summary.deferred_native, 1);

    let valid_measured = Scratch::new("vnni-measured-supported");
    write_vnni_receipt(&valid_measured.0, "measured", "none");
    let summary = xsi_gate::validate_inventory_with_vnni(
        &expected([policy]),
        &valid_measured.0,
        "avx2",
        true,
    )
    .expect("supported VNNI hardware must measure");
    assert_eq!(summary.measured_native, 1);
}

#[test]
fn workflow_runs_the_inventory_validator_after_the_asserting_suite() {
    let workflow = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml"),
    )
    .expect("read CI workflow");
    let job = workflow
        .split("  cross_substrate_identity:")
        .nth(1)
        .expect("cross-substrate job")
        .split("\n  mindfuzz_cross_runner_identity:")
        .next()
        .expect("cross-substrate job body");
    let identity = job
        .find("--test cross_substrate_identity")
        .expect("identity command");
    let receipts = job
        .find("--test cross_substrate_receipts")
        .expect("receipt verifier");
    assert!(job.contains(xsi_gate::RECEIPT_DIR_VAR));
    assert!(job.contains("--include-ignored"));
    assert!(
        identity < receipts,
        "inventory must be checked after workload execution"
    );
}

#[test]
#[ignore = "the cross-substrate CI job supplies the asserting receipt directory"]
fn asserting_runtime_inventory_is_complete() {
    let dir = std::env::var_os(xsi_gate::RECEIPT_DIR_VAR)
        .map(PathBuf::from)
        .expect("MIND_XSI_RECEIPTS_DIR is required for the asserting inventory gate");
    let inventory = xsi_gate::load_inventory(&xsi_gate::workload_root()).expect("load inventory");
    let summary = xsi_gate::validate_inventory(&inventory, &dir, xsi_gate::host_substrate())
        .unwrap_or_else(|errors| panic!("runtime inventory invalid:\n  {}", errors.join("\n  ")));
    println!(
        "XSI-COVERAGE expected={} measured_native={} measured_construction={} deferred_native={}",
        inventory.len(),
        summary.measured_native,
        summary.measured_construction,
        summary.deferred_native
    );
    assert_eq!(summary.total(), inventory.len());
}
