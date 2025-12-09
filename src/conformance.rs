use crate::eval::ir_interp::eval_ir;
use crate::eval::value::Value;
use crate::pipeline::{compile_source, CompileOptions};
use crate::runtime::types::BackendTarget;

#[cfg(feature = "mlir-lowering")]
use crate::pipeline::{lower_to_mlir, MlirProducts};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConformanceProfile {
    CpuBaseline,
    CpuAndGpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConformanceOptions {
    pub profile: ConformanceProfile,
}

#[derive(Debug, thiserror::Error)]
#[error("conformance failures: {0:?}")]
pub struct ConformanceFailure(pub Vec<String>);

#[derive(Debug)]
struct ConformanceCase {
    name: &'static str,
    source: &'static str,
    target: BackendTarget,
    func: Option<&'static str>,
    expected_ir: &'static str,
    expected_value: Option<ExpectedValue>,
    #[cfg_attr(not(feature = "mlir-lowering"), allow(dead_code))]
    expected_mlir: Option<&'static str>,
    #[cfg(feature = "autodiff")]
    expected_grad_ir: Option<&'static str>,
    expected_error: Option<&'static str>,
    run_autodiff: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExpectedValue {
    Int(i64),
}

pub fn run_conformance(opts: ConformanceOptions) -> Result<(), ConformanceFailure> {
    let mut failures = Vec::new();

    for case in cpu_cases() {
        if let Err(msg) = run_case(&case) {
            failures.push(format!("cpu:{} => {msg}", case.name));
        }
    }

    if matches!(opts.profile, ConformanceProfile::CpuAndGpu) {
        for case in gpu_cases() {
            if let Err(msg) = run_case(&case) {
                failures.push(format!("gpu:{} => {msg}", case.name));
            }
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(ConformanceFailure(failures))
    }
}

fn run_case(case: &ConformanceCase) -> Result<(), String> {
    let compile_opts = CompileOptions {
        func: case.func.map(ToOwned::to_owned),
        enable_autodiff: case.run_autodiff,
        target: case.target,
    };

    match compile_source(case.source, &compile_opts) {
        Ok(products) => {
            if let Some(expected) = case.expected_error {
                return Err(format!(
                    "expected failure containing '{expected}' but compilation succeeded"
                ));
            }

            let rendered = normalize(&format!("{}", products.ir));
            if rendered != normalize(case.expected_ir) {
                return Err(format!(
                    "IR mismatch. expected:\n{}\nactual:\n{}",
                    case.expected_ir.trim(),
                    rendered
                ));
            }

            if let Some(value) = &case.expected_value {
                let evaluated = eval_ir(&products.ir);
                match (value, evaluated) {
                    (ExpectedValue::Int(expected), Value::Int(actual)) if *expected == actual => {}
                    (ExpectedValue::Int(expected), got) => {
                        return Err(format!("expected runtime value {expected}, got {got:?}"));
                    }
                }
            }

            #[cfg(feature = "autodiff")]
            if case.run_autodiff {
                let grad = products
                    .grad
                    .as_ref()
                    .ok_or_else(|| "autodiff results missing".to_string())?;

                if let Some(expected) = case.expected_grad_ir {
                    let rendered_grad = normalize(&format!("{}", grad.gradient_module));
                    if rendered_grad != normalize(expected) {
                        return Err(format!(
                            "gradient IR mismatch. expected:\n{}\nactual:\n{}",
                            expected.trim(),
                            rendered_grad
                        ));
                    }
                }
            }

            #[cfg(feature = "mlir-lowering")]
            if let Some(expected_mlir) = case.expected_mlir {
                let grads = {
                    #[cfg(feature = "autodiff")]
                    {
                        products.grad.as_ref()
                    }
                    #[cfg(not(feature = "autodiff"))]
                    {
                        None
                    }
                };

                let mlir: MlirProducts = lower_to_mlir(&products.ir, grads)
                    .map_err(|err| format!("MLIR lowering failed: {err}"))?;
                let rendered_mlir = normalize(&mlir.primal_mlir);
                if rendered_mlir != normalize(expected_mlir) {
                    return Err(format!(
                        "MLIR mismatch. expected:\n{}\nactual:\n{}",
                        expected_mlir.trim(),
                        rendered_mlir
                    ));
                }
            }

            Ok(())
        }
        Err(err) => {
            if let Some(expected) = case.expected_error {
                let msg = format!("{err}").to_lowercase();
                if msg.contains(&expected.to_lowercase()) {
                    Ok(())
                } else {
                    Err(format!("expected error containing '{expected}', got {msg}"))
                }
            } else {
                Err(format!("unexpected compile error: {err:?}"))
            }
        }
    }
}

fn normalize(text: &str) -> String {
    text.trim().replace('\r', "")
}

fn cpu_cases() -> Vec<ConformanceCase> {
    #[allow(unused_mut)]
    let mut cases = vec![ConformanceCase {
        name: "simple_arith",
        source: include_str!("../tests/conformance/cpu_baseline/simple_arith.mind"),
        target: BackendTarget::Cpu,
        func: None,
        expected_ir: include_str!("../tests/conformance/cpu_baseline/simple_arith.ir"),
        expected_value: Some(ExpectedValue::Int(7)),
        expected_mlir: Some(include_str!(
            "../tests/conformance/cpu_baseline/simple_arith.mlir"
        )),
        #[cfg(feature = "autodiff")]
        expected_grad_ir: None,
        expected_error: None,
        run_autodiff: false,
    }];

    #[cfg(feature = "autodiff")]
    cases.push(ConformanceCase {
        name: "autodiff_pairwise",
        source: include_str!("../tests/conformance/cpu_baseline/autodiff_pairwise.mind"),
        target: BackendTarget::Cpu,
        func: Some("main"),
        expected_ir: include_str!("../tests/conformance/cpu_baseline/autodiff_pairwise.ir"),
        expected_value: Some(ExpectedValue::Int(0)),
        expected_mlir: Some(include_str!(
            "../tests/conformance/cpu_baseline/autodiff_pairwise.mlir"
        )),
        expected_grad_ir: Some(include_str!(
            "../tests/conformance/cpu_baseline/autodiff_pairwise.grad.ir"
        )),
        expected_error: None,
        run_autodiff: true,
    });

    cases
}

fn gpu_cases() -> Vec<ConformanceCase> {
    #[allow(unused_mut)]
    let mut cases = Vec::new();

    #[cfg(feature = "mlir-gpu")]
    {
        cases.push(ConformanceCase {
            name: "backend_unavailable",
            source: include_str!("../tests/conformance/gpu_profile/backend_unavailable.mind"),
            target: BackendTarget::Gpu,
            func: None,
            expected_ir: "",
            expected_value: None,
            expected_mlir: None,
            #[cfg(feature = "autodiff")]
            expected_grad_ir: None,
            expected_error: Some("no backend available"),
            run_autodiff: false,
        });
    }

    cases
}
