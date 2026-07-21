// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::ast::BinOp;

use crate::ast::Literal;

use crate::ast::Module;

use crate::ast::Node;

use crate::ast::TensorElemOp;
use crate::ast::TypeAnn;
use crate::runtime_interface::{MindRuntime, NoOpRuntime};

use crate::eval::autodiff::TensorEnvEntry;
use crate::types::DType;
use crate::types::ShapeDim;
use crate::types::ValueType;

#[cfg(feature = "cpu-exec")]
use crate::exec;

#[cfg(feature = "cpu-buffers")]
use value::Buffer;

pub mod abi_gate;
pub mod autodiff;
pub mod conv2d_grad;
pub mod ir_interp;
pub mod lower;
// RFC 0005 P0f Step 2 — pre-pass that builds a span-keyed side-table
// of `FieldAccess` receiver struct types so lowering can resolve
// chained access, fn returns, and struct-typed parameters.
#[cfg(feature = "mlir-build")]
pub mod mlir_build;
pub mod mlir_export;
#[cfg(feature = "mlir-gpu")]
pub mod mlir_gpu;
#[cfg(feature = "mlir-jit")]
pub mod mlir_jit;
pub mod mlir_opt;
#[cfg(feature = "mlir-exec")]
pub mod mlir_run;
#[cfg(feature = "std-surface")]
pub mod struct_resolver;
pub mod value;

/// Top-level evaluation context used by the compiler front-end.
///
/// Carries a handle to the runtime implementation. In the open-core
/// build the default runtime is `NoOpRuntime`; production CPU/GPU
/// backends are provided by the proprietary `mind-runtime` crate.
pub struct Evaluator {
    pub runtime: Box<dyn MindRuntime>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self {
            runtime: Box::new(NoOpRuntime),
        }
    }
}

impl Evaluator {
    /// Construct an evaluator with the default no-op runtime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an evaluator with an explicit runtime implementation.
    pub fn with_runtime(runtime: Box<dyn MindRuntime>) -> Self {
        Self { runtime }
    }
}

#[cfg(test)]
mod tensor_tests {
    use super::*;
    use crate::runtime_interface::{DeviceKind, TensorDesc};
    use crate::types::DType;

    #[test]
    fn evaluator_uses_default_runtime() {
        let eval = Evaluator::new();

        let desc = TensorDesc {
            shape: Vec::new(),
            dtype: DType::F32,
            device: Some(DeviceKind::Cpu),
        };

        let handle = eval.runtime.allocate(&desc).unwrap();
        assert_eq!(handle, 0);

        eval.runtime.run_op("noop", &[], &[]).unwrap();
        eval.runtime.synchronize().unwrap();
    }
}

#[cfg(test)]
mod enum_value_tests {
    use super::*;
    use crate::ast::{Literal, MatchArm, Node, Pattern, Span};

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn eval_expr(node: &Node) -> Result<Value, EvalError> {
        let env: HashMap<String, Value> = HashMap::new();
        let tensor_env: HashMap<String, autodiff::TensorEnvEntry> = HashMap::new();
        eval_value_expr_mode(node, &env, &tensor_env, ExecMode::Preview)
    }

    /// `Result::Ok(7)` constructs a `Value::Enum` carrying the payload.
    #[test]
    fn enum_ctor_call_builds_value_enum() {
        let ctor = Node::Call {
            callee: "Result::Ok".to_string(),
            args: vec![Node::Lit(Literal::Int(7), sp())],
            span: sp(),
        };
        let got = eval_expr(&ctor).expect("enum ctor should evaluate");
        assert_eq!(
            got,
            Value::Enum {
                variant: "Result::Ok".to_string(),
                payload: vec![Value::Int(7)],
            }
        );
    }

    /// `match Result::Ok(7) { Result::Ok(x) => x, _ => 0 }` yields 7, binding
    /// the payload element to `x`.
    #[test]
    fn match_enum_variant_binds_payload() {
        let scrutinee = Node::Call {
            callee: "Result::Ok".to_string(),
            args: vec![Node::Lit(Literal::Int(7), sp())],
            span: sp(),
        };
        let m = Node::Match {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        path: "Result::Ok".to_string(),
                        args: vec![Pattern::Ident("x".to_string())],
                    },
                    guard: None,
                    body: Node::Lit(Literal::Ident("x".to_string()), sp()),
                    span: sp(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard,
                    guard: None,
                    body: Node::Lit(Literal::Int(0), sp()),
                    span: sp(),
                },
            ],
            span: sp(),
        };
        assert_eq!(eval_expr(&m).expect("match should evaluate"), Value::Int(7));
    }

    /// A non-matching variant (`Result::Err(...)` against `Result::Ok`) falls
    /// through to the wildcard arm.
    #[test]
    fn match_enum_variant_non_matching_falls_through() {
        let scrutinee = Node::Call {
            callee: "Result::Err".to_string(),
            args: vec![Node::Lit(Literal::Int(7), sp())],
            span: sp(),
        };
        let m = Node::Match {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                MatchArm {
                    pattern: Pattern::EnumVariant {
                        path: "Result::Ok".to_string(),
                        args: vec![Pattern::Ident("x".to_string())],
                    },
                    guard: None,
                    body: Node::Lit(Literal::Ident("x".to_string()), sp()),
                    span: sp(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard,
                    guard: None,
                    body: Node::Lit(Literal::Int(0), sp()),
                    span: sp(),
                },
            ],
            span: sp(),
        };
        assert_eq!(eval_expr(&m).expect("match should evaluate"), Value::Int(0));
    }
}

pub use ir_interp::eval_ir;
pub use lower::lower_to_ir;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildError as MlirBuildError;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildOptions as MlirBuildOptions;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildProducts as MlirBuildProducts;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildTools as MlirBuildTools;
#[cfg(feature = "mlir-build")]
pub use mlir_build::build_all as build_mlir_artifacts;
#[cfg(feature = "mlir-build")]
pub use mlir_build::resolve_tools as resolve_mlir_build_tools;
pub use mlir_export::MlirEmitMode;
pub use mlir_export::MlirEmitOptions;
pub use mlir_export::MlirLowerPreset;
pub use mlir_export::emit_mlir_with_opts;
pub use mlir_export::to_mlir;
#[cfg(feature = "mlir-exec")]
pub use mlir_run::MlirExecConfig;
pub use value::TensorVal;
pub use value::Value;
pub use value::VarId;
pub use value::format_value_human;

pub fn emit_mlir_string(ir: &crate::ir::IRModule, preset: mlir_export::MlirLowerPreset) -> String {
    let opts = mlir_export::MlirEmitOptions {
        lower_preset: Some(preset.as_str().to_string()),
        ..Default::default()
    };
    mlir_export::emit_mlir_with_opts(ir, &opts)
}

pub fn emit_mlir_to_file(
    ir: &crate::ir::IRModule,
    preset: mlir_export::MlirLowerPreset,
    path: &std::path::Path,
) -> std::io::Result<()> {
    let txt = emit_mlir_string(ir, preset);
    std::fs::create_dir_all(path.parent().unwrap_or_else(|| std::path::Path::new(".")))?;
    std::fs::write(path, txt)
}

#[cfg(feature = "cpu-buffers")]
pub(crate) fn num_elems(shape: &[ShapeDim]) -> Option<usize> {
    let mut n: usize = 1;
    for d in shape {
        match d {
            ShapeDim::Known(k) => {
                n = n.saturating_mul(*k);
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(n)
}

#[cfg(feature = "cpu-buffers")]
pub(crate) const MATERIALIZE_MAX: usize = 1_048_576;

#[cfg(feature = "cpu-buffers")]
pub(crate) fn materialize_filled(t: &mut TensorVal) {
    if t.buf.is_some() {
        return;
    }
    let fill = match t.fill {
        Some(f) => f,
        None => return,
    };
    if let Some(ne) = num_elems(&t.shape) {
        if ne <= MATERIALIZE_MAX {
            match t.dtype {
                DType::I32 => {
                    let v = fill as i32;
                    t.buf = Some(Buffer::I32(vec![v; ne]));
                }
                DType::F32 => {
                    let v = fill as f32;
                    t.buf = Some(Buffer::F32(vec![v; ne]));
                }
                _ => {}
            }
        }
    }
}

mod stdlib;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecMode {
    Preview,
    CpuExec,
    /// GPU execution mode
    Cuda,
    #[cfg(feature = "mlir-exec")]
    MlirExternal(MlirExecConfig),
    #[cfg(feature = "mlir-jit")]
    MlirJitCpu,
    #[cfg(feature = "mlir-gpu")]
    MlirGpu {
        backend: GpuBackend,
        blocks: (u32, u32, u32),
        threads: (u32, u32, u32),
    },
}

#[cfg(feature = "mlir-gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    Rocm,
}

#[cfg(feature = "cpu-exec")]
pub(crate) fn exec_error_to_eval(err: exec::cpu::ExecError) -> EvalError {
    match err {
        exec::cpu::ExecError::Math(msg) => {
            if msg.contains("division by zero") {
                EvalError::DivZero
            } else {
                EvalError::UnsupportedMsg(msg)
            }
        }
        exec::cpu::ExecError::Unsupported(msg)
        | exec::cpu::ExecError::Shape(msg)
        | exec::cpu::ExecError::Type(msg) => EvalError::UnsupportedMsg(msg),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unsupported operation")]
    Unsupported,
    #[error("unsupported: {0}")]
    UnsupportedMsg(String),
    #[error("division by zero")]
    DivZero,
    #[error("unknown variable: {0}")]
    UnknownVar(String),
    #[error("type error: {msg}")]
    TypeError {
        msg: String,
        diagnostics: Vec<crate::diagnostics::Diagnostic>,
    },
    #[error("out of bounds")]
    OutOfBounds,
    /// Control-flow signal — NOT a real error. Carries the value of an early
    /// `return X` so the enclosing function-body evaluation short-circuits.
    /// The `?` operator propagates it up through `Block`/`If`/`For`/`While`
    /// (stopping any enclosing loop) until the `Node::Call` boundary catches it
    /// and unwraps the value. This mirrors the compiled/native path, where an
    /// early `return` correctly stops evaluation. It never escapes a function
    /// body: a bare `return` outside a function is rejected by the type checker.
    #[error("internal: unhandled return control-flow signal")]
    ReturnFlow(Box<Value>),
}

pub fn eval_module_value_with_env_mode(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if let Some(src) = src_for_types {
        let mut tenv: crate::type_checker::TypeEnv = crate::type_checker::TypeEnv::default();
        for name in env.keys() {
            tenv.insert(name.clone(), ValueType::ScalarI32);
        }
        // Pre-register for-loop variables in the type environment
        for item in &m.items {
            if let Node::For { var, .. } = item {
                tenv.insert(var.clone(), ValueType::ScalarI32);
            }
        }
        let diags = crate::type_checker::check_module_types(m, src, &tenv);
        // Filter out unknown-identifier errors for loop variables
        let real_diags: Vec<_> = diags
            .into_iter()
            .filter(|d| {
                // Keep all non-E2001 errors; for E2001, check if it's about a for-loop var
                if d.code != "E2001" {
                    return true;
                }
                // Check if any for-loop declares this variable
                !m.items.iter().any(|item| {
                    if let Node::For { var, body, .. } = item {
                        d.message.contains(var)
                            || body.iter().any(|s| {
                                if let Node::For { var: inner_var, .. } = s {
                                    d.message.contains(inner_var)
                                } else {
                                    false
                                }
                            })
                    } else {
                        false
                    }
                })
            })
            .collect();
        if !real_diags.is_empty() {
            let msg = real_diags
                .iter()
                .map(|diag| format!("[{}] {}", diag.code, diag.message))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(EvalError::TypeError {
                msg,
                diagnostics: real_diags,
            });
        }
    }

    let mut venv: HashMap<String, Value> = env
        .iter()
        .map(|(name, value)| (name.clone(), Value::Int(*value)))
        .collect();
    let mut tensor_env: HashMap<String, TensorEnvEntry> = HashMap::new();

    // Register all top-level functions so calls can dispatch to them (generic or
    // not). Restored after the module finishes so nested module evals don't leak.
    let _fn_prev = fn_table_install(m);
    // issue #99: start with a clean u64-var set (restored below) so a nested
    // module eval neither sees nor leaks the caller's u64 declarations.
    let _u64_prev = u64_vars_take();

    let mut last = Value::Int(0_i64);
    for item in &m.items {
        match item {
            Node::Let {
                name, ann, value, ..
            } => {
                let rhs = eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                let stored = match ann {
                    Some(TypeAnn::Tensor { dtype, dims, .. })
                    | Some(TypeAnn::DiffTensor { dtype, dims }) => {
                        let (dtype, shape) = parse_tensor_ann(dtype, dims)?;
                        let fill = match rhs {
                            Value::Int(n) => Some(n as f64),
                            Value::Tensor(ref t) => t.fill,
                            _ => None,
                        };
                        Value::Tensor(TensorVal::new(dtype, shape, fill))
                    }
                    Some(TypeAnn::ScalarI32)
                    | Some(TypeAnn::ScalarI64)
                    | Some(TypeAnn::ScalarF32)
                    | Some(TypeAnn::ScalarF64)
                    | Some(TypeAnn::ScalarBool)
                    | Some(TypeAnn::ScalarU32)
                    | Some(TypeAnn::Named(_))
                    | Some(TypeAnn::Slice { .. })
                    | Some(TypeAnn::Array { .. })
                    | Some(TypeAnn::Ref { .. })
                    | Some(TypeAnn::Generic { .. })
                    | Some(TypeAnn::Tuple { .. })
                    | Some(TypeAnn::SparseTensor { .. })
                    | Some(TypeAnn::RawPtr { .. })
                    | Some(TypeAnn::FnPtr { .. })
                    | None => rhs,
                };
                // issue #99: track (or clear, on a non-u64 shadow) the binding's
                // declared u64-ness so a later `b > c` / `b >> n` on this name
                // picks the UNSIGNED interpreter path (matching the artifact).
                u64_vars_set(name, matches!(ann, Some(TypeAnn::Named(n)) if n == "u64"));
                if let Value::Int(n) = stored {
                    env.insert(name.clone(), n);
                    venv.insert(name.clone(), Value::Int(n));
                    last = Value::Int(n);
                } else {
                    venv.insert(name.clone(), stored.clone());
                    last = stored;
                }
                match venv.get(name) {
                    Some(Value::Tensor(tensor)) => {
                        let expr = match ann {
                            Some(TypeAnn::Tensor { .. }) => None,
                            _ => Some((**value).clone()),
                        };
                        tensor_env.insert(
                            name.clone(),
                            TensorEnvEntry {
                                value: tensor.clone(),
                                expr,
                            },
                        );
                    }
                    _ => {
                        tensor_env.remove(name);
                    }
                }
            }
            Node::Assign { name, value, .. } => {
                let rhs = eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                if let Value::Int(n) = rhs {
                    env.insert(name.clone(), n);
                    venv.insert(name.clone(), Value::Int(n));
                    last = Value::Int(n);
                } else {
                    venv.insert(name.clone(), rhs.clone());
                    last = rhs;
                }
                match venv.get(name) {
                    Some(Value::Tensor(tensor)) => {
                        tensor_env.insert(
                            name.clone(),
                            TensorEnvEntry {
                                value: tensor.clone(),
                                expr: Some((**value).clone()),
                            },
                        );
                    }
                    _ => {
                        tensor_env.remove(name);
                    }
                }
            }
            // Phase 10.6: `arr[i] = v` for an array variable. The interpreter is
            // immutable-value, so we rebuild the tuple with the element replaced and
            // rebind it (no in-place mutation). Only a simple variable receiver is
            // supported here; tensor/slice writes need the runtime handle-table and
            // keep the prior placeholder behavior (the statement evaluates to its RHS).
            Node::IndexAssign {
                receiver,
                index,
                value,
                ..
            } => {
                let val = eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                if let Node::Lit(Literal::Ident(name), _) = receiver.as_ref() {
                    if let Some(Value::Tuple(mut items)) = venv.get(name).cloned() {
                        let idx =
                            match eval_value_expr_mode(index, &venv, &tensor_env, mode.clone())? {
                                Value::Int(i) => i,
                                other => {
                                    return Err(EvalError::UnsupportedMsg(format!(
                                        "array index must be an integer, got {other:?}"
                                    )));
                                }
                            };
                        if idx < 0 || idx as usize >= items.len() {
                            return Err(EvalError::UnsupportedMsg(format!(
                                "array index {idx} out of bounds (len {})",
                                items.len()
                            )));
                        }
                        items[idx as usize] = val.clone();
                        venv.insert(name.clone(), Value::Tuple(items));
                    }
                }
                last = val;
            }
            Node::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                // Module-level for-loop with mutable environment propagation
                let s = match eval_value_expr_mode(start, &venv, &tensor_env, mode.clone())? {
                    Value::Int(n) => n,
                    _ => {
                        return Err(EvalError::UnsupportedMsg(
                            "for-loop start must be int".into(),
                        ));
                    }
                };
                let e = match eval_value_expr_mode(end, &venv, &tensor_env, mode.clone())? {
                    Value::Int(n) => n,
                    _ => return Err(EvalError::UnsupportedMsg("for-loop end must be int".into())),
                };
                for i in s..e {
                    venv.insert(var.clone(), Value::Int(i));
                    for stmt in body {
                        match stmt {
                            Node::Assign { name, value, .. } => {
                                let rhs =
                                    eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                                if let Value::Int(n) = &rhs {
                                    env.insert(name.clone(), *n);
                                }
                                venv.insert(name.clone(), rhs.clone());
                                last = rhs;
                            }
                            Node::Let { name, value, .. } => {
                                let rhs =
                                    eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                                if let Value::Int(n) = &rhs {
                                    env.insert(name.clone(), *n);
                                }
                                venv.insert(name.clone(), rhs.clone());
                                last = rhs;
                            }
                            // `arr[i] = v` inside the loop body: rebuild + rebind
                            // the array tuple (mirrors the module-level handler).
                            Node::IndexAssign {
                                receiver,
                                index,
                                value,
                                ..
                            } => {
                                let val =
                                    eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                                if let Node::Lit(Literal::Ident(arr), _) = receiver.as_ref() {
                                    if let Some(Value::Tuple(mut items)) = venv.get(arr).cloned() {
                                        let idx = match eval_value_expr_mode(
                                            index,
                                            &venv,
                                            &tensor_env,
                                            mode.clone(),
                                        )? {
                                            Value::Int(i) => i,
                                            other => {
                                                return Err(EvalError::UnsupportedMsg(format!(
                                                    "array index must be an integer, got {other:?}"
                                                )));
                                            }
                                        };
                                        if idx < 0 || idx as usize >= items.len() {
                                            return Err(EvalError::UnsupportedMsg(format!(
                                                "array index {idx} out of bounds (len {})",
                                                items.len()
                                            )));
                                        }
                                        items[idx as usize] = val.clone();
                                        venv.insert(arr.clone(), Value::Tuple(items));
                                    }
                                }
                                last = val;
                            }
                            _ => {
                                last =
                                    eval_value_expr_mode(stmt, &venv, &tensor_env, mode.clone())?;
                            }
                        }
                    }
                }
            }
            _ => {
                last = eval_value_expr_mode(item, &venv, &tensor_env, mode.clone())?;
            }
        }
    }

    #[allow(unused_variables)]
    match mode {
        ExecMode::Preview | ExecMode::CpuExec | ExecMode::Cuda => Ok(last),
        #[cfg(feature = "mlir-exec")]
        ExecMode::MlirExternal(cfg) => {
            let ir = lower_to_ir(m);
            let opts = mlir_export::MlirEmitOptions {
                mode: mlir_export::MlirEmitMode::Executable,
                ..Default::default()
            };
            let mlir_text = mlir_export::emit_mlir_with_opts(&ir, &opts);
            match mlir_run::exec_mlir_text(&mlir_text, &cfg) {
                Ok(stdout) => {
                    if stdout.trim().is_empty() {
                        return Ok(last);
                    }
                    if let Some(parsed) = parse_mlir_stdout(&stdout) {
                        return Ok(parsed);
                    }
                    Ok(Value::Str(stdout))
                }
                Err(msg) => Err(EvalError::UnsupportedMsg(msg)),
            }
        }
        #[cfg(feature = "mlir-jit")]
        ExecMode::MlirJitCpu => {
            let ir = lower_to_ir(m);
            let mut opts = mlir_export::MlirEmitOptions::default();
            opts.mode = mlir_export::MlirEmitMode::Executable;
            opts.lower_preset = Some(MlirLowerPreset::JitCpu.as_str().to_string());
            let mlir_text = mlir_export::emit_mlir_with_opts(&ir, &opts);
            match mlir_jit::MlirJit::new() {
                Ok(jit) => match jit.run_mlir_text(&mlir_text, "main", &[]) {
                    Ok(()) => Ok(last),
                    Err(mlir_jit::JitError::NotFound) | Err(mlir_jit::JitError::Unsupported) => {
                        eprintln!(
                            "mlir-jit runtime unavailable; falling back to preview or external execution"
                        );
                        #[cfg(feature = "mlir-exec")]
                        {
                            let fallback_cfg = MlirExecConfig::default();
                            if let Ok(stdout) = mlir_run::exec_mlir_text(&mlir_text, &fallback_cfg)
                            {
                                if stdout.trim().is_empty() {
                                    return Ok(last);
                                }
                                if let Some(parsed) = parse_mlir_stdout(&stdout) {
                                    return Ok(parsed);
                                }
                                return Ok(Value::Str(stdout));
                            }
                        }
                        Ok(last)
                    }
                    Err(mlir_jit::JitError::Invoke(msg)) => Err(EvalError::UnsupportedMsg(msg)),
                },
                Err(mlir_jit::JitError::NotFound) => {
                    eprintln!(
                        "mlir-jit shared libraries not found; falling back to preview execution"
                    );
                    Ok(last)
                }
                Err(err) => Err(EvalError::UnsupportedMsg(err.to_string())),
            }
        }
        #[cfg(feature = "mlir-gpu")]
        ExecMode::MlirGpu {
            backend,
            blocks,
            threads,
        } => {
            let ir = lower_to_ir(m);
            let opts = mlir_export::MlirEmitOptions {
                mode: mlir_export::MlirEmitMode::Executable,
                lower_preset: Some(MlirLowerPreset::GpuDefault.as_str().to_string()),
            };
            let mlir_text = mlir_export::emit_mlir_with_opts(&ir, &opts);
            let cfg = mlir_gpu::GpuLaunchCfg { blocks, threads };
            match mlir_gpu::run_mlir_gpu_text(&mlir_text, backend, cfg) {
                Ok(()) => Ok(last),
                Err(err) => {
                    eprintln!("{}", err);
                    Ok(last)
                }
            }
        }
    }
}

pub fn eval_module_value_with_env(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
) -> Result<Value, EvalError> {
    eval_module_value_with_env_mode(m, env, src_for_types, ExecMode::Preview)
}

pub fn eval_module_with_env(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
) -> Result<i64, EvalError> {
    match eval_module_value_with_env_mode(m, env, src_for_types, ExecMode::Preview)? {
        Value::Int(n) => Ok(n),
        _ => Err(EvalError::Unsupported),
    }
}

pub fn eval_module(m: &Module) -> Result<i64, EvalError> {
    let mut env: HashMap<String, i64> = HashMap::new();
    eval_module_with_env(m, &mut env, None)
}

pub fn eval_first_expr(m: &Module) -> Result<i64, EvalError> {
    eval_module(m)
}

/// Phase 10.7: does this call-callee denote an enum/`Option`/`Result` variant
/// constructor with a payload? A `Type::Variant` path (carrying `::`) is the
/// canonical form; `Some` is the only bare payload-carrying constructor the
/// interpreter recognizes by name. Tensor-stdlib callees never use `::` and are
/// never named `Some`, so this never shadows them.
fn is_enum_variant_ctor(callee: &str) -> bool {
    callee.contains("::") || callee == "Some"
}

/// Phase 10.7: does this bare identifier denote a payload-less (unit) variant?
/// `Type::Variant` paths and the bare `None` constructor qualify.
fn is_enum_unit_ctor(name: &str) -> bool {
    name.contains("::") || name == "None"
}

/// Phase 10.7: bind an enum-variant pattern's payload sub-patterns against the
/// matched payload values. Returns `false` (no match) if any sub-pattern fails;
/// successful bindings are appended to `out`. Ident sub-patterns bind the
/// corresponding payload element; literals must compare equal; nested
/// enum-variant sub-patterns recurse; wildcard always matches.
fn match_enum_payload(
    patterns: &[crate::ast::Pattern],
    values: &[Value],
    out: &mut Vec<(String, Value)>,
) -> bool {
    for (pat, value) in patterns.iter().zip(values.iter()) {
        match pat {
            crate::ast::Pattern::Wildcard => {}
            crate::ast::Pattern::Ident(name) => out.push((name.clone(), value.clone())),
            crate::ast::Pattern::Literal(lit) => {
                let eq = match (lit, value) {
                    (crate::ast::Literal::Int(m), Value::Int(n)) => n == m,
                    (crate::ast::Literal::Float(a), Value::Float(b)) => a == b,
                    (crate::ast::Literal::Str(s), Value::Str(v)) => v == s,
                    _ => false,
                };
                if !eq {
                    return false;
                }
            }
            crate::ast::Pattern::EnumVariant { path, args } => match value {
                Value::Enum { variant, payload }
                    if variant == path && payload.len() == args.len() =>
                {
                    if !match_enum_payload(args, payload, out) {
                        return false;
                    }
                }
                _ => return false,
            },
            // Tuple sub-pattern `(a, b)` destructures a tuple value of equal
            // arity, recursing each element (enum_match #9, e.g.
            // `Ok((p1, decorators))`).
            crate::ast::Pattern::Tuple(elems) => match value {
                Value::Tuple(items) if items.len() == elems.len() => {
                    if !match_enum_payload(elems, items, out) {
                        return false;
                    }
                }
                _ => return false,
            },
            // Struct-variant sub-pattern `E.V { f, g }`. The interpreter is the
            // const-eval path (mind-flow uses the RUNS path), and lacks the enum
            // decl's field_names here, so it binds each named field's sub-pattern
            // positionally in source order against the enum payload.
            crate::ast::Pattern::EnumStruct { fields, .. } => match value {
                Value::Enum { payload, .. } if payload.len() >= fields.len() => {
                    let subs: Vec<crate::ast::Pattern> =
                        fields.iter().map(|(_, p)| p.clone()).collect();
                    if !match_enum_payload(&subs, payload, out) {
                        return false;
                    }
                }
                _ => return false,
            },
        }
    }
    true
}

pub(crate) fn eval_value_expr_mode(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    match node {
        Node::Lit(Literal::Int(n), _) => Ok(Value::Int(*n)),
        Node::Lit(Literal::Float(f), _) => Ok(Value::Float(*f)),
        Node::Lit(Literal::Str(s), _) => Ok(Value::Str(s.clone())),
        Node::Lit(Literal::Ident(name), _) => {
            if let Some(v) = env.get(name) {
                return Ok(v.clone());
            }
            // Phase 10.7: a bare unit enum/`Option` variant (`Mode::On`, `None`)
            // that is not a bound variable evaluates to a payload-less
            // `Value::Enum`.
            if is_enum_unit_ctor(name) {
                return Ok(Value::Enum {
                    variant: name.clone(),
                    payload: Vec::new(),
                });
            }
            Err(EvalError::UnknownVar(name.clone()))
        }
        Node::Paren(inner, _) => eval_value_expr_mode(inner, env, tensor_env, mode.clone()),
        Node::Tuple { elements, .. } => {
            let mut items = Vec::with_capacity(elements.len());
            for item in elements {
                items.push(eval_value_expr_mode(item, env, tensor_env, mode.clone())?);
            }
            Ok(Value::Tuple(items))
        }
        Node::Call { callee, args, .. } => {
            // Phase 10.7: enum/`Option`/`Result` variant construction. A callee
            // written as a `Type::Variant` path (e.g. `Result::Ok(x)`,
            // `Mode::On(v)`) or the bare `Option` constructors `Some`/`None`
            // builds a `Value::Enum` carrying the evaluated positional payload.
            // This is detected before the tensor-stdlib dispatch because those
            // callees never use `::` and are never named `Some`/`None`.
            if is_enum_variant_ctor(callee) {
                let mut payload = Vec::with_capacity(args.len());
                for arg in args {
                    payload.push(eval_value_expr_mode(arg, env, tensor_env, mode.clone())?);
                }
                return Ok(Value::Enum {
                    variant: callee.clone(),
                    payload,
                });
            }
            // User-defined function call: look up the installed function table,
            // bind args to params in a fresh scope, evaluate the body (last
            // statement's value wins). Generic fns work for free — the dynamically
            // typed interpreter binds the concrete arg Values; type params are
            // only recorded on the FnDef. Checked before the tensor stdlib so a
            // user fn shadowing a stdlib name resolves to the user fn.
            if let Some(func) = fn_table_lookup(callee) {
                if func.params.len() != args.len() {
                    return Err(EvalError::UnsupportedMsg(format!(
                        "function `{callee}` expects {} argument(s), got {}",
                        func.params.len(),
                        args.len()
                    )));
                }
                let mut call_env = env.clone();
                for (p, a) in func.params.iter().zip(args.iter()) {
                    let v = eval_value_expr_mode(a, env, tensor_env, mode.clone())?;
                    call_env.insert(p.clone(), v);
                }
                // Salov C3 (#179): the function-body boundary is where an early
                // `return` is caught. A `ReturnFlow` signal raised anywhere in the
                // body (directly, or bubbled up through `?` from a nested
                // `if`/`for`/`while`) short-circuits here and yields its value;
                // otherwise the last statement's value wins (implicit return).
                //
                // `let`/`assign` statements at the body's top level are threaded
                // into `call_env` so a later statement (e.g. a `while` condition
                // reading a loop counter declared just above it) sees the update —
                // the same binding-propagation the module / `For` / `While` body
                // loops already perform. Without it a body with a top-level
                // counter could not be const-evaluated at all.
                let mut result = Value::Int(0);
                for stmt in &func.body {
                    match stmt {
                        Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
                            let v =
                                eval_value_expr_mode(value, &call_env, tensor_env, mode.clone())?;
                            call_env.insert(name.clone(), v.clone());
                            result = v;
                        }
                        _ => {
                            match eval_value_expr_mode(stmt, &call_env, tensor_env, mode.clone()) {
                                Ok(v) => result = v,
                                Err(EvalError::ReturnFlow(v)) => return Ok(*v),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                return Ok(result);
            }
            stdlib::tensor::dispatch(callee, args, env, tensor_env, mode.clone())
        }
        Node::CallTensorSum {
            x, axes, keepdims, ..
        } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if t.buf.is_some() && axes.is_empty() {
                        if let Ok(result) = exec::cpu::exec_sum_all(&t) {
                            return Ok(Value::Tensor(result));
                        }
                    }
                    let result = stdlib::tensor::sum_tensor_preview(&t, axes, *keepdims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallTensorMean {
            x, axes, keepdims, ..
        } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if t.buf.is_some() && axes.is_empty() {
                        if let Ok(result) = exec::cpu::exec_mean_all(&t) {
                            return Ok(Value::Tensor(result));
                        }
                    }
                    let result = stdlib::tensor::mean_tensor_preview(&t, axes, *keepdims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallReshape { x, dims, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) && t.buf.is_some() {
                        let new_shape = stdlib::tensor::dims_from_strings_usize(dims);
                        if let Some(new_shape) = new_shape {
                            if let Ok(result) = exec::cpu::exec_reshape(&t, new_shape) {
                                return Ok(Value::Tensor(result));
                            }
                        }
                    }
                    let result = stdlib::tensor::reshape_tensor_preview(&t, dims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallExpandDims { x, axis, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) && t.buf.is_some() {
                        let rank = t.shape.len();
                        let a = stdlib::tensor::normalize_expand_axis_usize(*axis, rank);
                        if let Some(a) = a {
                            if let Ok(result) = exec::cpu::exec_expand_dims(&t, a) {
                                return Ok(Value::Tensor(result));
                            }
                        }
                    }
                    let result = stdlib::tensor::expand_dims_tensor_preview(&t, *axis)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallSqueeze { x, axes, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) && t.buf.is_some() {
                        let norm = stdlib::tensor::normalize_squeeze_axes_usize(&t.shape, axes);
                        if let Some(norm) = norm {
                            if let Ok(result) = exec::cpu::exec_squeeze(&t, &norm) {
                                return Ok(Value::Tensor(result));
                            }
                        }
                    }
                    let result = stdlib::tensor::squeeze_tensor_preview(&t, axes)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallTranspose { x, axes, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) && t.buf.is_some() {
                        let axes_ref = axes.as_ref().map(|v| v.as_slice());
                        let perm = stdlib::tensor::compute_transpose_perm(&t.shape, axes_ref);
                        if let Some(perm) = perm {
                            if let Ok(result) = exec::cpu::exec_transpose(&t, &perm) {
                                return Ok(Value::Tensor(result));
                            }
                        }
                    }
                    let axes_ref = axes.as_ref().map(|v| v.as_slice());
                    let (result, _) = stdlib::tensor::transpose_tensor_preview(&t, axes_ref)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallIndex { x, axis, i, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    #[cfg(feature = "cpu-exec")]
                    if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) && t.buf.is_some() {
                        let rank = t.shape.len();
                        let axis_n = stdlib::tensor::normalize_axis_usize(*axis, rank);
                        if let Some(axis_n) = axis_n {
                            let idx = if *i < 0 {
                                match t.shape.get(axis_n) {
                                    Some(ShapeDim::Known(n)) => Some((*n as i32 + *i) as usize),
                                    _ => None,
                                }
                            } else {
                                Some(*i as usize)
                            };
                            if let Some(idx) = idx {
                                if let Ok(result) = exec::cpu::exec_index(&t, axis_n, idx) {
                                    return Ok(Value::Tensor(result));
                                }
                            }
                        }
                    }
                    let result = stdlib::tensor::index_tensor_preview(&t, *axis, *i)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallSlice {
            x,
            axis,
            start,
            end,
            ..
        } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::slice_tensor_preview(&t, *axis, *start, *end)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            ..
        } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::slice_stride_tensor_preview(
                        &t, *axis, *start, *end, *step,
                    )?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallGather { x, axis, idx, .. } => {
            let base = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            let indices = eval_value_expr_mode(idx, env, tensor_env, mode.clone())?;
            match (base, indices) {
                (Value::Tensor(t), Value::Tensor(i)) => {
                    let result = stdlib::tensor::gather_tensor_preview(&t, *axis, &i)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallDot { a, b, .. } => {
            let left = eval_value_expr_mode(a, env, tensor_env, mode.clone())?;
            let right = eval_value_expr_mode(b, env, tensor_env, mode.clone())?;
            match (left, right) {
                (Value::Tensor(tl), Value::Tensor(tr)) => {
                    #[cfg(feature = "cpu-buffers")]
                    {
                        if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) {
                            let mut tl_exec = tl.clone();
                            let mut tr_exec = tr.clone();
                            materialize_filled(&mut tl_exec);
                            materialize_filled(&mut tr_exec);
                            #[cfg(feature = "cpu-exec")]
                            {
                                // TODO(runtime): dispatch through `Evaluator::runtime` once the
                                // runtime plumbing is threaded into evaluation.
                                if tl_exec.dtype == DType::F32 && tr_exec.dtype == DType::F32 {
                                    let exec_res = exec::cpu::exec_dot(&tl_exec, &tr_exec);
                                    match exec_res {
                                        Ok(t) => return Ok(Value::Tensor(t)),
                                        Err(err) => {
                                            let mapped = exec_error_to_eval(err);
                                            if matches!(mapped, EvalError::DivZero) {
                                                return Err(mapped);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let result = stdlib::tensor::dot_tensor_preview(&tl, &tr)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallMatMul { a, b, .. } => {
            let left = eval_value_expr_mode(a, env, tensor_env, mode.clone())?;
            let right = eval_value_expr_mode(b, env, tensor_env, mode.clone())?;
            match (left, right) {
                (Value::Tensor(tl), Value::Tensor(tr)) => {
                    #[cfg(feature = "cpu-buffers")]
                    {
                        if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) {
                            let mut tl_exec = tl.clone();
                            let mut tr_exec = tr.clone();
                            materialize_filled(&mut tl_exec);
                            materialize_filled(&mut tr_exec);
                            #[cfg(feature = "cpu-exec")]
                            {
                                if tl_exec.dtype == DType::F32 && tr_exec.dtype == DType::F32 {
                                    // Try GPU matmul dispatch if available
                                    if matches!(mode, ExecMode::Cuda) {
                                        let gpu_result = GPU_MATMUL_FN.with(|f| {
                                            let func = f.borrow();
                                            if let Some(ref matmul_fn) = *func {
                                                Some(matmul_fn(&tl_exec, &tr_exec))
                                            } else {
                                                None
                                            }
                                        });
                                        if let Some(Ok(result)) = gpu_result {
                                            return Ok(Value::Tensor(result));
                                        }
                                    }
                                    // CPU fallback
                                    let exec_res = exec::cpu::exec_matmul(&tl_exec, &tr_exec);
                                    match exec_res {
                                        Ok(t) => return Ok(Value::Tensor(t)),
                                        Err(err) => {
                                            let mapped = exec_error_to_eval(err);
                                            if matches!(mapped, EvalError::DivZero) {
                                                return Err(mapped);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let result = stdlib::tensor::matmul_tensor_preview(&tl, &tr)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        // RFC 0012 Phase B: `A @ B` evaluates identically to
        // `tensor.matmul(A, B)` — delegate to the same implementation.
        Node::TensorMatmul { lhs, rhs, span } => {
            let synthetic = Node::CallMatMul {
                a: lhs.clone(),
                b: rhs.clone(),
                span: *span,
            };
            eval_value_expr_mode(&synthetic, env, tensor_env, mode)
        }
        // RFC 0012 Phase B: elementwise `.+ .- .* ./` evaluate identically
        // to scalar `+ - * /` on tensor operands.
        Node::TensorElemwise { op, lhs, rhs, span } => {
            let scalar_op = match op {
                TensorElemOp::Add => BinOp::Add,
                TensorElemOp::Sub => BinOp::Sub,
                TensorElemOp::Mul => BinOp::Mul,
                TensorElemOp::Div => BinOp::Div,
            };
            let synthetic = Node::Binary {
                op: scalar_op,
                left: lhs.clone(),
                right: rhs.clone(),
                span: *span,
            };
            eval_value_expr_mode(&synthetic, env, tensor_env, mode)
        }
        Node::CallTensorRelu { x, .. } => {
            let value = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::relu_tensor(t, mode.clone())?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallTensorRand { shape, .. } => {
            let dims: Vec<ShapeDim> = shape.iter().map(|&d| ShapeDim::Known(d)).collect();
            let n = shape.iter().product::<usize>();
            // Actually fill with random data so GPU dispatch gets real values
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            let mut seed = hasher.finish();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    // xorshift64
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            #[cfg(feature = "cpu-buffers")]
            let mut t = TensorVal::new(DType::F32, dims, None);
            #[cfg(not(feature = "cpu-buffers"))]
            let t = TensorVal::new(DType::F32, dims, None);
            #[cfg(feature = "cpu-buffers")]
            {
                t.buf = Some(crate::eval::value::Buffer::F32(data));
            }
            #[cfg(not(feature = "cpu-buffers"))]
            let _ = data;
            Ok(Value::Tensor(t))
        }
        Node::CallTensorConv2d {
            x,
            w,
            stride_h,
            stride_w,
            padding,
            ..
        } => {
            let x_val = eval_value_expr_mode(x, env, tensor_env, mode.clone())?;
            let w_val = eval_value_expr_mode(w, env, tensor_env, mode.clone())?;
            match (x_val, w_val) {
                (Value::Tensor(x_tensor), Value::Tensor(w_tensor)) => {
                    let result = stdlib::tensor::conv2d_tensor(
                        x_tensor, w_tensor, *stride_h, *stride_w, *padding, mode,
                    )?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallGrad { loss, wrt, .. } => eval_grad_map(loss, env, tensor_env, wrt),
        Node::Binary {
            op, left, right, ..
        } => {
            let lv = eval_value_expr_mode(left, env, tensor_env, mode.clone())?;
            let rv = eval_value_expr_mode(right, env, tensor_env, mode.clone())?;
            // issue #99: a `u64` operand selects UNSIGNED scalar semantics so the
            // interpreter matches the compiled artifact (`ult`/`divui`/…).
            if let (Value::Int(a), Value::Int(b)) = (&lv, &rv) {
                if interp_expr_is_u64(left) || interp_expr_is_u64(right) {
                    return apply_int_op_u64(*op, *a, *b).map(Value::Int);
                }
            }
            apply_binary(*op, lv, rv, mode.clone())
        }
        Node::Let { value, .. } | Node::Assign { value, .. } | Node::LetTuple { value, .. } => {
            eval_value_expr_mode(value, env, tensor_env, mode.clone())
        }
        // Function definitions and control flow - placeholder implementation
        Node::FnDef(..) => Ok(Value::Int(0)), // Functions are not executed as expressions
        // Salov C3 (#179): an early `return X` must STOP the enclosing function
        // body and yield `X`, matching the compiled/native path. We evaluate the
        // operand then raise the `ReturnFlow` control-flow signal; `?` carries it
        // up through any enclosing `Block`/`If`/`For`/`While` (stopping the loop)
        // to the `Node::Call` boundary, which unwraps it. A bare `return;` yields
        // the unit placeholder `Value::Int(0)`.
        Node::Return { value, .. } => {
            let v = if let Some(v) = value {
                eval_value_expr_mode(v, env, tensor_env, mode.clone())?
            } else {
                Value::Int(0)
            };
            Err(EvalError::ReturnFlow(Box::new(v)))
        }
        // Best-effort only: the const-fold evaluator has no `break`/`continue`
        // control-flow signal. Mid-iteration break/continue semantics are
        // honored solely by the MLIR codegen path.
        #[cfg(feature = "std-surface")]
        Node::Break { .. } | Node::Continue { .. } => Ok(Value::Int(0)),
        Node::Block { stmts, .. } => {
            let mut result = Value::Int(0);
            for stmt in stmts {
                result = eval_value_expr_mode(stmt, env, tensor_env, mode.clone())?;
            }
            Ok(result)
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let cond_val = eval_value_expr_mode(cond, env, tensor_env, mode.clone())?;
            match cond_val {
                Value::Int(0) => {
                    // False branch
                    if let Some(else_stmts) = else_branch {
                        let mut result = Value::Int(0);
                        for stmt in else_stmts {
                            result = eval_value_expr_mode(stmt, env, tensor_env, mode.clone())?;
                        }
                        Ok(result)
                    } else {
                        Ok(Value::Int(0))
                    }
                }
                _ => {
                    // True branch
                    let mut result = Value::Int(0);
                    for stmt in then_branch {
                        result = eval_value_expr_mode(stmt, env, tensor_env, mode.clone())?;
                    }
                    Ok(result)
                }
            }
        }
        // Import statements are module-level declarations, no runtime value
        Node::Import { .. } => Ok(Value::Int(0)),
        Node::ArrayLit { elements, .. } => {
            let mut vals = Vec::with_capacity(elements.len());
            for el in elements {
                vals.push(eval_value_expr_mode(el, env, tensor_env, mode.clone())?);
            }
            Ok(Value::Tuple(vals))
        }
        Node::ForEach { .. } => Err(EvalError::UnsupportedMsg(
            "for-each (`for x in coll`) is lowered by the native compiler over the \
             std.vec runtime; the tree-walking eval fallback does not interpret it"
                .into(),
        )),
        Node::MapLit { .. } => Err(EvalError::UnsupportedMsg(
            "map literal (`{}` / `{k: v}`) is lowered by the native compiler over the \
             std.map runtime; the tree-walking eval fallback does not interpret it"
                .into(),
        )),
        Node::SetLit { .. } => Err(EvalError::UnsupportedMsg(
            "set literal (`{a, b, c}`) is lowered by the native compiler over the \
             std.map runtime; the tree-walking eval fallback does not interpret it"
                .into(),
        )),
        Node::For {
            var,
            start,
            end,
            body,
            ..
        } => {
            let s = match eval_value_expr_mode(start, env, tensor_env, mode.clone())? {
                Value::Int(n) => n,
                _ => {
                    return Err(EvalError::UnsupportedMsg(
                        "for-loop start must be int".into(),
                    ));
                }
            };
            let e = match eval_value_expr_mode(end, env, tensor_env, mode.clone())? {
                Value::Int(n) => n,
                _ => return Err(EvalError::UnsupportedMsg("for-loop end must be int".into())),
            };
            let mut loop_env = env.clone();
            let mut result = Value::Int(0);
            for i in s..e {
                loop_env.insert(var.clone(), Value::Int(i));
                for stmt in body {
                    // Handle assignments: propagate back to loop_env
                    match stmt {
                        Node::Assign { name, value, .. } => {
                            let val =
                                eval_value_expr_mode(value, &loop_env, tensor_env, mode.clone())?;
                            loop_env.insert(name.clone(), val.clone());
                            result = val;
                        }
                        Node::Let { name, value, .. } => {
                            let val =
                                eval_value_expr_mode(value, &loop_env, tensor_env, mode.clone())?;
                            loop_env.insert(name.clone(), val.clone());
                            result = val;
                        }
                        _ => {
                            result =
                                eval_value_expr_mode(stmt, &loop_env, tensor_env, mode.clone())?;
                        }
                    }
                }
            }
            Ok(result)
        }
        Node::Print { args, .. } => {
            let mut values = Vec::new();
            for arg in args {
                values.push(eval_value_expr_mode(arg, env, tensor_env, mode.clone())?);
            }
            // Format string mode: first arg is a string containing "{}" placeholders
            if let Some(Value::Str(fmt)) = values.first() {
                if fmt.contains("{}") && values.len() > 1 {
                    let mut result = fmt.clone();
                    for val in &values[1..] {
                        if let Some(pos) = result.find("{}") {
                            let formatted = format_value_human(val);
                            result.replace_range(pos..pos + 2, &formatted);
                        }
                    }
                    eprintln!("{}", result);
                    return Ok(Value::Int(0));
                }
            }
            // Plain mode: print all args space-separated using human-readable format
            let parts: Vec<String> = values.iter().map(format_value_human).collect();
            eprintln!("{}", parts.join(" "));
            Ok(Value::Int(0))
        }
        Node::Neg { operand, .. } => match eval_value_expr_mode(operand, env, tensor_env, mode)? {
            Value::Int(n) => Ok(Value::Int(-n)),
            Value::Float(f) => Ok(Value::Float(-f)),
            other => Err(EvalError::UnsupportedMsg(format!(
                "cannot negate {:?}",
                other
            ))),
        },
        // Unary logical NOT `!expr`: truthy/falsy on i64 — `1` when the operand
        // is `0`, else `0` (enum_match #9). Mirrors the `operand == 0` desugar
        // the lowerer uses, keeping interpreter and codegen consistent.
        Node::Not { operand, .. } => match eval_value_expr_mode(operand, env, tensor_env, mode)? {
            Value::Int(n) => Ok(Value::Int((n == 0) as i64)),
            other => Err(EvalError::UnsupportedMsg(format!(
                "cannot apply `!` to {:?}",
                other
            ))),
        },
        Node::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            let recv = eval_value_expr_mode(receiver, env, tensor_env, mode.clone())?;
            let mut eval_args = Vec::with_capacity(args.len());
            for arg in args {
                eval_args.push(eval_value_expr_mode(arg, env, tensor_env, mode.clone())?);
            }
            eval_method_call(recv, method, &eval_args)
        }
        Node::FieldAccess {
            receiver, field, ..
        } => {
            let recv = eval_value_expr_mode(receiver, env, tensor_env, mode)?;
            eval_field_access(recv, field)
        }
        // Phase 10.5 declarations are statement-level, not expression-level.
        // Top-level walkers handle them; reaching this arm in expression
        // evaluation means a misuse — return a zero so we don't crash, the
        // type checker catches the actual misuse.
        Node::Const { .. }
        | Node::ExternConst { .. }
        | Node::TypeAlias { .. }
        | Node::Export { .. }
        | Node::StructDef { .. }
        | Node::EnumDef { .. } => Ok(Value::Int(0)),
        // Phase 10.5 stretch: assert is a no-op at eval-time in the
        // preview path; runtime checking is a future-extension item.
        Node::Assert { .. } => Ok(Value::Int(0)),
        // `expr as type` — evaluate the operand, then apply the SAME narrowing
        // the codegen path (`lower.rs` `Node::As`) emits, so the interpreter and
        // the runnable artifact agree. Without this the cast was a transparent
        // no-op: `300 as u8` yielded the raw `300` while the compiled `.so`
        // masks it to `44`, and any downstream comparison (`(300 as u8) == 44`)
        // picked the WRONG branch in the interpreter — a silent miscompile.
        //
        //   * narrow SIGNED (`i8`/`i16`/`i32`) → truncate + sign-extend, exactly
        //     the `(x << (64-W)) >> (64-W)` shift-pair codegen uses.
        //   * narrow UNSIGNED (`u8`/`u16`/`u32`) → zero-extend via `x & ((1<<W)-1)`.
        //   * int → float (`f32`/`f64`) → IEEE round-to-nearest-even (`n as f64`
        //     is exactly that), mirroring the `sitofp`/`uitofp` lowering. A bool
        //     is `Int(0/1)` so `true as f64 == 1.0` (the compiled path treats i1
        //     as unsigned for the same reason).
        //   * float → int → SATURATING truncate-toward-zero (`f as iN`/`uN` in
        //     Rust is saturating: NaN→0, ±ovf→MIN/MAX), the EXACT mirror of the
        //     compiled `emit_saturating_fp_to_i64` — so the interpreter (the tier
        //     `mind-runtime` executes) and the runnable artifact agree bit-for-bit
        //     on every substrate. Without this the cast was a silent drop here.
        //   * float → float (`f64`→`f32` rounds; `f32`→`f64` widens); other/`i64`
        //     pass through unchanged so the keystone stays byte-identical.
        Node::As { expr, ty, .. } => {
            let inner = eval_value_expr_mode(expr, env, tensor_env, mode)?;
            match inner {
                Value::Int(n) => match float_cast_target(ty) {
                    // int → float: a `u64` source converts UNSIGNED, mirroring the
                    // compiled `arith.uitofp` (issue #99/#105). A signed `n as f64`
                    // would give the wrong sign for a u64 ≥ 2^63 (`1u64<<63` →
                    // −9.22e18 in the interpreter vs +9.22e18 in the artifact) — an
                    // interp-vs-artifact divergence. Non-u64 ints stay signed.
                    Some(32) if interp_expr_is_u64(expr) => {
                        Ok(Value::Float((n as u64) as f32 as f64))
                    }
                    Some(32) => Ok(Value::Float(n as f32 as f64)),
                    Some(_) if interp_expr_is_u64(expr) => Ok(Value::Float((n as u64) as f64)),
                    Some(_) => Ok(Value::Float(n as f64)),
                    // int → int: existing narrowing (or i64 identity).
                    None => Ok(Value::Int(apply_scalar_cast(n, ty))),
                },
                Value::Float(f) => {
                    if let Some(bits) = float_to_int_cast_bits(f, ty) {
                        // float → integer: saturating, mirrors the lowering.
                        Ok(Value::Int(bits))
                    } else {
                        match float_cast_target(ty) {
                            // float → f32: round to single precision.
                            Some(32) => Ok(Value::Float(f as f32 as f64)),
                            // float → f64 (or unknown): unchanged.
                            _ => Ok(Value::Float(f)),
                        }
                    }
                }
                // Tensors / enums keep the prior transparent behaviour — the cast
                // checks live in typecheck.
                other => Ok(other),
            }
        }
        // `a && b`, `a || b` — short-circuit boolean evaluation in i32.
        Node::Logical {
            op, left, right, ..
        } => {
            let l = eval_value_expr_mode(left, env, tensor_env, mode.clone())?;
            let l_truthy = matches!(l, Value::Int(n) if n != 0);
            match op {
                crate::ast::LogicalOp::And => {
                    if !l_truthy {
                        return Ok(Value::Int(0));
                    }
                }
                crate::ast::LogicalOp::Or => {
                    if l_truthy {
                        return Ok(Value::Int(1));
                    }
                }
            }
            let r = eval_value_expr_mode(right, env, tensor_env, mode)?;
            Ok(Value::Int(matches!(r, Value::Int(n) if n != 0) as i64))
        }
        // Bitwise: integer-only at preview level.
        Node::Bitwise {
            op, left, right, ..
        } => {
            let l = eval_value_expr_mode(left, env, tensor_env, mode.clone())?;
            let r = eval_value_expr_mode(right, env, tensor_env, mode)?;
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => {
                    // issue #99: `u64 >> n` is a LOGICAL shift (mask to width-1,
                    // zero-fill) to match the compiled `arith.shrui`; a signed i64
                    // `>>` stays ARITHMETIC. `<< & | ^` are sign-agnostic.
                    let u64_shr = matches!(op, crate::ast::BitOp::Shr)
                        && (interp_expr_is_u64(left) || interp_expr_is_u64(right));
                    let result = match op {
                        crate::ast::BitOp::Or => a | b,
                        crate::ast::BitOp::And => a & b,
                        crate::ast::BitOp::Xor => a ^ b,
                        crate::ast::BitOp::Shl => a.wrapping_shl(b as u32),
                        crate::ast::BitOp::Shr if u64_shr => {
                            (a as u64).wrapping_shr(b as u32) as i64
                        }
                        crate::ast::BitOp::Shr => a.wrapping_shr(b as u32),
                    };
                    Ok(Value::Int(result))
                }
                _ => Ok(Value::Int(0)),
            }
        }
        // Phase 10.6: struct literal preview-eval. Evaluate each field's
        // value sub-expression and pack them into a Tuple in declared
        // field order. Full structural eval (returning a typed aggregate
        // tied to the struct name) lands when AOT codegen needs it.
        Node::StructLit { fields, .. } => {
            let mut items = Vec::with_capacity(fields.len());
            for f in fields {
                items.push(eval_value_expr_mode(
                    &f.value,
                    env,
                    tensor_env,
                    mode.clone(),
                )?);
            }
            Ok(Value::Tuple(items))
        }
        // Phase 10.6: index access `receiver[index]`. For an array/tuple value
        // (what `[a, b, c]` literals evaluate to) the interpreter returns the
        // indexed element with bounds checking. Tensor/slice indexing still
        // needs the runtime handle-table threaded through eval, so a non-tuple
        // receiver preserves the previous receiver-placeholder behavior.
        Node::IndexAccess {
            receiver, index, ..
        } => {
            let recv = eval_value_expr_mode(receiver, env, tensor_env, mode.clone())?;
            match recv {
                Value::Tuple(items) => {
                    let idx = match eval_value_expr_mode(index, env, tensor_env, mode)? {
                        Value::Int(i) => i,
                        other => {
                            return Err(EvalError::UnsupportedMsg(format!(
                                "array index must be an integer, got {other:?}"
                            )));
                        }
                    };
                    if idx < 0 || idx as usize >= items.len() {
                        return Err(EvalError::UnsupportedMsg(format!(
                            "array index {idx} out of bounds (len {})",
                            items.len()
                        )));
                    }
                    Ok(items[idx as usize].clone())
                }
                other => Ok(other),
            }
        }
        // Phase 10.6: indexed assignment is a statement, not an
        // expression. Evaluating it as an expression returns the
        // assigned value (matches C-style semantics).
        Node::IndexAssign { value, .. } => eval_value_expr_mode(value, env, tensor_env, mode),
        // Phase 10.6: field assignment is also a statement; the
        // expression-position evaluation returns the assigned value.
        Node::FieldAssign { value, .. } => eval_value_expr_mode(value, env, tensor_env, mode),
        // Phase 10.7: `match`. Evaluate the scrutinee and take the first arm
        // whose pattern matches — `Wildcard`, `Ident` (binds the value), or
        // `Literal` (Int/Float/Str equality). Enum-variant patterns need a
        // sum-type value model the interpreter does not have yet, so they can
        // never match a scalar value and fall through to a `_`/ident arm.
        Node::Match {
            scrutinee, arms, ..
        } => {
            let val = eval_value_expr_mode(scrutinee, env, tensor_env, mode.clone())?;
            for arm in arms {
                // `bindings` carries the names introduced by this arm and the
                // values they bind to (a simple ident binding binds the whole
                // scrutinee; enum-variant sub-patterns bind payload elements).
                let mut bindings: Vec<(String, Value)> = Vec::new();
                let is_match = match &arm.pattern {
                    crate::ast::Pattern::Wildcard => true,
                    crate::ast::Pattern::Ident(name) => {
                        bindings.push((name.clone(), val.clone()));
                        true
                    }
                    crate::ast::Pattern::Literal(lit) => match (lit, &val) {
                        (crate::ast::Literal::Int(m), Value::Int(n)) => n == m,
                        (crate::ast::Literal::Float(a), Value::Float(b)) => a == b,
                        (crate::ast::Literal::Str(s), Value::Str(v)) => v == s,
                        _ => false,
                    },
                    // Phase 10.7: enum-variant pattern. Matches when the
                    // scrutinee is a `Value::Enum` whose variant path equals the
                    // pattern path and whose payload arity matches the
                    // sub-pattern count, recursively binding each sub-pattern.
                    crate::ast::Pattern::EnumVariant { path, args } => match &val {
                        Value::Enum { variant, payload }
                            if variant == path && payload.len() == args.len() =>
                        {
                            match_enum_payload(args, payload, &mut bindings)
                        }
                        _ => false,
                    },
                    // Tuple pattern `(a, b)`: matches a tuple scrutinee of equal
                    // arity, binding each element (enum_match #9).
                    crate::ast::Pattern::Tuple(elems) => match &val {
                        Value::Tuple(items) if items.len() == elems.len() => {
                            match_enum_payload(elems, items, &mut bindings)
                        }
                        _ => false,
                    },
                    // Struct-variant pattern `E.V { f, g }` — const-eval best
                    // effort: bind the named fields' sub-patterns positionally in
                    // source order against the enum payload (enum_match #9).
                    crate::ast::Pattern::EnumStruct { fields, .. } => match &val {
                        Value::Enum { payload, .. } if payload.len() >= fields.len() => {
                            let subs: Vec<crate::ast::Pattern> =
                                fields.iter().map(|(_, p)| p.clone()).collect();
                            match_enum_payload(&subs, payload, &mut bindings)
                        }
                        _ => false,
                    },
                };
                if is_match {
                    let mut arm_env = env.clone();
                    for (name, bound) in bindings {
                        arm_env.insert(name, bound);
                    }
                    // Pattern-guards W1.5a: a guarded arm matches only when its
                    // guard (evaluated with the pattern's bindings in scope) is
                    // truthy; a false guard falls through to the next arm.
                    if let Some(guard) = &arm.guard {
                        let g = eval_value_expr_mode(guard, &arm_env, tensor_env, mode.clone())?;
                        if matches!(g, Value::Int(0)) {
                            continue;
                        }
                    }
                    return eval_value_expr_mode(&arm.body, &arm_env, tensor_env, mode);
                }
            }
            Err(EvalError::UnsupportedMsg(
                "no match arm matched (non-exhaustive match)".into(),
            ))
        }
        // Phase 10.7: reference-taking `&expr` / `&mut expr`. The immutable
        // interpreter has no heap/handle model, so a borrow in expression
        // position is simply a READ of the inner expression: `&x` (and
        // `&mut x`) evaluates the inner expression and yields its value.
        Node::Ref { inner, .. } => eval_value_expr_mode(inner, env, tensor_env, mode.clone()),
        // W1.5f: postfix `?`. FAIL-CLOSED — a `?`-using fn is excluded from
        // compile-time tree-walking evaluation (const-eval / #[collapse]) this
        // slice. Bailing here (rather than folding a wrong value) makes any
        // attempt to const-fold a `?`-using fn a clean, loud error; the native
        // lowering path handles `?` correctly.
        // deferred: const-eval of `?` — upgrade path: evaluate `inner`, then on
        // the Ok/Some tag yield the payload and on the Err/None tag raise the
        // early-return via the #179 ReturnFlow signal the tree evaluator now
        // carries, mirroring the native `build_try_desugar` lowering.
        Node::Try { .. } => Err(EvalError::UnsupportedMsg(
            "postfix `?` is lowered by the native compiler into the `match` \
             error-propagation machinery; the tree-walking eval fallback does \
             not interpret it (a `?`-using fn is excluded from const-eval)"
                .into(),
        )),
        // RFC 0005 Gap 1: while loop. Mirrors the `For`-loop interpreter
        // contract — a fresh loop scope inheriting the outer env, with
        // `let`/`assign` statements in the body propagated back into that scope
        // so the loop condition can make progress. Truthiness matches `If`:
        // `Value::Int(0)` is false, anything else is true. A hard iteration cap
        // makes a non-terminating loop fail loudly: this runs at COMPILE-TIME
        // evaluation, so `while 1 { }` must error, not hang the compiler.
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            const MAX_EVAL_ITERS: u64 = 1_000_000;
            let mut loop_env = env.clone();
            let mut result = Value::Int(0);
            let mut iters: u64 = 0;
            loop {
                let cond_val = eval_value_expr_mode(cond, &loop_env, tensor_env, mode.clone())?;
                if matches!(cond_val, Value::Int(0)) {
                    break;
                }
                iters += 1;
                if iters > MAX_EVAL_ITERS {
                    return Err(EvalError::UnsupportedMsg(
                        "`while` exceeded the compile-time evaluation iteration cap \
                         (possible non-terminating loop)"
                            .into(),
                    ));
                }
                for stmt in body {
                    // Propagate `let`/`assign` into the loop scope so the next
                    // condition check sees the update (same handling as `For`).
                    match stmt {
                        Node::Assign { name, value, .. } | Node::Let { name, value, .. } => {
                            let val =
                                eval_value_expr_mode(value, &loop_env, tensor_env, mode.clone())?;
                            loop_env.insert(name.clone(), val.clone());
                            result = val;
                        }
                        // `arr[i] = v` inside the while body: rebuild + rebind the
                        // array tuple in the loop scope (same as For / module level).
                        Node::IndexAssign {
                            receiver,
                            index,
                            value,
                            ..
                        } => {
                            let val =
                                eval_value_expr_mode(value, &loop_env, tensor_env, mode.clone())?;
                            if let Node::Lit(Literal::Ident(arr), _) = receiver.as_ref() {
                                if let Some(Value::Tuple(mut items)) = loop_env.get(arr).cloned() {
                                    let idx = match eval_value_expr_mode(
                                        index,
                                        &loop_env,
                                        tensor_env,
                                        mode.clone(),
                                    )? {
                                        Value::Int(i) => i,
                                        other => {
                                            return Err(EvalError::UnsupportedMsg(format!(
                                                "array index must be an integer, got {other:?}"
                                            )));
                                        }
                                    };
                                    if idx < 0 || idx as usize >= items.len() {
                                        return Err(EvalError::UnsupportedMsg(format!(
                                            "array index {idx} out of bounds (len {})",
                                            items.len()
                                        )));
                                    }
                                    items[idx as usize] = val.clone();
                                    loop_env.insert(arr.clone(), Value::Tuple(items));
                                }
                            }
                            result = val;
                        }
                        _ => {
                            result =
                                eval_value_expr_mode(stmt, &loop_env, tensor_env, mode.clone())?;
                        }
                    }
                }
            }
            Ok(result)
        }
        // RFC 0010 Phase A: `extern "C"` blocks are declarations, not
        // expressions. The interpreter returns a unit placeholder; the
        // actual call site is handled by the MLIR lowering path.
        Node::ExternBlock { .. } => Ok(Value::Int(0)),
        // RFC 0010 Phase J-A: `region { ... }` block.
        //
        // The interpreter evaluates the body in a local scope that inherits
        // the outer environment and propagates `let`/`assign` bindings within
        // the region. Returns the last expression's value. The interpreter
        // does not call `__mind_region_enter` / `__mind_region_exit` (it
        // manages no real heap) but maintains semantic parity with the
        // codegen path: a region evaluates to its last expression.
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            let mut region_env = env.clone();
            let mut result = Value::Int(0);
            for stmt in body {
                match stmt {
                    Node::Let { name, value, .. } => {
                        let val =
                            eval_value_expr_mode(value, &region_env, tensor_env, mode.clone())?;
                        region_env.insert(name.clone(), val.clone());
                        result = val;
                    }
                    Node::Assign { name, value, .. } => {
                        let val =
                            eval_value_expr_mode(value, &region_env, tensor_env, mode.clone())?;
                        region_env.insert(name.clone(), val.clone());
                        result = val;
                    }
                    // `arr[i] = v` inside a region block: rebuild + rebind the array
                    // tuple in the region scope (same as for / while / module level).
                    Node::IndexAssign {
                        receiver,
                        index,
                        value,
                        ..
                    } => {
                        let val =
                            eval_value_expr_mode(value, &region_env, tensor_env, mode.clone())?;
                        if let Node::Lit(Literal::Ident(arr), _) = receiver.as_ref() {
                            if let Some(Value::Tuple(mut items)) = region_env.get(arr).cloned() {
                                let idx = match eval_value_expr_mode(
                                    index,
                                    &region_env,
                                    tensor_env,
                                    mode.clone(),
                                )? {
                                    Value::Int(i) => i,
                                    other => {
                                        return Err(EvalError::UnsupportedMsg(format!(
                                            "array index must be an integer, got {other:?}"
                                        )));
                                    }
                                };
                                if idx < 0 || idx as usize >= items.len() {
                                    return Err(EvalError::UnsupportedMsg(format!(
                                        "array index {idx} out of bounds (len {})",
                                        items.len()
                                    )));
                                }
                                items[idx as usize] = val.clone();
                                region_env.insert(arr.clone(), Value::Tuple(items));
                            }
                        }
                        result = val;
                    }
                    _ => {
                        result = eval_value_expr_mode(stmt, &region_env, tensor_env, mode.clone())?;
                    }
                }
            }
            Ok(result)
        }
    }
}

fn eval_method_call(receiver: Value, method: &str, _args: &[Value]) -> Result<Value, EvalError> {
    match method {
        "len" => match &receiver {
            Value::Tuple(items) => Ok(Value::Int(items.len() as i64)),
            Value::Str(s) => Ok(Value::Int(s.len() as i64)),
            Value::Tensor(t) => {
                let mut total: i64 = 1;
                for d in &t.shape {
                    match d {
                        ShapeDim::Known(n) => total *= *n as i64,
                        ShapeDim::Sym(_) => {
                            return Err(EvalError::UnsupportedMsg(
                                "cannot compute .len() on tensor with symbolic dimensions".into(),
                            ));
                        }
                    }
                }
                Ok(Value::Int(total))
            }
            other => Err(EvalError::UnsupportedMsg(format!(
                "no method .len() for {:?}",
                other
            ))),
        },
        "shape" => match &receiver {
            Value::Tensor(t) => {
                let mut dims = Vec::with_capacity(t.shape.len());
                for d in &t.shape {
                    match d {
                        ShapeDim::Known(n) => dims.push(Value::Int(*n as i64)),
                        ShapeDim::Sym(_) => {
                            return Err(EvalError::UnsupportedMsg(
                                "cannot return .shape() on a tensor with symbolic dimensions"
                                    .into(),
                            ));
                        }
                    }
                }
                Ok(Value::Tuple(dims))
            }
            other => Err(EvalError::UnsupportedMsg(format!(
                "no method .shape() for {:?}",
                other
            ))),
        },
        "last" => match &receiver {
            Value::Tuple(items) => {
                if items.is_empty() {
                    Err(EvalError::OutOfBounds)
                } else {
                    Ok(items.last().unwrap().clone())
                }
            }
            other => Err(EvalError::UnsupportedMsg(format!(
                "no method .last() for {:?}",
                other
            ))),
        },
        "clone" => Ok(receiver),
        "item" => match &receiver {
            Value::Tensor(t) => {
                let total = t.shape.iter().try_fold(1usize, |acc, d| match d {
                    ShapeDim::Known(n) => Some(acc * n),
                    ShapeDim::Sym(_) => None,
                });
                match total {
                    Some(1) => {
                        if let Some(fill) = t.fill {
                            Ok(Value::Float(fill))
                        } else {
                            #[cfg(feature = "cpu-buffers")]
                            {
                                match &t.buf {
                                    Some(value::Buffer::F32(data)) if data.len() == 1 => {
                                        Ok(Value::Float(data[0] as f64))
                                    }
                                    Some(value::Buffer::I32(data)) if data.len() == 1 => {
                                        Ok(Value::Int(data[0] as i64))
                                    }
                                    _ => Err(EvalError::UnsupportedMsg(
                                        ".item() requires a materialized single-element tensor"
                                            .into(),
                                    )),
                                }
                            }
                            #[cfg(not(feature = "cpu-buffers"))]
                            Err(EvalError::UnsupportedMsg(
                                ".item() requires a tensor with a known fill value".into(),
                            ))
                        }
                    }
                    Some(n) => Err(EvalError::UnsupportedMsg(format!(
                        ".item() requires a scalar tensor, got {} elements",
                        n
                    ))),
                    None => Err(EvalError::UnsupportedMsg(
                        ".item() cannot be called on tensor with symbolic dims".into(),
                    )),
                }
            }
            other => Err(EvalError::UnsupportedMsg(format!(
                "no method .item() for {:?}",
                other
            ))),
        },
        _ => Err(EvalError::UnsupportedMsg(format!(
            "unknown method .{}()",
            method
        ))),
    }
}

fn eval_field_access(receiver: Value, field: &str) -> Result<Value, EvalError> {
    match field {
        "len" => match &receiver {
            Value::Tuple(items) => Ok(Value::Int(items.len() as i64)),
            Value::Str(s) => Ok(Value::Int(s.len() as i64)),
            _ => Err(EvalError::UnsupportedMsg(format!(
                "no field .len for {:?}",
                receiver
            ))),
        },
        _ => Err(EvalError::UnsupportedMsg(format!(
            "unknown field .{}",
            field
        ))),
    }
}

pub fn eval_grad_map(
    loss_expr: &Node,
    _env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    wrt: &[String],
) -> Result<Value, EvalError> {
    let mut tenv: HashMap<String, TensorEnvEntry> = HashMap::new();
    for (name, entry) in tensor_env {
        tenv.insert(name.clone(), entry.clone());
    }

    let mut expanding = BTreeSet::new();
    let (loss_id, tape, vars_all) =
        crate::eval::autodiff::build_graph_loss(loss_expr, &tenv, &mut expanding)
            .map_err(EvalError::UnsupportedMsg)?;

    if !tape.node_shape(loss_id).is_empty() {
        return Err(EvalError::UnsupportedMsg(
            "grad() expects the loss expression to have shape ()".to_string(),
        ));
    }

    let requested: BTreeMap<String, crate::eval::autodiff::NodeId> = vars_all
        .into_iter()
        .filter(|(name, _)| wrt.contains(name))
        .collect();

    let mut grads =
        crate::eval::autodiff::backprop_to_vars_with_tenv(loss_id, &tape, &requested, &tenv);
    for name in wrt {
        if !grads.contains_key(name) {
            if let Some(entry) = tenv.get(name) {
                grads.insert(
                    name.clone(),
                    TensorVal::new(
                        entry.value.dtype.clone(),
                        entry.value.shape.clone(),
                        Some(0.0),
                    ),
                );
            }
        }
    }

    let mut out = BTreeMap::new();
    for name in wrt {
        if let Some(tensor) = grads.get(name) {
            out.insert(VarId(name.clone()), tensor.clone());
        }
    }

    Ok(Value::GradMap(out))
}

fn apply_float_op(op: BinOp, l: f64, r: f64) -> Result<f64, EvalError> {
    Ok(match op {
        BinOp::Add => l + r,
        BinOp::Sub => l - r,
        BinOp::Mul => l * r,
        BinOp::Div => {
            if r == 0.0 {
                return Err(EvalError::DivZero);
            }
            l / r
        }
        BinOp::Mod => {
            if r == 0.0 {
                return Err(EvalError::DivZero);
            }
            l % r
        }
        BinOp::Lt => {
            if l < r {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Le => {
            if l <= r {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Gt => {
            if l > r {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Ge => {
            if l >= r {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Eq => {
            if l == r {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Ne => {
            if l != r {
                1.0
            } else {
                0.0
            }
        }
    })
}

fn apply_binary(op: BinOp, left: Value, right: Value, mode: ExecMode) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => apply_int_op(op, l, r).map(Value::Int),
        (Value::Float(l), Value::Float(r)) => apply_float_op(op, l, r).map(Value::Float),
        (Value::Float(l), Value::Int(r)) => apply_float_op(op, l, r as f64).map(Value::Float),
        (Value::Int(l), Value::Float(r)) => apply_float_op(op, l as f64, r).map(Value::Float),
        (Value::Str(s), Value::Int(n)) => match op {
            BinOp::Mul => Ok(Value::Str(s.repeat(n.max(0) as usize))),
            _ => Err(EvalError::Unsupported),
        },
        (Value::Tensor(t), Value::Int(s)) => apply_tensor_scalar(op, t, s as f64, true, mode),
        (Value::Int(s), Value::Tensor(t)) => apply_tensor_scalar(op, t, s as f64, false, mode),
        (Value::Tensor(t), Value::Float(s)) => apply_tensor_scalar(op, t, s, true, mode),
        (Value::Float(s), Value::Tensor(t)) => apply_tensor_scalar(op, t, s, false, mode),
        (Value::Tensor(a), Value::Tensor(b)) => apply_tensor_tensor(op, a, b, mode),
        _ => Err(EvalError::Unsupported),
    }
}

fn apply_int_op(op: BinOp, left: i64, right: i64) -> Result<i64, EvalError> {
    Ok(match op {
        // MIND integer overflow = defined two's-complement wraparound (== the MLIR
        // artifact's `arith.addi`, no nsw/nuw); use explicit wrapping so debug and
        // release mindc agree with the shipped .so and never panic on overflow.
        BinOp::Add => left.wrapping_add(right),
        BinOp::Sub => left.wrapping_sub(right),
        BinOp::Mul => left.wrapping_mul(right),
        BinOp::Div => {
            if right == 0 {
                return Err(EvalError::DivZero);
            }
            left / right
        }
        BinOp::Mod => {
            if right == 0 {
                return Err(EvalError::DivZero);
            }
            left % right
        }
        BinOp::Lt => (left < right) as i64,
        BinOp::Le => (left <= right) as i64,
        BinOp::Gt => (left > right) as i64,
        BinOp::Ge => (left >= right) as i64,
        BinOp::Eq => (left == right) as i64,
        BinOp::Ne => (left != right) as i64,
    })
}

/// issue #99 — UNSIGNED (`u64`) scalar op, the interpreter mirror of the
/// compiled plain-i64 arm's `ScalarU64` path. Operands are the raw i64 bit
/// patterns; they are reinterpreted as `u64` for the sign-sensitive ops
/// (`/ % < <= > >=`), so a value with the high bit set behaves as an unsigned
/// magnitude. Division / remainder by zero is the SAME deterministic total
/// contract the codegen emits (`x/0 == 0`, `x%0 == 0`) rather than a trap — so
/// interp == artifact. `+ − × == !=` are bit-identical to the signed path (they
/// never inspect the operand as an ordered value), so they defer to
/// `apply_int_op` for exact parity.
fn apply_int_op_u64(op: BinOp, left: i64, right: i64) -> Result<i64, EvalError> {
    let a = left as u64;
    let b = right as u64;
    Ok(match op {
        // `x/0 == 0` / `x%0 == 0` — the deterministic total-division contract the
        // codegen emits (no trap). `checked_div`/`checked_rem` yield `None` on a
        // zero divisor, mapped to 0.
        BinOp::Div => a.checked_div(b).unwrap_or(0) as i64,
        BinOp::Mod => a.checked_rem(b).unwrap_or(0) as i64,
        BinOp::Lt => (a < b) as i64,
        BinOp::Le => (a <= b) as i64,
        BinOp::Gt => (a > b) as i64,
        BinOp::Ge => (a >= b) as i64,
        // Sign-agnostic — identical to the signed path.
        _ => return apply_int_op(op, left, right),
    })
}

/// Floating-point width of a scalar `as` target type (`f32`/`f64`), else `None`.
/// Mirror of `lower.rs::scalar_float_cast_width` for the interpreter tier.
fn float_cast_target(ty: &TypeAnn) -> Option<u32> {
    match ty {
        TypeAnn::ScalarF32 => Some(32),
        TypeAnn::ScalarF64 => Some(64),
        TypeAnn::Named(name) => match name.as_str() {
            "f32" => Some(32),
            "f64" => Some(64),
            _ => None,
        },
        _ => None,
    }
}

/// Interpreter-tier float→integer `as` cast — the EXACT mirror of the compiled
/// lowering (`emit_saturating_fp_to_i64` + the narrowing branches), so the
/// tree-walking evaluator (the path `mind-runtime` executes) and the runnable
/// artifact agree bit-for-bit. Returns the target value as an i64 bit pattern,
/// or `None` for a non-integer / unsupported target (caller keeps the float).
///
/// Rust's `f as T` is a SATURATING truncate-toward-zero conversion (NaN→0,
/// ±overflow→`T::MIN`/`T::MAX`) — which is precisely the saturating semantics the
/// compiled path now emits, so `f as i64` here == the lowered result on x86 and
/// ARM. Narrow targets saturate to the TARGET range then sign/zero-extend to the
/// i64 slot, matching `f as iN`/`uN` exactly.
fn float_to_int_cast_bits(f: f64, ty: &TypeAnn) -> Option<i64> {
    let name = match ty {
        TypeAnn::ScalarI32 => "i32",
        TypeAnn::ScalarI64 => "i64",
        TypeAnn::ScalarU32 => "u32",
        TypeAnn::Named(n) => n.as_str(),
        _ => return None,
    };
    Some(match name {
        "i8" => f as i8 as i64,
        "i16" => f as i16 as i64,
        "i32" => f as i32 as i64,
        "i64" => f as i64,
        "u8" => f as u8 as i64,
        "u16" => f as u16 as i64,
        "u32" => f as u32 as i64,
        "u64" => f as u64 as i64,
        _ => return None,
    })
}

/// Apply a scalar `as`-cast to an i64-carried interpreter value, matching the
/// codegen path (`src/eval/lower.rs` `Node::As` + `scalar_int_cast_width` /
/// `scalar_uint_cast_width`) bit-for-bit so the interpreter never diverges from
/// the runnable artifact.
///
///   * SIGNED narrow (`i8`/`i16`/`i32`, also `TypeAnn::ScalarI32`): truncate to
///     the low `W` bits then sign-extend — `(n << (64-W)) >> (64-W)` with an
///     arithmetic right shift, exactly the codegen shift pair.
///   * UNSIGNED narrow (`u8`/`u16`/`u32`, also `TypeAnn::ScalarU32`): zero-extend
///     by masking the low `W` bits — `n & ((1<<W)-1)`.
///   * `i64`/`u64`/floats/pointers/aliases / any non-narrowing target: returned
///     unchanged (the transparent codegen fall-through).
fn apply_scalar_cast(n: i64, ty: &TypeAnn) -> i64 {
    // Signed narrowing widths (mirror of lower.rs::scalar_int_cast_width).
    let signed_width = match ty {
        TypeAnn::ScalarI32 => Some(32u32),
        TypeAnn::ScalarI64 => Some(64),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" => Some(8),
            "i16" => Some(16),
            "i32" => Some(32),
            "i64" => Some(64),
            _ => None,
        },
        _ => None,
    };
    if let Some(width) = signed_width {
        if width < 64 {
            let shift = 64 - width as i64;
            return (n << shift) >> shift;
        }
        return n;
    }
    // Unsigned narrowing widths (mirror of lower.rs::scalar_uint_cast_width).
    let unsigned_width = match ty {
        TypeAnn::ScalarU32 => Some(32u32),
        TypeAnn::Named(name) => match name.as_str() {
            "u8" => Some(8),
            "u16" => Some(16),
            "u32" => Some(32),
            _ => None,
        },
        _ => None,
    };
    if let Some(width) = unsigned_width {
        if width < 64 {
            let mask: i64 = if width == 32 {
                0xFFFF_FFFF
            } else {
                (1i64 << width) - 1
            };
            return n & mask;
        }
    }
    n
}

fn apply_tensor_scalar(
    op: BinOp,
    tensor: TensorVal,
    scalar: f64,
    tensor_on_left: bool,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if matches!(op, BinOp::Div) && tensor_on_left && scalar == 0.0 {
        return Err(EvalError::DivZero);
    }

    #[cfg(not(feature = "cpu-buffers"))]
    let _ = mode;

    #[cfg(feature = "cpu-buffers")]
    {
        if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) {
            let mut tensor_exec = tensor.clone();
            materialize_filled(&mut tensor_exec);
            #[cfg(feature = "cpu-exec")]
            {
                // TODO(runtime): dispatch through `Evaluator::runtime` once the runtime plumbing is
                // threaded into evaluation.
                if tensor_exec.dtype == DType::F32 {
                    let exec_res = match op {
                        BinOp::Add => exec::cpu::exec_add_scalar(&tensor_exec, scalar as f32),
                        BinOp::Sub => {
                            if tensor_on_left {
                                exec::cpu::exec_sub_scalar(&tensor_exec, scalar as f32)
                            } else {
                                exec::cpu::exec_scalar_sub(scalar as f32, &tensor_exec)
                            }
                        }
                        BinOp::Mul => exec::cpu::exec_mul_scalar(&tensor_exec, scalar as f32),
                        BinOp::Div => {
                            exec::cpu::exec_div_scalar(&tensor_exec, scalar as f32, tensor_on_left)
                        }
                        _ => {
                            return Ok(Value::Tensor(TensorVal::new(
                                tensor.dtype.clone(),
                                tensor.shape.clone(),
                                Some(0.0),
                            )));
                        }
                    };
                    match exec_res {
                        Ok(t) => return Ok(Value::Tensor(t)),
                        Err(err) => {
                            let mapped = exec_error_to_eval(err);
                            if matches!(mapped, EvalError::DivZero) {
                                return Err(mapped);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "cpu-buffers")]
    let tensor_buf = tensor.buf.clone();

    let dtype = tensor.dtype.clone();
    let shape = tensor.shape.clone();
    let fill = tensor.fill;

    let result_fill = match fill {
        Some(f) => {
            if matches!(op, BinOp::Div) && !tensor_on_left && f == 0.0 {
                return Err(EvalError::DivZero);
            }
            Some(match op {
                BinOp::Add => f + scalar,
                BinOp::Sub => {
                    if tensor_on_left {
                        f - scalar
                    } else {
                        scalar - f
                    }
                }
                BinOp::Mul => f * scalar,
                BinOp::Div => {
                    if tensor_on_left {
                        f / scalar
                    } else {
                        scalar / f
                    }
                }
                BinOp::Mod => {
                    if tensor_on_left {
                        f % scalar
                    } else {
                        scalar % f
                    }
                }
                BinOp::Lt => {
                    if (tensor_on_left && f < scalar) || (!tensor_on_left && scalar < f) {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Le => {
                    if (tensor_on_left && f <= scalar) || (!tensor_on_left && scalar <= f) {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Gt => {
                    if (tensor_on_left && f > scalar) || (!tensor_on_left && scalar > f) {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Ge => {
                    if (tensor_on_left && f >= scalar) || (!tensor_on_left && scalar >= f) {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Eq => {
                    if f == scalar {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Ne => {
                    if f != scalar {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
        }
        None => None,
    };

    #[cfg_attr(not(feature = "cpu-buffers"), allow(unused_mut))]
    let mut result = TensorVal::new(dtype.clone(), shape, result_fill);

    #[cfg(feature = "cpu-buffers")]
    {
        if let Some(buf) = tensor_buf.as_ref() {
            match (buf, &dtype) {
                (Buffer::I32(values), DType::I32) => {
                    if matches!(op, BinOp::Div) && !tensor_on_left && values.contains(&0) {
                        return Err(EvalError::DivZero);
                    }
                    let scalar_i32 = scalar as i32;
                    let mut out = Vec::with_capacity(values.len());
                    for &v in values {
                        let computed = match op {
                            BinOp::Add => v + scalar_i32,
                            BinOp::Sub => {
                                if tensor_on_left {
                                    v - scalar_i32
                                } else {
                                    scalar_i32 - v
                                }
                            }
                            BinOp::Mul => v * scalar_i32,
                            BinOp::Div => {
                                if tensor_on_left {
                                    v / scalar_i32
                                } else {
                                    scalar_i32 / v
                                }
                            }
                            _ => 0, // comparison ops
                        };
                        out.push(computed);
                    }
                    result.buf = Some(Buffer::I32(out));
                }
                (Buffer::F32(values), DType::F32) => {
                    if matches!(op, BinOp::Div) && !tensor_on_left && values.contains(&0.0) {
                        return Err(EvalError::DivZero);
                    }
                    let scalar_f32 = scalar as f32;
                    let mut out = Vec::with_capacity(values.len());
                    for &v in values {
                        let computed = match op {
                            BinOp::Add => v + scalar_f32,
                            BinOp::Sub => {
                                if tensor_on_left {
                                    v - scalar_f32
                                } else {
                                    scalar_f32 - v
                                }
                            }
                            BinOp::Mul => v * scalar_f32,
                            BinOp::Div => {
                                if tensor_on_left {
                                    v / scalar_f32
                                } else {
                                    scalar_f32 / v
                                }
                            }
                            _ => 0.0, // comparison ops
                        };
                        out.push(computed);
                    }
                    result.buf = Some(Buffer::F32(out));
                }
                _ => {}
            }
        } else if result.fill.is_some() {
            materialize_filled(&mut result);
        }
    }

    Ok(Value::Tensor(result))
}

#[cfg(feature = "mlir-exec")]
fn parse_mlir_stdout(stdout: &str) -> Option<Value> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(int_val) = trimmed.parse::<i64>() {
        return Some(Value::Int(int_val));
    }
    None
}

fn apply_tensor_tensor(
    op: BinOp,
    left: TensorVal,
    right: TensorVal,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    #[cfg(not(feature = "cpu-buffers"))]
    let _ = mode;

    #[cfg(feature = "cpu-buffers")]
    {
        if matches!(mode, ExecMode::CpuExec | ExecMode::Cuda) {
            let mut left_exec = left.clone();
            let mut right_exec = right.clone();
            materialize_filled(&mut left_exec);
            materialize_filled(&mut right_exec);
            #[cfg(feature = "cpu-exec")]
            {
                if left_exec.dtype == DType::F32 && right_exec.dtype == DType::F32 {
                    let exec_res = match op {
                        BinOp::Add => exec::cpu::exec_add(&left_exec, &right_exec),
                        BinOp::Sub => exec::cpu::exec_sub(&left_exec, &right_exec),
                        BinOp::Mul => exec::cpu::exec_mul(&left_exec, &right_exec),
                        BinOp::Div => exec::cpu::exec_div(&left_exec, &right_exec),
                        _ => Err(exec::cpu::ExecError::Unsupported(
                            "comparison op on tensors".to_string(),
                        )),
                    };
                    match exec_res {
                        Ok(t) => return Ok(Value::Tensor(t)),
                        Err(err) => {
                            let mapped = exec_error_to_eval(err);
                            if matches!(mapped, EvalError::DivZero) {
                                return Err(mapped);
                            }
                        }
                    }
                }
            }
        }
    }

    if left.dtype != right.dtype {
        return Err(EvalError::Unsupported);
    }

    let shape = broadcast_shapes(&left.shape, &right.shape).ok_or(EvalError::Unsupported)?;

    #[cfg(feature = "cpu-buffers")]
    let left_buf = left.buf.clone();
    #[cfg(feature = "cpu-buffers")]
    let right_buf = right.buf.clone();

    let left_fill = left.fill;
    let right_fill = right.fill;
    let dtype = left.dtype.clone();

    if matches!(op, BinOp::Div) {
        if let Some(fill) = right_fill {
            if fill == 0.0 {
                return Err(EvalError::DivZero);
            }
        }
        #[cfg(feature = "cpu-buffers")]
        if let Some(buf) = right_buf.as_ref() {
            match buf {
                Buffer::I32(values) => {
                    if values.contains(&0) {
                        return Err(EvalError::DivZero);
                    }
                }
                Buffer::F32(values) => {
                    if values.contains(&0.0) {
                        return Err(EvalError::DivZero);
                    }
                }
            }
        }
    }

    let fill = match (left_fill, right_fill) {
        (Some(a), Some(b)) => Some(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Mod => a % b,
            BinOp::Lt => {
                if a < b {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Le => {
                if a <= b {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Gt => {
                if a > b {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Ge => {
                if a >= b {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Eq => {
                if a == b {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Ne => {
                if a != b {
                    1.0
                } else {
                    0.0
                }
            }
        }),
        _ => None,
    };

    #[cfg_attr(not(feature = "cpu-buffers"), allow(unused_mut))]
    let mut result = TensorVal::new(dtype, shape, fill);

    #[cfg(feature = "cpu-buffers")]
    {
        if let (Some(lb), Some(rb)) = (left_buf.as_ref(), right_buf.as_ref()) {
            if let Some(ne) = num_elems(&result.shape) {
                match (lb, rb) {
                    (Buffer::I32(lv), Buffer::I32(rv)) if lv.len() == ne && rv.len() == ne => {
                        let mut out = Vec::with_capacity(ne);
                        for i in 0..ne {
                            let computed = match op {
                                BinOp::Add => lv[i] + rv[i],
                                BinOp::Sub => lv[i] - rv[i],
                                BinOp::Mul => lv[i] * rv[i],
                                BinOp::Div => lv[i] / rv[i],
                                _ => 0, // comparison ops not supported in element-wise
                            };
                            out.push(computed);
                        }
                        result.buf = Some(Buffer::I32(out));
                    }
                    (Buffer::F32(lv), Buffer::F32(rv)) if lv.len() == ne && rv.len() == ne => {
                        let mut out = Vec::with_capacity(ne);
                        for i in 0..ne {
                            let computed = match op {
                                BinOp::Add => lv[i] + rv[i],
                                BinOp::Sub => lv[i] - rv[i],
                                BinOp::Mul => lv[i] * rv[i],
                                BinOp::Div => lv[i] / rv[i],
                                _ => 0.0, // comparison ops not supported in element-wise
                            };
                            out.push(computed);
                        }
                        result.buf = Some(Buffer::F32(out));
                    }
                    _ => {}
                }
            }
        }

        if result.buf.is_none()
            && result.fill.is_some()
            && left_fill.is_some()
            && right_fill.is_some()
        {
            materialize_filled(&mut result);
        }
    }

    Ok(Value::Tensor(result))
}

pub(crate) fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    let mut result = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;
    while i >= 0 || j >= 0 {
        let da = if i >= 0 {
            &a[i as usize]
        } else {
            &ShapeDim::Known(1)
        };
        let db = if j >= 0 {
            &b[j as usize]
        } else {
            &ShapeDim::Known(1)
        };
        let dim = match (da, db) {
            (ShapeDim::Known(x), ShapeDim::Known(y)) if x == y => ShapeDim::Known(*x),
            (ShapeDim::Known(1), ShapeDim::Known(y)) => ShapeDim::Known(*y),
            (ShapeDim::Known(x), ShapeDim::Known(1)) => ShapeDim::Known(*x),
            (ShapeDim::Sym(s1), ShapeDim::Sym(s2)) if s1 == s2 => ShapeDim::Sym(s1),
            (ShapeDim::Sym(sym), ShapeDim::Known(1)) | (ShapeDim::Known(1), ShapeDim::Sym(sym)) => {
                ShapeDim::Sym(sym)
            }
            _ => return None,
        };
        result.push(dim);
        i -= 1;
        j -= 1;
    }
    result.reverse();
    Some(result)
}

fn parse_tensor_ann(dtype: &str, dims: &[String]) -> Result<(DType, Vec<ShapeDim>), EvalError> {
    let dtype = dtype.parse().map_err(|_| EvalError::Unsupported)?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        if let Ok(n) = dim.parse::<usize>() {
            shape.push(ShapeDim::Known(n));
        } else {
            shape.push(ShapeDim::Sym(crate::types::intern::intern_str(dim)));
        }
    }
    Ok((dtype, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn eval_tensor_add_scalar_preview() {
        let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.fill, Some(1.0));
            }
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn eval_borrow_read_yields_inner_value() {
        // Phase 10.7: the immutable interpreter has no real handles, so a borrow
        // expression `&x` (and `&mut x`) is a READ that evaluates the inner
        // expression and yields its value.
        let src = "let x = 5; &x";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 5, "`&x` should read x's value"),
            other => panic!("expected Int(5), got {other:?}"),
        }
    }

    #[test]
    fn eval_mut_borrow_read_yields_inner_value() {
        // `&mut x` in expression position is also a read in the interpreter.
        let src = "let x = 7; &mut x";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 7, "`&mut x` should read x's value"),
            other => panic!("expected Int(7), got {other:?}"),
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn eval_while_loop_counts() {
        // RFC 0005 Gap 1: the interpreter executes `while`. `i` is incremented
        // in the loop scope until the condition fails; the loop value is the
        // last body statement's result (i after the final increment).
        let src = "let i = 0; while i < 3 { i = i + 1 }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 3, "while should count i to 3"),
            other => panic!("expected Int(3), got {other:?}"),
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn eval_while_false_condition_is_zero() {
        // A while whose condition is immediately false runs zero iterations and
        // evaluates to the unit placeholder (Int(0)), never touching the body.
        let src = "let i = 5; while i < 3 { i = i + 1 }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 0, "zero-iteration while should be Int(0)"),
            other => panic!("expected Int(0), got {other:?}"),
        }
    }

    #[test]
    fn eval_match_selects_literal_arm() {
        // Phase 10.7: the interpreter takes the first arm whose literal pattern
        // equals the scrutinee.
        let src = "let x = 2; match x { 1 => 10, 2 => 20, _ => 99 }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 20, "match should select the `2 =>` arm"),
            other => panic!("expected Int(20), got {other:?}"),
        }
    }

    #[test]
    fn eval_match_wildcard_fallthrough() {
        // An unmatched scrutinee falls through to the wildcard arm.
        let src = "let x = 7; match x { 1 => 10, 2 => 20, _ => 99 }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 99, "unmatched scrutinee hits `_`"),
            other => panic!("expected Int(99), got {other:?}"),
        }
    }

    #[test]
    fn eval_array_index_access() {
        // Phase 10.6: array literals evaluate to tuples; indexing returns the
        // bounds-checked element.
        let src = "let arr = [10, 20, 30]; arr[1]";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 20, "arr[1] should be 20"),
            other => panic!("expected Int(20), got {other:?}"),
        }
    }

    #[test]
    fn eval_tensor_shape_method() {
        // `.shape()` on a tensor returns its dimensions as a tuple — which
        // composes with array indexing (`t.shape()[0]`).
        let src = "let t: Tensor[f32,(2,3)] = 0; t.shape()";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Tuple(items) => {
                assert_eq!(items.len(), 2, "shape of (2,3) has 2 dims");
                assert!(matches!(items[0], Value::Int(2)), "dim0 = 2");
                assert!(matches!(items[1], Value::Int(3)), "dim1 = 3");
            }
            other => panic!("expected shape tuple [2, 3], got {other:?}"),
        }
    }

    #[test]
    fn eval_array_index_assign() {
        // Phase 10.6: `arr[i] = v` rebinds the array; a later read sees the write.
        let src = "let arr = [1, 2, 3]; arr[1] = 20; arr[1]";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 20, "arr[1] should be 20 after assignment"),
            other => panic!("expected Int(20), got {other:?}"),
        }
    }

    #[test]
    fn eval_array_index_out_of_bounds_errors() {
        let src = "let arr = [10, 20, 30]; arr[5]";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let err = eval_module_value_with_env(&module, &mut env, Some(src));
        assert!(
            err.is_err(),
            "out-of-bounds index should error, got {err:?}"
        );
    }

    #[test]
    fn eval_for_loop_array_fill() {
        // Array-building in a for loop: each iteration writes arr[i]; the final
        // read sees all writes.
        let src = "let arr = [0, 0, 0]; for i in 0..3 { arr[i] = i + 10 }; arr[2]";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 12, "arr[2] should be 12 after the fill loop"),
            other => panic!("expected Int(12), got {other:?}"),
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn eval_while_loop_array_mutation() {
        // `arr[i] = v` inside a while body: the write is visible to a later read in
        // the same loop scope. (The while value is its last body statement.)
        let src = "let arr = [0, 0, 0]; let i = 0; let sum = 0; \
                   while i < 3 { arr[i] = i + 5; i = i + 1; sum = sum + arr[i - 1] }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 18, "5 + 6 + 7 = 18"),
            other => panic!("expected Int(18), got {other:?}"),
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn eval_region_array_mutation() {
        // `arr[i] = v` inside a region block mutates within the region scope.
        let src = "region { let a = [0, 0, 0]; a[1] = 7; a[1] }";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 7, "a[1] should be 7 after region mutation"),
            other => panic!("expected Int(7), got {other:?}"),
        }
    }

    #[test]
    fn eval_generic_fn_call_runs() {
        // Phase 10.x: a generic function call executes in the interpreter. Type
        // params are recorded on the FnDef but the dynamically-typed evaluator
        // just binds the concrete argument Value, so `id<T>(x) = x` instantiated
        // at int runs. (Real codegen monomorphization is a later slice.)
        let src = "fn id<T>(x: T) -> T { x }\nid(5)";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Int(n) => assert_eq!(n, 5, "generic id(5) should run and return 5"),
            other => panic!("expected Int(5), got {other:?}"),
        }
    }

    /// Regression: the interpreter's narrowing `as`-cast must truncate to the
    /// target width, matching the lowered IR / compiled artifact. Before the
    /// fix the `Node::As` arm was a transparent no-op, so `300 as u8` yielded
    /// the RAW `300` while codegen masked it to `44` (300 & 0xFF) — a silent
    /// miscompile that flipped any downstream comparison (`(300 as u8) == 44`
    /// evaluated false in the interpreter but true in the runnable artifact).
    #[cfg(feature = "std-surface")]
    #[test]
    fn eval_as_cast_narrows_to_width() {
        let eval_int = |src: &str| -> i64 {
            let module = parser::parse(src).unwrap();
            let mut env = HashMap::new();
            match eval_module_value_with_env(&module, &mut env, Some(src)).unwrap() {
                Value::Int(n) => n,
                other => panic!("expected Int for `{src}`, got {other:?}"),
            }
        };
        // Unsigned narrowings zero-extend (mask the low W bits).
        assert_eq!(eval_int("300 as u8"), 44, "300 as u8 must mask to 44");
        assert_eq!(eval_int("(0 - 1) as u8"), 255, "-1 as u8 must mask to 255");
        assert_eq!(eval_int("5000 as u8"), 136, "5000 as u8 must mask to 136");
        assert_eq!(
            eval_int("4294967296 as u32"),
            0,
            "2^32 as u32 must mask to 0"
        );
        // Signed narrowings truncate then sign-extend.
        assert_eq!(
            eval_int("200 as i8"),
            -56,
            "200 as i8 must sign-extend to -56"
        );
        assert_eq!(
            eval_int("100000 as i16"),
            -31072,
            "100000 as i16 must sign-extend"
        );
        // The downstream EQUALITY must now pick the right branch, matching the
        // runnable artifact (this is the comparison the no-op cast corrupted).
        assert_eq!(
            eval_int("(300 as u8) == 44"),
            1,
            "(300 as u8) == 44 must be true (1), as in the compiled artifact"
        );
        // A non-narrowing cast (i64, u64, pointer-width) stays transparent.
        assert_eq!(eval_int("300 as i64"), 300, "300 as i64 is unchanged");
    }

    /// Salov C3 (#179): an early `return X` must STOP the enclosing function
    /// body and yield `X`, exactly like the compiled/native path. Before the
    /// fix the tree evaluator ran the body straight-line, so a later statement
    /// clobbered the returned value (an `if`-branch return was ignored, and a
    /// loop return did not stop the loop or the fn).
    #[cfg(feature = "std-surface")]
    #[test]
    fn early_return_short_circuits_function_body() {
        // Const-eval `<fn>(<arg>)` by binding it at top level and reading the
        // module's last value (Preview / const-eval mode).
        let call = |defs: &str, expr: &str| -> i64 {
            let src = format!("{defs}\nlet __probe: i64 = {expr}\n");
            let module = parser::parse(&src).unwrap();
            let mut env = HashMap::new();
            match eval_module_value_with_env(&module, &mut env, Some(&src)).unwrap() {
                Value::Int(n) => n,
                other => panic!("expected Int for `{expr}`, got {other:?}"),
            }
        };

        // [A] return inside an `if` branch — a dispatch chain. A regression
        // falls through to the trailing `return 1`.
        let classify = "pub fn classify(x: i64) -> i64 { \
            if x < 0 { return -1 } if x == 0 { return 0 } return 1 }";
        assert_eq!(call(classify, "classify(0 - 5)"), -1, "return-in-if (neg)");
        assert_eq!(call(classify, "classify(0)"), 0, "return-in-if (zero)");
        assert_eq!(call(classify, "classify(7)"), 1, "return-in-if (pos)");

        // [B] return inside a `while` loop — must stop the loop AND the fn.
        // A regression runs the loop to its cap and returns the trailing -1.
        let first_ge = "pub fn first_ge(limit: i64) -> i64 { \
            let mut i: i64 = 0 while i < 1000 { if i * i >= limit { return i } i = i + 1 } \
            return -1 }";
        assert_eq!(
            call(first_ge, "first_ge(50)"),
            8,
            "return-in-loop stops loop+fn"
        );
        assert_eq!(
            call(first_ge, "first_ge(0)"),
            0,
            "return-in-loop first iter"
        );

        // [C] return as the last statement — the normal implicit-return path,
        // must not regress.
        let double = "pub fn double(x: i64) -> i64 { let y: i64 = x * 2 return y }";
        assert_eq!(call(double, "double(21)"), 42, "return-last-stmt");

        // [D] return BEFORE a side-effecting statement — the early exit must
        // skip the later mutation. A regression returns 120 for guarded(20).
        let guarded = "pub fn guarded(x: i64) -> i64 { \
            let mut acc: i64 = x if x > 10 { return acc } acc = acc + 100 return acc }";
        assert_eq!(
            call(guarded, "guarded(20)"),
            20,
            "return-before-side-effect"
        );
        assert_eq!(
            call(guarded, "guarded(5)"),
            105,
            "no-return runs side effect"
        );
    }

    /// A bare `return;` (no operand) yields the unit placeholder and still
    /// short-circuits the body.
    #[cfg(feature = "std-surface")]
    #[test]
    fn bare_return_short_circuits_with_unit() {
        let src = "pub fn f(x: i64) -> i64 { if x > 0 { return } let y: i64 = 99 return y }\n\
                   let __probe: i64 = f(1)\n";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        match eval_module_value_with_env(&module, &mut env, Some(src)).unwrap() {
            Value::Int(n) => assert_eq!(n, 0, "bare `return;` yields unit (0) and skips the rest"),
            other => panic!("expected Int, got {other:?}"),
        }
    }
}

// GPU runtime dispatch — set by the runtime before eval
use std::cell::RefCell;

// ── User-defined function table (interpreter) ────────────────────────
//
// The immutable value-interpreter previously had no way to *call* a
// user-defined `fn`: `Node::Call` fell through to the tensor stdlib,
// which returns `EvalError::Unsupported` for an unknown callee. This
// table is populated from the module's `FnDef` items before the
// statement loop runs and cleared afterward, so a call like `id(5)`
// resolves to the function body bound against the concrete argument
// `Value`s.
//
// Generics are free here: the interpreter is dynamically typed over
// `Value`, so a generic `fn id<T>(x: T) -> T { x }` is evaluated with
// the concrete argument value — no monomorphization is needed at this
// level. `type_params` is recorded on the AST node for later codegen
// monomorphization; the interpreter simply ignores it.
#[derive(Clone)]
struct UserFn {
    params: Vec<String>,
    body: Vec<Node>,
}

thread_local! {
    static FN_TABLE: RefCell<HashMap<String, UserFn>> = RefCell::new(HashMap::new());
}

/// Register every top-level `FnDef` (generic or not) so `Node::Call`
/// can resolve a user-defined callee. Returns the previous table so the
/// caller can restore it (nested module evals are not expected, but this
/// keeps the thread-local re-entrant and leak-free).
fn fn_table_install(m: &Module) -> HashMap<String, UserFn> {
    let mut table: HashMap<String, UserFn> = HashMap::new();
    for item in &m.items {
        if let Node::FnDef(fd, _) = item {
            let (name, params, body) = (&fd.name, &fd.params, &fd.body);
            // E2023 defence-in-depth: never let a user `fn __mind_*` shadow a
            // reserved intrinsic on the interpreter oracle. The type checker
            // rejects such a module fail-loud (E2023); this guard keeps the
            // fn-table honest even if an install path bypasses the checker.
            if name.starts_with("__mind_") {
                continue;
            }
            table.insert(
                name.clone(),
                UserFn {
                    params: params.iter().map(|p| p.name.clone()).collect(),
                    body: body.clone(),
                },
            );
        }
    }
    FN_TABLE.with(|t| std::mem::replace(&mut *t.borrow_mut(), table))
}

/// Restore a previously-saved function table (paired with
/// `fn_table_install`).
// Wired up when generic-call execution gains scoped fn-tables (the install
// site at `fn_table_install` currently discards `_fn_prev`); kept to land that
// slice without re-deriving the restore half.
#[allow(dead_code)]
fn fn_table_restore(prev: HashMap<String, UserFn>) {
    FN_TABLE.with(|t| *t.borrow_mut() = prev);
}

/// Look up a user-defined function by callee name.
fn fn_table_lookup(name: &str) -> Option<UserFn> {
    FN_TABLE.with(|t| t.borrow().get(name).cloned())
}

// issue #99: names of top-level bindings declared `u64`. The const-fold
// interpreter carries scalars as untyped `Value::Int(i64)`, so signedness is
// not on the value — it is recovered here from the DECLARED binding type (the
// same source the type checker uses). Consulted by the `Binary`/`Bitwise` arms
// to pick UNSIGNED semantics (`>`/`>=`/`<`/`<=`/`/`/`%`/`>>`) for u64 operands,
// so the tree-walking evaluator agrees bit-for-bit with the compiled artifact.
// Populated by the top-level eval loop's `Let` arm; snapshotted/restored per
// module eval so nested evals do not leak. Empty for any u64-free program, so
// its results are unchanged.
thread_local! {
    static U64_VARS: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Record (or, for a shadowing non-u64 rebind, clear) a top-level binding's
/// u64-ness by declared annotation.
fn u64_vars_set(name: &str, is_u64: bool) {
    U64_VARS.with(|s| {
        if is_u64 {
            s.borrow_mut().insert(name.to_string());
        } else {
            s.borrow_mut().remove(name);
        }
    });
}

/// Snapshot + clear the u64-var set (paired with `u64_vars_restore`) so a
/// nested module eval starts clean and cannot leak into the caller.
fn u64_vars_take() -> std::collections::HashSet<String> {
    U64_VARS.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

// Paired with `u64_vars_take`. Mirrors `fn_table_restore`: the install site
// discards its snapshot (each module eval starts from a `take`-cleared slate,
// which is the actual correctness property), so this restore half is kept for
// the scoped-nested-eval slice without being wired yet.
#[allow(dead_code)]
fn u64_vars_restore(prev: std::collections::HashSet<String>) {
    U64_VARS.with(|s| *s.borrow_mut() = prev);
}

/// Whether an expression is a declared-`u64` value in the interpreter: a
/// `u64`-tracked ident, a `… as u64` cast, or either wrapped in parentheses.
/// Mirror of `type_checker::expr_is_u64` — declaration/cast-driven only, so the
/// unsigned-op selection it feeds never over-fires onto a genuine i64.
fn interp_expr_is_u64(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Ident(name), _) => U64_VARS.with(|s| s.borrow().contains(name)),
        Node::As { ty, .. } => matches!(ty, TypeAnn::Named(n) if n == "u64"),
        Node::Paren(inner, _) => interp_expr_is_u64(inner),
        _ => false,
    }
}

thread_local! {
    static GPU_RUNTIME: RefCell<Option<Box<dyn crate::runtime_interface::MindRuntime>>> = RefCell::new(None);
}

/// Set the GPU runtime for tensor dispatch during eval.
/// Called by mind-runtime before eval_module_value_with_env_mode.
pub fn set_gpu_runtime(rt: Box<dyn crate::runtime_interface::MindRuntime>) {
    GPU_RUNTIME.with(|r| *r.borrow_mut() = Some(rt));
}

/// Clear the GPU runtime after eval completes.
pub fn clear_gpu_runtime() {
    GPU_RUNTIME.with(|r| *r.borrow_mut() = None);
}

// GPU matmul dispatch function — set by runtime with CUDA backend
pub type GpuMatmulFn = Box<dyn Fn(&TensorVal, &TensorVal) -> Result<TensorVal, EvalError> + Send>;

thread_local! {
    pub static GPU_MATMUL_FN: RefCell<Option<GpuMatmulFn>> = RefCell::new(None);
}

/// Set the GPU matmul function for CUDA dispatch.
pub fn set_gpu_matmul(f: GpuMatmulFn) {
    GPU_MATMUL_FN.with(|r| *r.borrow_mut() = Some(f));
}
