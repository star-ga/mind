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

pub mod autodiff;
pub mod conv2d_grad;
pub mod ir_interp;
pub mod lower;
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

        let handle = eval.runtime.allocate(&desc);
        assert_eq!(handle, 0);

        eval.runtime.run_op("noop", &[], &[]);
        eval.runtime.synchronize();
    }
}

pub use ir_interp::eval_ir;
pub use lower::lower_to_ir;
#[cfg(feature = "mlir-build")]
pub use mlir_build::build_all as build_mlir_artifacts;
#[cfg(feature = "mlir-build")]
pub use mlir_build::resolve_tools as resolve_mlir_build_tools;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildError as MlirBuildError;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildOptions as MlirBuildOptions;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildProducts as MlirBuildProducts;
#[cfg(feature = "mlir-build")]
pub use mlir_build::BuildTools as MlirBuildTools;
pub use mlir_export::emit_mlir_with_opts;
pub use mlir_export::to_mlir;
pub use mlir_export::MlirEmitMode;
pub use mlir_export::MlirEmitOptions;
pub use mlir_export::MlirLowerPreset;
#[cfg(feature = "mlir-exec")]
pub use mlir_run::MlirExecConfig;
pub use value::format_value_human;
pub use value::TensorVal;
pub use value::Value;
pub use value::VarId;

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
    #[error("type error: {0}")]
    TypeError(String),
    #[error("out of bounds")]
    OutOfBounds,
}

pub fn eval_module_value_with_env_mode(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if let Some(src) = src_for_types {
        let mut tenv: HashMap<String, ValueType> = HashMap::new();
        for (name, _value) in env.iter() {
            tenv.insert(name.clone(), ValueType::ScalarI32);
        }
        let diags = crate::type_checker::check_module_types(m, src, &tenv);
        if !diags.is_empty() {
            let msg = diags
                .into_iter()
                .map(|diag| diag.message)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(EvalError::TypeError(msg));
        }
    }

    let mut venv: HashMap<String, Value> = env
        .iter()
        .map(|(name, value)| (name.clone(), Value::Int(*value)))
        .collect();
    let mut tensor_env: HashMap<String, TensorEnvEntry> = HashMap::new();

    let mut last = Value::Int(0_i64);
    for item in &m.items {
        match item {
            Node::Let {
                name, ann, value, ..
            } => {
                let rhs = eval_value_expr_mode(value, &venv, &tensor_env, mode.clone())?;
                let stored = match ann {
                    Some(TypeAnn::Tensor { dtype, dims })
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
                    | None => rhs,
                };
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
            _ => {
                last = eval_value_expr_mode(item, &venv, &tensor_env, mode.clone())?;
            }
        }
    }

    #[allow(unused_variables)]
    match mode {
        ExecMode::Preview | ExecMode::CpuExec => Ok(last),
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

pub(crate) fn eval_value_expr_mode(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    match node {
        Node::Lit(Literal::Int(n), _) => Ok(Value::Int(*n)),
        Node::Lit(Literal::Ident(name), _) => env
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::UnknownVar(name.clone())),
        Node::Paren(inner, _) => eval_value_expr_mode(inner, env, tensor_env, mode.clone()),
        Node::Tuple { elements, .. } => {
            let mut items = Vec::with_capacity(elements.len());
            for item in elements {
                items.push(eval_value_expr_mode(item, env, tensor_env, mode.clone())?);
            }
            Ok(Value::Tuple(items))
        }
        Node::Call { callee, args, .. } => {
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
                        if matches!(mode, ExecMode::CpuExec) {
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
                        if matches!(mode, ExecMode::CpuExec) {
                            let mut tl_exec = tl.clone();
                            let mut tr_exec = tr.clone();
                            materialize_filled(&mut tl_exec);
                            materialize_filled(&mut tr_exec);
                            #[cfg(feature = "cpu-exec")]
                            {
                                // TODO(runtime): dispatch through `Evaluator::runtime` once the
                                // runtime plumbing is threaded into evaluation.
                                if tl_exec.dtype == DType::F32 && tr_exec.dtype == DType::F32 {
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
            apply_binary(*op, lv, rv, mode.clone())
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => {
            eval_value_expr_mode(value, env, tensor_env, mode.clone())
        }
        // Function definitions and control flow - placeholder implementation
        Node::FnDef { .. } => Ok(Value::Int(0)), // Functions are not executed as expressions
        Node::Return { value, .. } => {
            if let Some(v) = value {
                eval_value_expr_mode(v, env, tensor_env, mode.clone())
            } else {
                Ok(Value::Int(0))
            }
        }
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

fn apply_binary(op: BinOp, left: Value, right: Value, mode: ExecMode) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => apply_int_op(op, l, r).map(Value::Int),
        (Value::Tensor(t), Value::Int(s)) => apply_tensor_scalar(op, t, s as f64, true, mode),
        (Value::Int(s), Value::Tensor(t)) => apply_tensor_scalar(op, t, s as f64, false, mode),
        (Value::Tensor(a), Value::Tensor(b)) => apply_tensor_tensor(op, a, b, mode),
        _ => Err(EvalError::Unsupported),
    }
}

fn apply_int_op(op: BinOp, left: i64, right: i64) -> Result<i64, EvalError> {
    Ok(match op {
        BinOp::Add => left + right,
        BinOp::Sub => left - right,
        BinOp::Mul => left * right,
        BinOp::Div => {
            if right == 0 {
                return Err(EvalError::DivZero);
            }
            left / right
        }
    })
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
        if matches!(mode, ExecMode::CpuExec) {
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
        if matches!(mode, ExecMode::CpuExec) {
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
            let leaked: &'static str = Box::leak(dim.clone().into_boxed_str());
            shape.push(ShapeDim::Sym(leaked));
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
}
