use crate::eval::value::TensorVal;
use crate::types::{ConvPadding, ShapeDim};

use super::cpu::ExecError;

fn shape_as_usize(shape: &[ShapeDim]) -> Result<Vec<usize>, ExecError> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            ShapeDim::Known(v) => out.push(*v),
            ShapeDim::Sym(_) => return Err(ExecError::Shape("symbolic dims".into())),
        }
    }
    Ok(out)
}

pub fn exec_conv2d(
    input: &TensorVal,
    weights: &TensorVal,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> Result<TensorVal, ExecError> {
    if stride_h == 0 || stride_w == 0 {
        return Err(ExecError::Shape("strides must be positive".into()));
    }
    if input.dtype != crate::types::DType::F32 || weights.dtype != crate::types::DType::F32 {
        return Err(ExecError::Type("only f32 tensors supported in cpu-exec".into()));
    }

    let in_shape = shape_as_usize(&input.shape)?;
    let wt_shape = shape_as_usize(&weights.shape)?;
    if in_shape.len() != 4 || wt_shape.len() != 4 {
        return Err(ExecError::Shape("`tensor.conv2d` expects NHWC x HWIO tensors".into()));
    }

    let n = in_shape[0];
    let in_h = in_shape[1];
    let in_w = in_shape[2];
    let in_c = in_shape[3];

    let kernel_h = wt_shape[0];
    let kernel_w = wt_shape[1];
    let kernel_c = wt_shape[2];
    let out_c = wt_shape[3];

    if kernel_h == 0 || kernel_w == 0 {
        return Err(ExecError::Shape("kernel dimensions must be positive".into()));
    }

    if in_c != kernel_c {
        return Err(ExecError::Shape("input/output channel mismatch".into()));
    }

    let (pad_top, pad_left, out_h, out_w) = match padding {
        ConvPadding::Valid => {
            if in_h < kernel_h || in_w < kernel_w {
                return Err(ExecError::Shape("kernel larger than input".into()));
            }
            let out_h = (in_h - kernel_h) / stride_h + 1;
            let out_w = (in_w - kernel_w) / stride_w + 1;
            (0, 0, out_h, out_w)
        }
        ConvPadding::Same => {
            let out_h = (in_h + stride_h - 1) / stride_h;
            let out_w = (in_w + stride_w - 1) / stride_w;
            let pad_h = out_h
                .saturating_sub(1)
                .saturating_mul(stride_h)
                .saturating_add(kernel_h)
                .saturating_sub(in_h);
            let pad_w = out_w
                .saturating_sub(1)
                .saturating_mul(stride_w)
                .saturating_add(kernel_w)
                .saturating_sub(in_w);
            let pad_top = pad_h / 2;
            let pad_left = pad_w / 2;
            (pad_top, pad_left, out_h, out_w)
        }
    };

    let input_buf = input
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("input tensor not materialized".into()))?;
    let weight_buf = weights
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("weight tensor not materialized".into()))?;

    let mut output = vec![0.0f32; n * out_h * out_w * out_c];

    let pad_top_i = pad_top as isize;
    let pad_left_i = pad_left as isize;

    for batch in 0..n {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let out_index = ((batch * out_h + oh) * out_w + ow) * out_c;
                let out_slice = &mut output[out_index..out_index + out_c];
                for val in out_slice.iter_mut() {
                    *val = 0.0;
                }

                for kh in 0..kernel_h {
                    let ih = oh * stride_h + kh;
                    let ih_padded = ih as isize - pad_top_i;
                    if ih_padded < 0 || ih_padded >= in_h as isize {
                        continue;
                    }
                    for kw in 0..kernel_w {
                        let iw = ow * stride_w + kw;
                        let iw_padded = iw as isize - pad_left_i;
                        if iw_padded < 0 || iw_padded >= in_w as isize {
                            continue;
                        }
                        let input_offset = (((batch * in_h + ih_padded as usize) * in_w
                            + iw_padded as usize)
                            * in_c) as usize;
                        let weight_offset_base = ((kh * kernel_w + kw) * in_c) * out_c;
                        for ic in 0..in_c {
                            let input_val = input_buf[input_offset + ic];
                            let weight_offset = weight_offset_base + ic * out_c;
                            for oc in 0..out_c {
                                out_slice[oc] += input_val * weight_buf[weight_offset + oc];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(TensorVal::from_materialized_f32(vec![n, out_h, out_w, out_c], output))
}
