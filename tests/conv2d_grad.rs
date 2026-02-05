// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Tests for Conv2d gradient computation (VJP).
//!
//! These tests verify correctness of the analytical gradients computed by
//! conv2d_vjp_nhwc_hwio_f32 against numerical finite-difference approximations.

use libmind::eval::conv2d_grad::{conv2d_output_shape, conv2d_vjp_nhwc_hwio_f32, Conv2dVjpParams};
use libmind::types::ConvPadding;

/// Compute forward Conv2d: y = conv2d(x, w)
/// Input x: NHWC [N, H, W, C]
/// Filter w: HWIO [KH, KW, C, O]
/// Output y: NHWC [N, OH, OW, O]
fn conv2d_forward(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> (Vec<f32>, [usize; 4]) {
    let out_shape = conv2d_output_shape(x_shape, w_shape, stride_h, stride_w, padding);
    let [n, oh, ow, o] = out_shape;
    let [_n, h, ww, c] = x_shape;
    let [kh, kw, _c, _o] = w_shape;

    let (pad_top, pad_left) = compute_padding(h, ww, kh, kw, stride_h, stride_w, padding);

    let mut y = vec![0.0f32; n * oh * ow * o];

    for ni in 0..n {
        for ohi in 0..oh {
            for owi in 0..ow {
                for oi in 0..o {
                    let mut sum = 0.0f32;
                    for khi in 0..kh {
                        let ih = (ohi * stride_h + khi) as isize - pad_top as isize;
                        if ih < 0 || ih >= h as isize {
                            continue;
                        }
                        for kwi in 0..kw {
                            let iw = (owi * stride_w + kwi) as isize - pad_left as isize;
                            if iw < 0 || iw >= ww as isize {
                                continue;
                            }
                            for ci in 0..c {
                                let x_idx = idx_nhwc(ni, ih as usize, iw as usize, ci, h, ww, c);
                                let w_idx = idx_hwio(khi, kwi, ci, oi, kw, c, o);
                                sum += x[x_idx] * w[w_idx];
                            }
                        }
                    }
                    let y_idx = idx_nhwc(ni, ohi, owi, oi, oh, ow, o);
                    y[y_idx] = sum;
                }
            }
        }
    }

    (y, out_shape)
}

fn compute_padding(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    padding: ConvPadding,
) -> (usize, usize) {
    match padding {
        ConvPadding::Valid => (0, 0),
        ConvPadding::Same => {
            let oh = (h + sh - 1) / sh;
            let ow = (w + sw - 1) / sw;
            let pad_h = (oh.saturating_sub(1)) * sh + kh;
            let pad_h = pad_h.saturating_sub(h);
            let pad_w = (ow.saturating_sub(1)) * sw + kw;
            let pad_w = pad_w.saturating_sub(w);
            (pad_h / 2, pad_w / 2)
        }
    }
}

#[inline]
fn idx_nhwc(n: usize, h: usize, w: usize, c: usize, hh: usize, ww: usize, cc: usize) -> usize {
    (((n * hh + h) * ww + w) * cc) + c
}

#[inline]
fn idx_hwio(kh: usize, kw: usize, c: usize, o: usize, kww: usize, cc: usize, oo: usize) -> usize {
    (((kh * kww + kw) * cc + c) * oo) + o
}

/// Compute loss = sum(conv2d(x, w) * r) where r is a fixed upstream tensor.
fn compute_loss(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    r: &[f32],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> f32 {
    let (y, _) = conv2d_forward(x, x_shape, w, w_shape, stride_h, stride_w, padding);
    y.iter().zip(r.iter()).map(|(a, b)| a * b).sum()
}

/// Compute numerical gradient using finite differences.
fn numerical_gradient_x(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    r: &[f32],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
    eps: f32,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + eps;
        x_minus[i] = x[i] - eps;
        let loss_plus = compute_loss(&x_plus, x_shape, w, w_shape, r, stride_h, stride_w, padding);
        let loss_minus = compute_loss(
            &x_minus, x_shape, w, w_shape, r, stride_h, stride_w, padding,
        );
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

fn numerical_gradient_w(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    r: &[f32],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
    eps: f32,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; w.len()];
    let mut w_plus = w.to_vec();
    let mut w_minus = w.to_vec();

    for i in 0..w.len() {
        w_plus[i] = w[i] + eps;
        w_minus[i] = w[i] - eps;
        let loss_plus = compute_loss(x, x_shape, &w_plus, w_shape, r, stride_h, stride_w, padding);
        let loss_minus = compute_loss(
            x, x_shape, &w_minus, w_shape, r, stride_h, stride_w, padding,
        );
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
        w_plus[i] = w[i];
        w_minus[i] = w[i];
    }

    grad
}

fn check_gradient_match(analytic: &[f32], numerical: &[f32], rtol: f32, atol: f32) -> bool {
    if analytic.len() != numerical.len() {
        return false;
    }

    for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
        let diff = (a - n).abs();
        let tol = atol + rtol * n.abs().max(a.abs());
        if diff > tol {
            eprintln!(
                "Gradient mismatch at index {}: analytic={}, numerical={}, diff={}, tol={}",
                i, a, n, diff, tol
            );
            return false;
        }
    }

    true
}

#[test]
fn test_conv2d_grad_valid_stride1() {
    // Test case 1: padding=Valid, stride=1
    let x_shape = [1, 5, 6, 2];
    let w_shape = [3, 3, 2, 3];

    let [n, h, ww, c] = x_shape;
    let [kh, kw, _c, o] = w_shape;

    // Initialize with deterministic values
    let mut x: Vec<f32> = (0..(n * h * ww * c))
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let mut w: Vec<f32> = (0..(kh * kw * c * o))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    // Normalize to avoid numerical issues
    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let w_norm: f32 = w.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in &mut x {
        *v /= x_norm.max(1.0);
    }
    for v in &mut w {
        *v /= w_norm.max(1.0);
    }

    let out_shape = conv2d_output_shape(x_shape, w_shape, 1, 1, ConvPadding::Valid);
    let [n_o, oh, ow, o_o] = out_shape;

    // Upstream gradient (random but deterministic)
    let r: Vec<f32> = (0..(n_o * oh * ow * o_o))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    // Compute analytical gradients
    let (dx_analytic, dw_analytic) = conv2d_vjp_nhwc_hwio_f32(Conv2dVjpParams {
        x: &x,
        x_shape,
        w: &w,
        w_shape,
        dy: &r,
        dy_shape: out_shape,
        stride: [1, 1],
        padding: ConvPadding::Valid,
    });

    // Compute numerical gradients
    let eps = 1e-4;
    let dx_numerical =
        numerical_gradient_x(&x, x_shape, &w, w_shape, &r, 1, 1, ConvPadding::Valid, eps);
    let dw_numerical =
        numerical_gradient_w(&x, x_shape, &w, w_shape, &r, 1, 1, ConvPadding::Valid, eps);

    // Check match with tolerance
    let rtol = 1e-3;
    let atol = 1e-4;

    assert!(
        check_gradient_match(&dx_analytic, &dx_numerical, rtol, atol),
        "dx gradient mismatch for padding=Valid, stride=1"
    );
    assert!(
        check_gradient_match(&dw_analytic, &dw_numerical, rtol, atol),
        "dw gradient mismatch for padding=Valid, stride=1"
    );
}

#[test]
fn test_conv2d_grad_same_stride2() {
    // Test case 2: padding=Same, stride=2
    let x_shape = [1, 5, 6, 2];
    let w_shape = [3, 3, 2, 3];

    let [n, h, ww, c] = x_shape;
    let [kh, kw, _c, o] = w_shape;

    // Initialize with deterministic values
    let mut x: Vec<f32> = (0..(n * h * ww * c))
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let mut w: Vec<f32> = (0..(kh * kw * c * o))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    // Normalize
    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let w_norm: f32 = w.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in &mut x {
        *v /= x_norm.max(1.0);
    }
    for v in &mut w {
        *v /= w_norm.max(1.0);
    }

    let out_shape = conv2d_output_shape(x_shape, w_shape, 2, 2, ConvPadding::Same);
    let [n_o, oh, ow, o_o] = out_shape;

    // Upstream gradient
    let r: Vec<f32> = (0..(n_o * oh * ow * o_o))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    // Compute analytical gradients
    let (dx_analytic, dw_analytic) = conv2d_vjp_nhwc_hwio_f32(Conv2dVjpParams {
        x: &x,
        x_shape,
        w: &w,
        w_shape,
        dy: &r,
        dy_shape: out_shape,
        stride: [2, 2],
        padding: ConvPadding::Same,
    });

    // Compute numerical gradients
    let eps = 1e-4;
    let dx_numerical =
        numerical_gradient_x(&x, x_shape, &w, w_shape, &r, 2, 2, ConvPadding::Same, eps);
    let dw_numerical =
        numerical_gradient_w(&x, x_shape, &w, w_shape, &r, 2, 2, ConvPadding::Same, eps);

    // Check match with tolerance
    let rtol = 1e-3;
    let atol = 1e-4;

    assert!(
        check_gradient_match(&dx_analytic, &dx_numerical, rtol, atol),
        "dx gradient mismatch for padding=Same, stride=2"
    );
    assert!(
        check_gradient_match(&dw_analytic, &dw_numerical, rtol, atol),
        "dw gradient mismatch for padding=Same, stride=2"
    );
}

#[test]
fn test_conv2d_output_shape_valid() {
    let x_shape = [2, 5, 6, 3];
    let w_shape = [3, 3, 3, 8];
    let out = conv2d_output_shape(x_shape, w_shape, 1, 1, ConvPadding::Valid);
    assert_eq!(out, [2, 3, 4, 8]);
}

#[test]
fn test_conv2d_output_shape_same() {
    let x_shape = [2, 5, 6, 3];
    let w_shape = [3, 3, 3, 8];
    let out = conv2d_output_shape(x_shape, w_shape, 2, 2, ConvPadding::Same);
    assert_eq!(out, [2, 3, 3, 8]);
}

#[test]
fn test_conv2d_grad_small_exact() {
    // Small exact test case to verify basic correctness
    // 1x2x2x1 input, 2x2x1x1 filter, stride 1, valid padding
    // Output: 1x1x1x1

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let x_shape = [1, 2, 2, 1];

    let w = vec![1.0, 0.0, 0.0, 1.0]; // Identity-like filter
    let w_shape = [2, 2, 1, 1];

    // Forward: y = 1*1 + 2*0 + 3*0 + 4*1 = 5
    let (y, y_shape) = conv2d_forward(&x, x_shape, &w, w_shape, 1, 1, ConvPadding::Valid);
    assert_eq!(y_shape, [1, 1, 1, 1]);
    assert!((y[0] - 5.0).abs() < 1e-6);

    // dy = 1 (upstream gradient)
    let dy = vec![1.0];

    let (dx, dw) = conv2d_vjp_nhwc_hwio_f32(Conv2dVjpParams {
        x: &x,
        x_shape,
        w: &w,
        w_shape,
        dy: &dy,
        dy_shape: y_shape,
        stride: [1, 1],
        padding: ConvPadding::Valid,
    });

    // dx = dy * w (scatter)
    // dx[0,0,0,0] = dy * w[0,0,0,0] = 1 * 1 = 1
    // dx[0,0,1,0] = dy * w[0,1,0,0] = 1 * 0 = 0
    // dx[0,1,0,0] = dy * w[1,0,0,0] = 1 * 0 = 0
    // dx[0,1,1,0] = dy * w[1,1,0,0] = 1 * 1 = 1
    assert!((dx[0] - 1.0).abs() < 1e-6, "dx[0] = {}", dx[0]);
    assert!((dx[1] - 0.0).abs() < 1e-6, "dx[1] = {}", dx[1]);
    assert!((dx[2] - 0.0).abs() < 1e-6, "dx[2] = {}", dx[2]);
    assert!((dx[3] - 1.0).abs() < 1e-6, "dx[3] = {}", dx[3]);

    // dw = dy * x (gather)
    // dw[0,0,0,0] = dy * x[0,0,0,0] = 1 * 1 = 1
    // dw[0,1,0,0] = dy * x[0,0,1,0] = 1 * 2 = 2
    // dw[1,0,0,0] = dy * x[0,1,0,0] = 1 * 3 = 3
    // dw[1,1,0,0] = dy * x[0,1,1,0] = 1 * 4 = 4
    assert!((dw[0] - 1.0).abs() < 1e-6, "dw[0] = {}", dw[0]);
    assert!((dw[1] - 2.0).abs() < 1e-6, "dw[1] = {}", dw[1]);
    assert!((dw[2] - 3.0).abs() < 1e-6, "dw[2] = {}", dw[2]);
    assert!((dw[3] - 4.0).abs() < 1e-6, "dw[3] = {}", dw[3]);
}
