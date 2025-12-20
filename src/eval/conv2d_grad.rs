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

//! Reference implementation of Conv2d gradient computation (VJP).
//!
//! This module provides correctness-first implementations of the backward
//! pass for 2D convolution with NHWC input layout and HWIO filter layout.
//! These are used by the eval-based autodiff and can serve as a reference
//! oracle for verifying optimized MLIR lowerings.

use crate::types::ConvPadding;

/// Ceiling division for usize.
#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Compute pad_top/pad_left for SAME padding given input and kernel and stride.
fn same_padding_2d(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> (usize, usize) {
    let out_h = ceil_div(h, sh);
    let out_w = ceil_div(w, sw);

    let pad_h_total = (out_h.saturating_sub(1)) * sh + kh;
    let pad_h_total = pad_h_total.saturating_sub(h);

    let pad_w_total = (out_w.saturating_sub(1)) * sw + kw;
    let pad_w_total = pad_w_total.saturating_sub(w);

    (pad_h_total / 2, pad_w_total / 2)
}

/// NHWC indexing helper: index into a flat buffer with shape [N, H, W, C].
#[inline]
#[allow(non_snake_case)]
fn idx_nhwc(n: usize, h: usize, w: usize, c: usize, H: usize, W: usize, C: usize) -> usize {
    (((n * H + h) * W + w) * C) + c
}

/// HWIO indexing helper: index into a flat buffer with shape [KH, KW, C, O].
#[inline]
#[allow(non_snake_case)]
fn idx_hwio(kh: usize, kw: usize, c: usize, o: usize, KW: usize, C: usize, O: usize) -> usize {
    (((kh * KW + kw) * C + c) * O) + o
}

/// Compute (dX, dW) for Conv2d with:
/// - X: NHWC [N, H, W, C]
/// - W: HWIO [KH, KW, C, O]
/// - dY: NHWC [N, OH, OW, O]
///
/// Returns:
/// - dX: NHWC [N, H, W, C]
/// - dW: HWIO [KH, KW, C, O]
///
/// This is a reference correctness-first implementation with complexity
/// O(N * OH * OW * KH * KW * C * O).
#[allow(non_snake_case)]
pub fn conv2d_vjp_nhwc_hwio_f32(
    x: &[f32],
    x_shape: [usize; 4],
    w: &[f32],
    w_shape: [usize; 4],
    dy: &[f32],
    dy_shape: [usize; 4],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> (Vec<f32>, Vec<f32>) {
    let [N, H, W_in, C] = x_shape;
    let [KH, KW, Cw, O] = w_shape;
    let [Ny, OH, OW, Oy] = dy_shape;

    assert_eq!(N, Ny, "batch mismatch");
    assert_eq!(C, Cw, "in_channels mismatch");
    assert_eq!(O, Oy, "out_channels mismatch");

    let (pad_top, pad_left) = match padding {
        ConvPadding::Valid => (0usize, 0usize),
        ConvPadding::Same => same_padding_2d(H, W_in, KH, KW, stride_h, stride_w),
    };

    let mut dx = vec![0.0f32; N * H * W_in * C];
    let mut dw = vec![0.0f32; KH * KW * C * O];

    for n in 0..N {
        for oh in 0..OH {
            for ow in 0..OW {
                for o in 0..O {
                    let dy_i = idx_nhwc(n, oh, ow, o, OH, OW, O);
                    let g = dy[dy_i];

                    // For each kernel position
                    for kh in 0..KH {
                        let ih_base = oh * stride_h + kh;
                        let ih = ih_base as isize - pad_top as isize;
                        if ih < 0 || ih >= H as isize {
                            continue;
                        }

                        for kw in 0..KW {
                            let iw_base = ow * stride_w + kw;
                            let iw = iw_base as isize - pad_left as isize;
                            if iw < 0 || iw >= W_in as isize {
                                continue;
                            }

                            let ih = ih as usize;
                            let iw = iw as usize;

                            for c in 0..C {
                                let w_i = idx_hwio(kh, kw, c, o, KW, C, O);
                                let x_i = idx_nhwc(n, ih, iw, c, H, W_in, C);

                                // dx += dy * w
                                dx[x_i] += g * w[w_i];

                                // dw += dy * x
                                dw[w_i] += g * x[x_i];
                            }
                        }
                    }
                }
            }
        }
    }

    (dx, dw)
}

/// Compute output shape for Conv2d.
#[allow(non_snake_case)]
pub fn conv2d_output_shape(
    x_shape: [usize; 4],
    w_shape: [usize; 4],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> [usize; 4] {
    let [N, H, W, _C] = x_shape;
    let [KH, KW, _Cw, O] = w_shape;

    let (oh, ow) = match padding {
        ConvPadding::Valid => {
            // When kernel is larger than input, output is 0 (no valid positions)
            let oh = if H >= KH {
                (H - KH) / stride_h + 1
            } else {
                0
            };
            let ow = if W >= KW {
                (W - KW) / stride_w + 1
            } else {
                0
            };
            (oh, ow)
        }
        ConvPadding::Same => (ceil_div(H, stride_h), ceil_div(W, stride_w)),
    };

    [N, oh, ow, O]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_vjp_valid_stride1() {
        // Simple 1x3x3x1 input, 2x2x1x1 filter, stride 1, valid padding
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x_shape = [1, 3, 3, 1];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let w_shape = [2, 2, 1, 1];
        // Output shape: [1, 2, 2, 1]
        // Forward: y[0,0,0,0] = x[0,0,0]*w[0,0] + x[0,0,1]*w[0,1] + x[0,1,0]*w[1,0] + x[0,1,1]*w[1,1]
        //        = 1*1 + 2*0 + 4*0 + 5*1 = 6
        // y[0,0,1,0] = 2*1 + 3*0 + 5*0 + 6*1 = 8
        // y[0,1,0,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
        // y[0,1,1,0] = 5*1 + 6*0 + 8*0 + 9*1 = 14
        let dy = vec![1.0, 1.0, 1.0, 1.0];
        let dy_shape = [1, 2, 2, 1];

        let (dx, dw) = conv2d_vjp_nhwc_hwio_f32(
            &x,
            x_shape,
            &w,
            w_shape,
            &dy,
            dy_shape,
            1,
            1,
            ConvPadding::Valid,
        );

        // Check shapes
        assert_eq!(dx.len(), 9);
        assert_eq!(dw.len(), 4);

        // dw[kh,kw,c,o] = sum over (n,oh,ow) of dy[n,oh,ow,o] * x[n, oh*s+kh, ow*s+kw, c]
        // For all dy = 1:
        // dw[0,0,0,0] = x[0,0,0] + x[0,0,1] + x[0,1,0] + x[0,1,1] = 1+2+4+5 = 12
        // dw[0,1,0,0] = x[0,0,1] + x[0,0,2] + x[0,1,1] + x[0,1,2] = 2+3+5+6 = 16
        // dw[1,0,0,0] = x[0,1,0] + x[0,1,1] + x[0,2,0] + x[0,2,1] = 4+5+7+8 = 24
        // dw[1,1,0,0] = x[0,1,1] + x[0,1,2] + x[0,2,1] + x[0,2,2] = 5+6+8+9 = 28
        assert_eq!(dw[0], 12.0);
        assert_eq!(dw[1], 16.0);
        assert_eq!(dw[2], 24.0);
        assert_eq!(dw[3], 28.0);
    }

    #[test]
    fn test_conv2d_vjp_same_stride2() {
        // Test SAME padding with stride 2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let x_shape = [1, 3, 4, 1];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let w_shape = [2, 2, 1, 1];
        // SAME padding with stride 2: output shape is ceil(3/2) x ceil(4/2) = 2x2
        let dy = vec![1.0, 1.0, 1.0, 1.0];
        let dy_shape = [1, 2, 2, 1];

        let (dx, dw) = conv2d_vjp_nhwc_hwio_f32(
            &x,
            x_shape,
            &w,
            w_shape,
            &dy,
            dy_shape,
            2,
            2,
            ConvPadding::Same,
        );

        // Check shapes
        assert_eq!(dx.len(), 12);
        assert_eq!(dw.len(), 4);

        // Just verify it runs without panicking and produces finite values
        assert!(dx.iter().all(|v| v.is_finite()));
        assert!(dw.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_output_shape_valid() {
        let x_shape = [2, 5, 6, 3];
        let w_shape = [3, 3, 3, 8];
        let out = conv2d_output_shape(x_shape, w_shape, 1, 1, ConvPadding::Valid);
        assert_eq!(out, [2, 3, 4, 8]);
    }

    #[test]
    fn test_output_shape_same() {
        let x_shape = [2, 5, 6, 3];
        let w_shape = [3, 3, 3, 8];
        let out = conv2d_output_shape(x_shape, w_shape, 2, 2, ConvPadding::Same);
        assert_eq!(out, [2, 3, 3, 8]);
    }
}
