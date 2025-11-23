// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

#[cfg(feature = "autodiff")]
#[test]
fn grad_placeholder_runs() {
    let g = mind::autodiff::grad(|| 40 + 2);
    let (v, d) = g();
    assert_eq!(v, 42);
    assert!(d.contains("placeholder"));
}
