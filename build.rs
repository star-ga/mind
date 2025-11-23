// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

fn main() {
    if std::env::var("CARGO_FEATURE_FFI_EXAMPLES").is_ok() {
        println!("cargo:rerun-if-changed=examples/c/min.c");
        cc::Build::new()
            .file("examples/c/min.c")
            .warnings(false)
            .compile("mind_ffi_examples");
    }
}
