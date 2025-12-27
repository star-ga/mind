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

fn main() {
    if std::env::var("CARGO_FEATURE_FFI_EXAMPLES").is_ok() {
        println!("cargo:rerun-if-changed=examples/c/min.c");
        cc::Build::new()
            .file("examples/c/min.c")
            .warnings(false)
            .compile("mind_ffi_examples");
    }
}
