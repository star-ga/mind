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

use super::TensorType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    ScalarI32,
    ScalarI64,
    ScalarF32,
    ScalarF64,
    ScalarBool,
    Tensor(TensorType),
    GradMap(Vec<(String, TensorType)>),
    /// Deref-assign track D2: a typed reference capability `&T` / `&mut T` whose
    /// `target` is the canonical (module-qualified) name of the referenced
    /// record type — the identity used for owner-exact parameter comparison.
    /// `mutable` distinguishes `&mut T` (may drive `*p = v` / `(*p).f = v`) from
    /// `&T` (read-only; may not produce a writable place by coercion). The
    /// carrier is an i64 address at runtime, but this capability MUST survive
    /// field projection, dereference, casts, and call boundaries so a read-only
    /// reference can never be laundered into a writable record.
    Ref {
        mutable: bool,
        target: String,
    },
}

impl ValueType {
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            ValueType::ScalarI32
                | ValueType::ScalarI64
                | ValueType::ScalarF32
                | ValueType::ScalarF64
                | ValueType::ScalarBool
        )
    }
}
