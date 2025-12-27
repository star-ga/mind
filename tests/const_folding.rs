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

use mind::ast::Node;
use mind::opt::fold;
use mind::parser;

#[test]
fn folds_simple_arith() {
    let m = parser::parse("1 + 2 * 3").unwrap();
    let node = &m.items[0];
    let f = fold::fold(node);
    if let Node::Lit(_, _) = f {
        // folded to literal
    } else {
        panic!("not folded");
    }
}
