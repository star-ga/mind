// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Candidate scalar native delivery profile. This does not select a compiler,
//! change the historical frozen predicate, or establish seed qualification.
//! Source facts are mandatory: legacy IR erases scalar widths and annotations.
//! Only the closed i64/f64 scalar surface is admitted, plus reads of a fixed
//! i64 array bound by `let` to a literal (`arrays.rs`). Unsigned/narrow
//! arithmetic and other aggregates refuse, each with a named kind (`names.rs`);
//! std seed functions are admitted only as reached and walked under the same
//! rules (`seed`); fixed-arity `extern "C"` declarations are admitted only
//! against the closed libc table and only under a compiler carrying the
//! `nbextc01` mark (`admit_extern`).

mod admit;
mod admit_extern;
mod admit_extern_cuda;
mod admit_ir;
mod alloc_size;
mod arrays;
mod names;
mod seed;
mod types;

pub use admit::profile_lane_admits_source;
pub use admit_ir::profile_lane_admits;
pub use seed::{
    LaneEdges, LaneSeed, profile_lane_admits_seeded, profile_lane_admits_source_seeded,
};
pub use types::LaneProfileRejection;
// The native bridge (std-surface) routes every other size through its guard.
#[cfg_attr(not(feature = "std-surface"), allow(unused_imports))]
pub(crate) use alloc_size::positive_literal as alloc_size_is_positive_literal;
pub(crate) use types::intrinsic_arity;

#[cfg(test)]
use crate::ast;
#[cfg(test)]
use crate::ir::{BinOp, IRModule, Instr, ValueId};
#[cfg(test)]
use admit_ir::diverges;
#[cfg(test)]
use types::Check;

#[cfg(test)]
mod alloc_size_tests;
#[cfg(test)]
mod arrays_tests;
#[cfg(test)]
mod tests;
