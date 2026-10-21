// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Candidate scalar native delivery profile. This does not select a compiler,
//! change the historical frozen predicate, or establish seed qualification.
//! Source facts are mandatory: legacy IR erases scalar widths and annotations.
//! Only the closed i64/f64 scalar surface is admitted. C extern compatibility,
//! unsigned/narrow arithmetic, aggregates and implicit std dependencies refuse.

mod admit;
mod types;

pub use admit::{profile_lane_admits, profile_lane_admits_source};
pub use types::LaneProfileRejection;
pub(crate) use types::intrinsic_arity;

#[cfg(test)]
use crate::ast;
#[cfg(test)]
pub use crate::ir::BinOp;
#[cfg(test)]
use crate::ir::{IRModule, Instr, ValueId};
#[cfg(test)]
use admit::diverges;
#[cfg(test)]
use types::Check;

#[cfg(test)]
mod tests;
