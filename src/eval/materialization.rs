// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Invocation-scoped accounting for compiler-side aggregate materialisation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializationLimits {
    pub payload_bytes: u64,
    pub ir_items: u64,
    pub temp_slots: u64,
}

impl Default for MaterializationLimits {
    fn default() -> Self {
        Self {
            payload_bytes: 2 * 1024 * 1024,
            ir_items: 1_000_000,
            temp_slots: 262_144,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaterializationUsage {
    pub payload_bytes: u64,
    pub ir_items: u64,
    pub temp_slots: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MaterializationRefusal {
    #[error("compiler materialization payload {attempted} bytes exceeds limit {limit}")]
    PayloadLimit { attempted: u64, limit: u64 },
    #[error("compiler materialization emits {attempted} IR items, exceeding limit {limit}")]
    IrItemsLimit { attempted: u64, limit: u64 },
    #[error("compiler materialization needs {attempted} temporary slots, exceeding limit {limit}")]
    TempSlotsLimit { attempted: u64, limit: u64 },
    #[error("compiler materialization accounting overflow")]
    CounterOverflow,
    #[error(
        "unsupported lowering operation `{operation}` at source span {start}..{end}: the inline struct-field ABI has no representation"
    )]
    UnsupportedLoweringOperation {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "invalid lowering receiver `{operation}` at source span {start}..{end}: no SSA binding exists"
    )]
    InvalidLoweringReceiver {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "NonConstantElement: `{operation}` at source span {start}..{end}: an array literal element is not a constant (spec 3.2 fold)"
    )]
    NonConstantElement {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "Q16ConstantOutOfRange: `{operation}` at source span {start}..{end}: the constant is NaN, infinite, or out of the Q16.16 range (spec 3.3)"
    )]
    Q16ConstantOutOfRange {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "Q16Unsupported: `{operation}` at source span {start}..{end}: this route has no Q16.16 conversion for it (spec 3.3)"
    )]
    Q16Unsupported {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "DenseLeafCount: `{operation}` at source span {start}..{end}: the nested literal has a leaf count that is not the product of the declared shape"
    )]
    DenseLeafCount {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "RaggedNest: `{operation}` at source span {start}..{end}: the nested literal is ragged or is not a nest of the declared shape"
    )]
    RaggedNest {
        operation: &'static str,
        start: usize,
        end: usize,
    },
    #[error(
        "InexactIntegerFill: `{operation}` at source span {start}..{end}: the fill of an integer tensor is not an integer the element type holds"
    )]
    InexactIntegerFill {
        operation: &'static str,
        start: usize,
        end: usize,
    },
}

/// The named refusals of a constant element or fill (spec 3.2 and 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstRefusal {
    NonConstant,
    Q16OutOfRange,
    Q16Unsupported,
    InexactIntegerFill,
    DenseLeafCount,
    RaggedNest,
}

impl From<crate::opt::fold::ConstElem> for ConstRefusal {
    fn from(elem: crate::opt::fold::ConstElem) -> Self {
        match elem {
            crate::opt::fold::ConstElem::NonConstant => Self::NonConstant,
            crate::opt::fold::ConstElem::Q16OutOfRange => Self::Q16OutOfRange,
        }
    }
}

impl ConstRefusal {
    /// The refusal for `operation` at source span `start..end`.
    pub fn at(self, operation: &'static str, start: usize, end: usize) -> MaterializationRefusal {
        match self {
            Self::NonConstant => MaterializationRefusal::NonConstantElement {
                operation,
                start,
                end,
            },
            Self::Q16OutOfRange => MaterializationRefusal::Q16ConstantOutOfRange {
                operation,
                start,
                end,
            },
            Self::Q16Unsupported => MaterializationRefusal::Q16Unsupported {
                operation,
                start,
                end,
            },
            Self::InexactIntegerFill => MaterializationRefusal::InexactIntegerFill {
                operation,
                start,
                end,
            },
            Self::DenseLeafCount => MaterializationRefusal::DenseLeafCount {
                operation,
                start,
                end,
            },
            Self::RaggedNest => MaterializationRefusal::RaggedNest {
                operation,
                start,
                end,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaterializationBudget {
    limits: MaterializationLimits,
    usage: MaterializationUsage,
}

impl MaterializationBudget {
    pub fn new(limits: MaterializationLimits) -> Self {
        Self {
            limits,
            usage: MaterializationUsage::default(),
        }
    }

    pub fn usage(&self) -> MaterializationUsage {
        self.usage
    }

    /// Charge one expansion atomically before its allocation or emission.
    pub fn charge(
        &mut self,
        payload_bytes: u64,
        ir_items: u64,
        temp_slots: u64,
    ) -> Result<(), MaterializationRefusal> {
        let payload = self
            .usage
            .payload_bytes
            .checked_add(payload_bytes)
            .ok_or(MaterializationRefusal::CounterOverflow)?;
        let items = self
            .usage
            .ir_items
            .checked_add(ir_items)
            .ok_or(MaterializationRefusal::CounterOverflow)?;
        let slots = self
            .usage
            .temp_slots
            .checked_add(temp_slots)
            .ok_or(MaterializationRefusal::CounterOverflow)?;
        if payload > self.limits.payload_bytes {
            return Err(MaterializationRefusal::PayloadLimit {
                attempted: payload,
                limit: self.limits.payload_bytes,
            });
        }
        if items > self.limits.ir_items {
            return Err(MaterializationRefusal::IrItemsLimit {
                attempted: items,
                limit: self.limits.ir_items,
            });
        }
        if slots > self.limits.temp_slots {
            return Err(MaterializationRefusal::TempSlotsLimit {
                attempted: slots,
                limit: self.limits.temp_slots,
            });
        }
        self.usage = MaterializationUsage {
            payload_bytes: payload,
            ir_items: items,
            temp_slots: slots,
        };
        Ok(())
    }

    /// Shared cost helper for the future repeat-literal lowering path.
    pub fn charge_repeat(
        &mut self,
        count: u32,
        bytes_per_element: u64,
        ir_items_per_element: u64,
        temp_slots_per_element: u64,
    ) -> Result<(), MaterializationRefusal> {
        let count = u64::from(count);
        self.charge(
            bytes_per_element
                .checked_mul(count)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            ir_items_per_element
                .checked_mul(count)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            temp_slots_per_element
                .checked_mul(count)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
        )
    }

    pub fn charge_fixed_literal(
        &mut self,
        length: u32,
        runtime_elements: bool,
    ) -> Result<(), MaterializationRefusal> {
        let n = u64::from(length);
        self.charge(
            n.checked_mul(8)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            if runtime_elements {
                n.checked_mul(2)
                    .and_then(|items| items.checked_add(1))
                    .ok_or(MaterializationRefusal::CounterOverflow)?
            } else {
                1
            },
            n,
        )
    }

    pub fn charge_static_elements(
        &mut self,
        length: usize,
        runtime_elements: bool,
    ) -> Result<(), MaterializationRefusal> {
        let n = u64::try_from(length).map_err(|_| MaterializationRefusal::CounterOverflow)?;
        self.charge(
            n.checked_mul(8)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            if runtime_elements {
                n.checked_mul(2)
                    .ok_or(MaterializationRefusal::CounterOverflow)?
            } else {
                1
            },
            n,
        )
    }

    pub fn charge_vec_literal(&mut self, length: usize) -> Result<(), MaterializationRefusal> {
        let n = u64::try_from(length).map_err(|_| MaterializationRefusal::CounterOverflow)?;
        self.charge(
            0,
            n.checked_add(1)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            0,
        )
    }

    /// Charge one compiler-generated inline fixed-array field store. The
    /// formula counts the canonical scalar-cell instructions emitted per
    /// element; a zero-length field has no element work.
    pub fn charge_fixed_field_store(
        &mut self,
        length: u32,
        f64_bits: bool,
    ) -> Result<(), MaterializationRefusal> {
        let n = u64::from(length);
        let per_element = if f64_bits { 6 } else { 5 };
        let items = if n == 0 {
            0
        } else {
            n.checked_mul(per_element)
                .and_then(|v| v.checked_sub(2))
                .ok_or(MaterializationRefusal::CounterOverflow)?
        };
        self.charge(
            n.checked_mul(8)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            items,
            n,
        )
    }

    /// Charge the scaffold plus element loads for a whole fixed-array field
    /// copyout. Indexed scalar reads intentionally use the O(1) path and do
    /// not call this helper.
    pub fn charge_fixed_field_copyout(
        &mut self,
        length: u32,
        f64_bits: bool,
    ) -> Result<(), MaterializationRefusal> {
        let n = u64::from(length);
        let per_element = if f64_bits { 6 } else { 5 };
        let body = if n == 0 {
            0
        } else {
            n.checked_mul(per_element)
                .and_then(|v| v.checked_sub(2))
                .ok_or(MaterializationRefusal::CounterOverflow)?
        };
        self.charge(
            n.checked_mul(8)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            body.checked_add(1)
                .ok_or(MaterializationRefusal::CounterOverflow)?,
            n,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_boundary_and_one_over_are_atomic() {
        let mut b = MaterializationBudget::new(MaterializationLimits {
            payload_bytes: 8,
            ir_items: 3,
            temp_slots: 2,
        });
        b.charge(8, 3, 2).expect("exact boundary");
        let before = b.usage();
        assert_eq!(
            b.charge(1, 0, 0),
            Err(MaterializationRefusal::PayloadLimit {
                attempted: 9,
                limit: 8
            })
        );
        assert_eq!(b.usage(), before);
    }

    #[test]
    fn checked_repeat_and_counter_overflow_refuse() {
        let mut b = MaterializationBudget::new(MaterializationLimits {
            payload_bytes: u64::MAX,
            ir_items: u64::MAX,
            temp_slots: u64::MAX,
        });
        b.charge(u64::MAX, 0, 0).expect("boundary");
        assert_eq!(
            b.charge(1, 0, 0),
            Err(MaterializationRefusal::CounterOverflow)
        );
        let mut repeat = MaterializationBudget::new(MaterializationLimits {
            payload_bytes: u64::MAX,
            ir_items: u64::MAX,
            temp_slots: u64::MAX,
        });
        assert_eq!(
            repeat.charge_repeat(u32::MAX, u64::MAX, 0, 0),
            Err(MaterializationRefusal::CounterOverflow)
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn lowering_refuses_before_fixed_array_payload_expansion() {
        let module = crate::parser::parse("let xs: [i64; 2] = [1, 2]").expect("fixed-array source");
        let result = crate::eval::lower::lower_to_ir_with_limits(
            &module,
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 10,
                temp_slots: 10,
            },
        );
        assert!(matches!(
            result,
            Err(MaterializationRefusal::PayloadLimit {
                attempted: 16,
                limit: 15,
            })
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn lowering_refuses_cumulative_generated_vec_work() {
        let module = crate::parser::parse("let a: array<i64> = [1, 2]\nlet b: array<i64> = [3, 4]")
            .expect("dynamic-array source");
        let result = crate::eval::lower::lower_to_ir_with_limits(
            &module,
            MaterializationLimits {
                payload_bytes: 1_000,
                ir_items: 5,
                temp_slots: 10,
            },
        );
        assert!(matches!(
            result,
            Err(MaterializationRefusal::IrItemsLimit {
                attempted: 6,
                limit: 5,
            })
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn lowering_accepts_zero_length_and_typed_float_boundaries() {
        let empty = crate::parser::parse("let xs: [i64; 0] = []").expect("empty source");
        crate::eval::lower::lower_to_ir_with_limits(
            &empty,
            MaterializationLimits {
                payload_bytes: 0,
                ir_items: 1,
                temp_slots: 0,
            },
        )
        .expect("zero-length arrays have no payload or temporary slots");

        let floats = crate::parser::parse("let xs: [f64; 2] = [1, -2]").expect("float source");
        crate::eval::lower::lower_to_ir_with_limits(
            &floats,
            MaterializationLimits {
                payload_bytes: 16,
                ir_items: 1,
                temp_slots: 2,
            },
        )
        .expect("typed float boundary");
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn nested_struct_array_receiver_refuses_without_lowering_panic() {
        let module = crate::parser::parse(
            "struct Item { value: i64 }\n\
             struct Batch { items: [Item; 2] }\n\
             fn read(b: Batch) -> i64 {\n\
                 let mut i: i64 = 0\n\
                 let mut total: i64 = 0\n\
                 while i < 2 {\n\
                     total = total + b.items[i].value\n\
                     i = i + 1\n\
                 }\n\
                 return total\n\
             }",
        )
        .expect("generic nested struct-array source");
        let result =
            crate::eval::lower::lower_to_ir_with_limits(&module, MaterializationLimits::default());
        match result {
            Err(MaterializationRefusal::UnsupportedLoweringOperation {
                operation,
                start: 160,
                end: 176,
            }) => assert_eq!(operation, "struct field read"),
            other => panic!("unexpected refusal: {other:?}"),
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn unsupported_struct_array_refusal_does_not_depend_on_budget_capacity() {
        let module = crate::parser::parse(
            "struct Item { value: i64 }\n\
             struct Batch { items: [Item; 2] }\n\
             fn read(b: Batch) -> i64 { return b.items[0].value }",
        )
        .expect("generic nested struct-array source");
        let result = crate::eval::lower::lower_to_ir_with_limits(
            &module,
            MaterializationLimits {
                payload_bytes: u64::MAX,
                ir_items: u64::MAX,
                temp_slots: u64::MAX,
            },
        );
        assert!(matches!(
            result,
            Err(MaterializationRefusal::UnsupportedLoweringOperation {
                operation,
                ..
            }) if operation == "struct field read"
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn public_pipeline_surfaces_materialization_refusal() {
        let result = crate::pipeline::compile_source_with_limits(
            "let xs: [i64; 2] = [1, 2]",
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 10,
                temp_slots: 10,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 16,
                    limit: 15,
                }
            ))
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn public_budget_is_shared_across_functions_and_branches() {
        let functions = r#"
fn first() -> [i64; 2] { return [1, 2] }
fn second() -> [i64; 2] { return [3, 4] }
"#;
        let result = crate::pipeline::compile_source_with_limits(
            functions,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 16,
                ir_items: 10,
                temp_slots: 10,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 32,
                    limit: 16,
                }
            ))
        ));

        let branch = r#"
fn branch(flag: i64) -> i64 {
    if flag == 0 { let a: [i64; 2] = [1, 2] }
    else { let b: [i64; 2] = [3, 4] }
    return flag
}
"#;
        let result = crate::pipeline::compile_source_with_limits(
            branch,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 16,
                ir_items: 10,
                temp_slots: 10,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 32,
                    limit: 16,
                }
            ))
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn public_field_owner_aliases_resolve_by_struct_and_index_is_constant_work() {
        let source = r#"
type LeftWords = [i64; 2]
type RightWords = [i64; 3]
struct Left { xs: LeftWords }
struct Right { xs: RightWords }
fn left(s: Left) -> i64 { return s.xs[1] }
fn right(s: Right) -> i64 { return s.xs[2] }
"#;
        let result = crate::pipeline::compile_source_with_limits(
            source,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 0,
                ir_items: 100,
                temp_slots: 0,
            },
        );
        assert!(
            result.is_ok(),
            "indexed field aliases must not copy out arrays: {result:?}"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn public_whole_field_copyout_is_charged_and_refusal_is_fail_closed() {
        let source = r#"
struct Pair { xs: [i64; 2] }
fn copyout(s: Pair) -> i64 {
    let xs = s.xs
    return xs[0]
}
"#;
        let result = crate::pipeline::compile_source_with_limits(
            source,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 100,
                temp_slots: 100,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 16,
                    limit: 15,
                }
            ))
        ));

        // The first failed charge is preserved even when later work would
        // exceed a different counter; lowering returns no partial IR product.
        let two = "let a: [i64; 2] = [1, 2]\nlet b: [i64; 2] = [3, 4]";
        let result = crate::pipeline::compile_source_with_limits(
            two,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 0,
                temp_slots: 100,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 16,
                    limit: 15,
                }
            ))
        ));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn refusal_stops_later_lowering_assumptions_and_emits_no_partial_ir() {
        // ExternConst is deliberately a loud lowering refusal today.  It is
        // placed after an over-budget literal to prove the first structured
        // materialisation error stops the item walk instead of reaching that
        // later assumption or returning a partial module.
        let source = "let xs: [i64; 2] = [1, 2]\nextern const TABLE: [i64; 1]";
        let module = crate::parser::parse(source).expect("refusal control source");
        let result = crate::eval::lower::lower_to_ir_with_limits(
            &module,
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 0,
                temp_slots: 0,
            },
        );
        assert!(matches!(
            result,
            Err(MaterializationRefusal::PayloadLimit {
                attempted: 16,
                limit: 15,
            })
        ));

        // A later direct fixed-field mutation must not be reached after the
        // first refusal.  This exercises the public pipeline path rather than
        // only the top-level item guard above.
        let later_mutation = r#"
struct Pair { xs: [i64; 2] }
fn run(p: Pair) -> i64 {
    let over: [i64; 2] = [1, 2]
    p.xs[0] = 9
    return 0
}
"#;
        let result = crate::pipeline::compile_source_with_limits(
            later_mutation,
            &crate::pipeline::CompileOptions::default(),
            MaterializationLimits {
                payload_bytes: 15,
                ir_items: 100,
                temp_slots: 100,
            },
        );
        assert!(matches!(
            result,
            Err(crate::pipeline::CompileError::Materialization(
                MaterializationRefusal::PayloadLimit {
                    attempted: 16,
                    limit: 15,
                }
            ))
        ));
    }
}
