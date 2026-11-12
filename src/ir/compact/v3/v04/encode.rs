// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{IRModule, Instr, ValueId, verify_canonical_metadata};
use crate::types::{
    CanonicalModuleTypes, FunctionIdentity, FunctionKind, ScalarType, SchemaRegistry, SemanticType,
};

use super::{
    DESCRIPTOR_CHARGE, FIELD_CHARGE, FUNCTION_CHARGE, FUNCTION_EXTERNAL, FUNCTION_INTRINSIC,
    FUNCTION_LOCAL, INSTRUCTION_CHARGE, MAX_INSTR_DEPTH, NAMED_VALUE_CHARGE, PARAMETER_CHARGE,
    REQUIRED_STD_SURFACE, SCALAR_BF16, SCALAR_BOOL, SCALAR_F16, SCALAR_F32, SCALAR_F64, SCALAR_I32,
    SCALAR_I64, SCALAR_Q16, SCALAR_U32, SCHEMA_CHARGE, STRING_REFERENCE_MULTIPLIER,
    STRING_SLOT_CHARGE, TYPE_DYNAMIC_ARRAY, TYPE_FIXED_ARRAY, TYPE_RECORD_REF, TYPE_SCALAR,
    VALUE_ID_CHARGE, VALUE_ROW_CHARGE, decoded_budget_limit, validate_core_value_ids,
};
use crate::ir::compact::v2::zigzag_encode;
use crate::ir::compact::v3::emit::{
    OP_BINOP, OP_CALL, OP_CONST_F64, OP_CONST_I64, OP_FN_DEF, OP_OUTPUT, OP_PARAM, OP_RETURN,
    binop_to_byte,
};
use crate::ir::compact::v3::{MAX_MIC3_INPUT, MIC3_MAGIC, MIC3_VERSION_V04, Mic3EncodeError};

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn put(&mut self, bytes: &[u8]) -> Result<(), Mic3EncodeError> {
        let next = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
            Mic3EncodeError::V04ResourceLimit("encoded size overflow".to_string())
        })?;
        if next > MAX_MIC3_INPUT {
            return Err(Mic3EncodeError::V04ResourceLimit(format!(
                "encoded body exceeds {MAX_MIC3_INPUT} bytes"
            )));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn byte(&mut self, byte: u8) -> Result<(), Mic3EncodeError> {
        self.put(&[byte])
    }

    fn uleb(&mut self, mut value: u64) -> Result<(), Mic3EncodeError> {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            self.byte(byte)?;
            if value == 0 {
                return Ok(());
            }
        }
    }

    fn count(&mut self, value: usize) -> Result<(), Mic3EncodeError> {
        self.uleb(
            u64::try_from(value).map_err(|_| {
                Mic3EncodeError::V04ResourceLimit("count does not fit u64".to_string())
            })?,
        )
    }

    fn vid(&mut self, value: ValueId) -> Result<(), Mic3EncodeError> {
        self.count(value.0)
    }

    fn opt_vid(&mut self, value: Option<ValueId>) -> Result<(), Mic3EncodeError> {
        match value {
            None => self.byte(0),
            Some(value) => {
                self.byte(1)?;
                self.vid(value)
            }
        }
    }

    fn opt_f64(&mut self, value: Option<f64>) -> Result<(), Mic3EncodeError> {
        match value {
            None => self.byte(0),
            Some(value) => {
                self.byte(1)?;
                self.put(&value.to_bits().to_le_bytes())
            }
        }
    }
}

struct StringTable {
    entries: Vec<String>,
    indices: BTreeMap<String, usize>,
}

impl StringTable {
    fn from_set(strings: BTreeSet<String>) -> Self {
        let entries: Vec<String> = strings.into_iter().collect();
        let indices = entries
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, value)| (value, index))
            .collect();
        Self { entries, indices }
    }

    fn index(&self, value: &str) -> Result<usize, Mic3EncodeError> {
        self.indices.get(value).copied().ok_or_else(|| {
            Mic3EncodeError::InvalidCanonicalMetadata(format!(
                "v0x04 string was not collected: {value}"
            ))
        })
    }
}

pub(crate) fn emit_v04(module: &IRModule) -> Result<Vec<u8>, Mic3EncodeError> {
    validate_core_value_ids(module).map_err(Mic3EncodeError::InvalidCanonicalMetadata)?;
    let requirements = surface_requirements(module)?;
    if requirements.unsupported_core {
        return Err(Mic3EncodeError::UnsupportedV04Surface(
            "core instruction outside the scalar stage".to_string(),
        ));
    }
    if requirements.bits != 0 {
        return Err(Mic3EncodeError::UnsupportedV04Surface(format!(
            "required_surface_bits 0x{:x} needs the unfinished std codec",
            requirements.bits
        )));
    }
    verify_canonical_metadata(module)
        .map_err(|error| Mic3EncodeError::InvalidCanonicalMetadata(error.to_string()))?;
    let bundle = module.canonical_types.as_deref().ok_or_else(|| {
        Mic3EncodeError::InvalidCanonicalMetadata("v0x04 requires a canonical bundle".to_string())
    })?;
    if bundle.is_empty()
        && !crate::ir::canonical_verify::instruction_metadata_present(&module.instrs)
    {
        return Err(Mic3EncodeError::InvalidCanonicalMetadata(
            "v0x04 requires nonempty semantic authority".to_string(),
        ));
    }
    let required_surface = requirements.bits;

    let table = StringTable::from_set(collect_strings(module, bundle)?);
    let function_indices: BTreeMap<FunctionIdentity, usize> = bundle
        .functions()
        .keys()
        .cloned()
        .enumerate()
        .map(|(index, identity)| (identity, index))
        .collect();
    let mut out = Encoder::new();
    out.put(&MIC3_MAGIC)?;
    out.byte(MIC3_VERSION_V04)?;
    out.uleb(required_surface)?;

    out.count(table.entries.len())?;
    for string in &table.entries {
        out.count(string.len())?;
        out.put(string.as_bytes())?;
    }

    let registry = bundle.schema_registry();
    out.count(registry.schemas().len())?;
    for schema in registry.schemas() {
        out.count(table.index(schema.identity().owner())?)?;
        out.count(table.index(schema.identity().name())?)?;
    }
    for schema in registry.schemas() {
        out.count(schema.fields().len())?;
        for field in schema.fields() {
            out.count(table.index(field.name())?)?;
            put_type(&mut out, field.ty(), registry)?;
        }
    }

    out.count(bundle.functions().len())?;
    for declaration in bundle.functions().values() {
        out.count(table.index(declaration.identity().owner())?)?;
        out.count(table.index(declaration.identity().name())?)?;
        out.byte(match declaration.kind() {
            FunctionKind::Local => FUNCTION_LOCAL,
            FunctionKind::External => FUNCTION_EXTERNAL,
            FunctionKind::Intrinsic => FUNCTION_INTRINSIC,
        })?;
        out.count(declaration.signature().params().len())?;
        for parameter in declaration.signature().params() {
            put_type(&mut out, parameter, registry)?;
        }
        match declaration.signature().return_type() {
            None => out.byte(0)?,
            Some(return_type) => {
                out.byte(1)?;
                put_type(&mut out, return_type, registry)?;
            }
        }
    }

    out.count(module.next_id)?;
    let mut exports: Vec<&str> = module.exports.iter().map(String::as_str).collect();
    exports.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    out.count(exports.len())?;
    for export in exports {
        out.count(table.index(export)?)?;
    }
    out.count(module.instrs.len())?;
    for instruction in &module.instrs {
        put_instr(
            &mut out,
            instruction,
            &table,
            registry,
            &function_indices,
            0,
        )?;
    }

    // Fixed compatibility framing. The core stage rejects every populated
    // std-surface section but still writes each count in every build profile.
    for _ in 0..4 {
        out.uleb(0)?;
    }
    out.count(bundle.module_values().len())?;
    for (value, semantic_type) in bundle.module_values() {
        out.vid(*value)?;
        put_type(&mut out, semantic_type, registry)?;
    }
    enforce_decode_budget(module, &table.entries, out.bytes.len())?;
    Ok(out.bytes)
}

struct SurfaceRequirements {
    bits: u64,
    unsupported_core: bool,
}

fn surface_requirements(module: &IRModule) -> Result<SurfaceRequirements, Mic3EncodeError> {
    let mut requirements = SurfaceRequirements {
        bits: 0,
        unsupported_core: false,
    };
    #[cfg(feature = "std-surface")]
    if !module.struct_defs.is_empty()
        || !module.const_array_defs.is_empty()
        || !module.repr_c_structs.is_empty()
        || !module.value_types.is_empty()
    {
        requirements.bits |= REQUIRED_STD_SURFACE;
    }
    classify_instrs(&module.instrs, &mut requirements, 0)?;
    Ok(requirements)
}

fn classify_instrs(
    instructions: &[Instr],
    requirements: &mut SurfaceRequirements,
    depth: usize,
) -> Result<(), Mic3EncodeError> {
    if depth >= MAX_INSTR_DEPTH && !instructions.is_empty() {
        return Err(Mic3EncodeError::V04ResourceLimit(
            "instruction nesting exceeds depth 256".to_string(),
        ));
    }
    for instruction in instructions {
        match instruction {
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::Output(..)
            | Instr::Return { .. }
            | Instr::Param { .. } => {}
            Instr::BinOp { op, .. } => {
                if binop_to_byte(*op) > 0x0a {
                    requirements.bits |= REQUIRED_STD_SURFACE;
                }
            }
            Instr::FnDef { body, .. } => {
                #[cfg(feature = "std-surface")]
                if let Instr::FnDef { value_types, .. } = instruction {
                    if !value_types.is_empty() {
                        requirements.bits |= REQUIRED_STD_SURFACE;
                    }
                }
                classify_instrs(body, requirements, depth + 1)?;
            }
            Instr::Call { .. } => {}
            Instr::ConstTensor(..)
            | Instr::ConstDenseTensor { .. }
            | Instr::Sum { .. }
            | Instr::Mean { .. }
            | Instr::Relu { .. }
            | Instr::ReluGrad { .. }
            | Instr::Reshape { .. }
            | Instr::ExpandDims { .. }
            | Instr::Squeeze { .. }
            | Instr::Transpose { .. }
            | Instr::Dot { .. }
            | Instr::MatMul { .. }
            | Instr::Conv2d { .. }
            | Instr::Conv2dGradInput { .. }
            | Instr::Conv2dGradFilter { .. }
            | Instr::Index { .. }
            | Instr::Slice { .. }
            | Instr::Gather { .. }
            | Instr::SparseAttr { .. } => requirements.unsupported_core = true,
            // The extern binding record (`OP_EXTERN_BIND`) has no v0x04 row yet.
            #[cfg(feature = "std-surface")]
            Instr::ExternFnDecl { binding, .. } if *binding != Default::default() => {
                return Err(Mic3EncodeError::UnsupportedV04Surface(
                    "extern binding record has no v0x04 row".to_string(),
                ));
            }
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { .. }
            | Instr::ArrayLoad { .. }
            | Instr::ArrayStore { .. }
            | Instr::While { .. }
            | Instr::If { .. }
            | Instr::VecLoad { .. }
            | Instr::VecFma { .. }
            | Instr::VecReduceAdd { .. }
            | Instr::VecStore { .. }
            | Instr::VecLoadI32 { .. }
            | Instr::VecMulAddQ16 { .. }
            | Instr::VecReduceAddI64 { .. }
            | Instr::Region { .. }
            | Instr::Break { .. }
            | Instr::Continue { .. }
            | Instr::ExternFnDecl { .. } => requirements.bits |= REQUIRED_STD_SURFACE,
        }
    }
    Ok(())
}

fn collect_strings(
    module: &IRModule,
    bundle: &CanonicalModuleTypes,
) -> Result<BTreeSet<String>, Mic3EncodeError> {
    let mut strings = BTreeSet::new();
    strings.extend(module.exports.iter().cloned());
    for schema in bundle.schema_registry().schemas() {
        strings.insert(schema.identity().owner().to_string());
        strings.insert(schema.identity().name().to_string());
        for field in schema.fields() {
            strings.insert(field.name().to_string());
        }
    }
    for identity in bundle.functions().keys() {
        strings.insert(identity.owner().to_string());
        strings.insert(identity.name().to_string());
    }
    collect_instr_strings(&module.instrs, &mut strings, 0)?;
    Ok(strings)
}

fn collect_instr_strings(
    instructions: &[Instr],
    strings: &mut BTreeSet<String>,
    depth: usize,
) -> Result<(), Mic3EncodeError> {
    if depth >= MAX_INSTR_DEPTH && !instructions.is_empty() {
        return Err(Mic3EncodeError::V04ResourceLimit(
            "instruction nesting exceeds depth 256".to_string(),
        ));
    }
    for instruction in instructions {
        match instruction {
            Instr::FnDef {
                name, params, body, ..
            } => {
                strings.insert(name.clone());
                strings.extend(params.iter().map(|(name, _)| name.clone()));
                collect_instr_strings(body, strings, depth + 1)?;
            }
            Instr::Call { name, .. } | Instr::Param { name, .. } => {
                strings.insert(name.clone());
            }
            _ => {}
        }
    }
    Ok(())
}

fn put_type(
    out: &mut Encoder,
    semantic_type: &SemanticType,
    registry: &SchemaRegistry,
) -> Result<(), Mic3EncodeError> {
    match semantic_type {
        SemanticType::Scalar(scalar) => {
            out.byte(TYPE_SCALAR)?;
            out.byte(match scalar {
                ScalarType::I32 => SCALAR_I32,
                ScalarType::I64 => SCALAR_I64,
                ScalarType::U32 => SCALAR_U32,
                ScalarType::F32 => SCALAR_F32,
                ScalarType::F64 => SCALAR_F64,
                ScalarType::Bool => SCALAR_BOOL,
                ScalarType::BF16 => SCALAR_BF16,
                ScalarType::F16 => SCALAR_F16,
                ScalarType::Q16 => SCALAR_Q16,
            })
        }
        SemanticType::RecordRef(identity) => {
            out.byte(TYPE_RECORD_REF)?;
            let schema = registry.schema_id(identity).ok_or_else(|| {
                Mic3EncodeError::InvalidCanonicalMetadata(format!(
                    "unknown schema reference {identity}"
                ))
            })?;
            out.count(schema.index())
        }
        SemanticType::FixedArray { element, extent } => {
            out.byte(TYPE_FIXED_ARRAY)?;
            out.uleb(*extent)?;
            put_type(out, element, registry)
        }
        SemanticType::DynamicArray { element } => {
            out.byte(TYPE_DYNAMIC_ARRAY)?;
            put_type(out, element, registry)
        }
    }
}

fn function_index(
    identity: &FunctionIdentity,
    indices: &BTreeMap<FunctionIdentity, usize>,
) -> Result<usize, Mic3EncodeError> {
    indices.get(identity).copied().ok_or_else(|| {
        Mic3EncodeError::InvalidCanonicalMetadata(format!(
            "unregistered function identity {identity}"
        ))
    })
}

fn put_instr(
    out: &mut Encoder,
    instruction: &Instr,
    strings: &StringTable,
    registry: &SchemaRegistry,
    functions: &BTreeMap<FunctionIdentity, usize>,
    depth: usize,
) -> Result<(), Mic3EncodeError> {
    if depth >= MAX_INSTR_DEPTH {
        return Err(Mic3EncodeError::V04ResourceLimit(
            "instruction nesting exceeds depth 256".to_string(),
        ));
    }
    match instruction {
        Instr::ConstI64(dst, value) => {
            out.byte(OP_CONST_I64)?;
            out.vid(*dst)?;
            out.uleb(zigzag_encode(*value))
        }
        Instr::ConstF64(dst, value) => {
            out.byte(OP_CONST_F64)?;
            out.vid(*dst)?;
            out.put(&value.to_bits().to_le_bytes())
        }
        Instr::BinOp { dst, op, lhs, rhs } => {
            out.byte(OP_BINOP)?;
            out.vid(*dst)?;
            out.byte(binop_to_byte(*op))?;
            out.vid(*lhs)?;
            out.vid(*rhs)
        }
        Instr::Output(value) => {
            out.byte(OP_OUTPUT)?;
            out.vid(*value)
        }
        Instr::FnDef {
            name,
            params,
            ret_id,
            body,
            reap_threshold,
            semantic_types,
            ..
        } => {
            out.byte(OP_FN_DEF)?;
            out.count(strings.index(name)?)?;
            out.count(params.len())?;
            for (name, value) in params {
                out.count(strings.index(name)?)?;
                out.vid(*value)?;
            }
            out.opt_vid(*ret_id)?;
            out.opt_f64(*reap_threshold)?;
            out.count(body.len())?;
            for nested in body {
                put_instr(out, nested, strings, registry, functions, depth + 1)?;
            }
            out.uleb(0)?; // always-present legacy function ArrayType count
            let semantic_types = semantic_types.as_deref().ok_or_else(|| {
                Mic3EncodeError::InvalidCanonicalMetadata(format!(
                    "function {name} lacks v0x04 semantic metadata"
                ))
            })?;
            out.count(function_index(semantic_types.identity(), functions)?)?;
            out.count(semantic_types.values().len())?;
            for (value, semantic_type) in semantic_types.values() {
                out.vid(*value)?;
                put_type(out, semantic_type, registry)?;
            }
            Ok(())
        }
        Instr::Call {
            dst,
            name,
            args,
            resolved_callee,
        } => {
            out.byte(OP_CALL)?;
            out.vid(*dst)?;
            out.count(strings.index(name)?)?;
            out.count(args.len())?;
            for argument in args {
                out.vid(*argument)?;
            }
            let identity = resolved_callee.as_deref().ok_or_else(|| {
                Mic3EncodeError::InvalidCanonicalMetadata(format!(
                    "call {name} lacks a resolved identity"
                ))
            })?;
            out.count(function_index(identity, functions)?)
        }
        Instr::Return { value } => {
            out.byte(OP_RETURN)?;
            out.opt_vid(*value)
        }
        Instr::Param { dst, name, index } => {
            out.byte(OP_PARAM)?;
            out.vid(*dst)?;
            out.count(strings.index(name)?)?;
            out.count(*index)
        }
        _ => Err(Mic3EncodeError::UnsupportedV04Surface(
            "instruction outside the core scalar subset".to_string(),
        )),
    }
}

struct LogicalCost {
    used: usize,
}

impl LogicalCost {
    fn add(&mut self, count: usize, unit: usize, label: &str) -> Result<(), Mic3EncodeError> {
        let amount = count.checked_mul(unit).ok_or_else(|| {
            Mic3EncodeError::V04ResourceLimit(format!("{label} logical charge overflow"))
        })?;
        self.used = self.used.checked_add(amount).ok_or_else(|| {
            Mic3EncodeError::V04ResourceLimit("decoded logical charge overflow".to_string())
        })?;
        Ok(())
    }

    fn string_ref(&mut self, value: &str) -> Result<(), Mic3EncodeError> {
        self.add(
            value.len(),
            STRING_REFERENCE_MULTIPLIER,
            "owned string reference",
        )
    }

    fn identity(&mut self, owner: &str, name: &str, label: &str) -> Result<(), Mic3EncodeError> {
        self.string_ref(owner)?;
        self.string_ref(name).map_err(|_| {
            Mic3EncodeError::V04ResourceLimit(format!("{label} projection charge overflow"))
        })
    }

    fn semantic_type(&mut self, ty: &SemanticType) -> Result<(), Mic3EncodeError> {
        self.add(1, DESCRIPTOR_CHARGE, "semantic descriptor")?;
        match ty {
            SemanticType::RecordRef(identity) => {
                self.identity(identity.owner(), identity.name(), "schema identity")
            }
            SemanticType::FixedArray { element, .. } | SemanticType::DynamicArray { element } => {
                self.semantic_type(element)
            }
            SemanticType::Scalar(_) => Ok(()),
        }
    }
}

fn enforce_decode_budget(
    module: &IRModule,
    strings: &[String],
    encoded_len: usize,
) -> Result<(), Mic3EncodeError> {
    let bundle = module.canonical_types.as_deref().ok_or_else(|| {
        Mic3EncodeError::InvalidCanonicalMetadata("v0x04 requires a canonical bundle".to_string())
    })?;
    let mut cost = LogicalCost { used: 0 };
    cost.add(strings.len(), STRING_SLOT_CHARGE, "string table")?;
    for string in strings {
        cost.add(string.len(), 1, "string bytes")?;
    }
    let schemas = bundle.schema_registry().schemas();
    cost.add(schemas.len(), SCHEMA_CHARGE, "schema headers")?;
    for schema in schemas {
        cost.identity(
            schema.identity().owner(),
            schema.identity().name(),
            "schema identity",
        )?;
    }
    for schema in schemas {
        cost.add(schema.fields().len(), FIELD_CHARGE, "schema fields")?;
        for field in schema.fields() {
            cost.string_ref(field.name())?;
            cost.semantic_type(field.ty())?;
        }
    }
    cost.add(
        bundle.functions().len(),
        FUNCTION_CHARGE,
        "function declarations",
    )?;
    for declaration in bundle.functions().values() {
        cost.identity(
            declaration.identity().owner(),
            declaration.identity().name(),
            "function identity",
        )?;
        cost.add(
            declaration.signature().params().len(),
            PARAMETER_CHARGE,
            "signature parameters",
        )?;
        for parameter in declaration.signature().params() {
            cost.semantic_type(parameter)?;
        }
        if let Some(return_type) = declaration.signature().return_type() {
            cost.semantic_type(return_type)?;
        }
    }
    cost.add(module.exports.len(), STRING_SLOT_CHARGE, "exports")?;
    for export in &module.exports {
        cost.string_ref(export)?;
    }
    cost_instrs(&module.instrs, &mut cost)?;
    cost.add(
        bundle.module_values().len(),
        VALUE_ROW_CHARGE,
        "module semantic values",
    )?;
    for semantic_type in bundle.module_values().values() {
        cost.semantic_type(semantic_type)?;
    }
    cost.add(encoded_len, 1, "canonical re-emission")?;
    let limit = decoded_budget_limit(encoded_len);
    if cost.used > limit {
        return Err(Mic3EncodeError::V04ResourceLimit(format!(
            "decoded allocation cost {} exceeds budget {limit}",
            cost.used
        )));
    }
    Ok(())
}

fn cost_instrs(instructions: &[Instr], cost: &mut LogicalCost) -> Result<(), Mic3EncodeError> {
    cost.add(instructions.len(), INSTRUCTION_CHARGE, "instruction vector")?;
    for instruction in instructions {
        cost.add(1, INSTRUCTION_CHARGE, "instruction")?;
        match instruction {
            Instr::FnDef {
                name,
                params,
                body,
                semantic_types,
                ..
            } => {
                cost.string_ref(name)?;
                cost.add(params.len(), NAMED_VALUE_CHARGE, "function parameters")?;
                for (name, _) in params {
                    cost.string_ref(name)?;
                }
                cost_instrs(body, cost)?;
                let semantic = semantic_types.as_deref().ok_or_else(|| {
                    Mic3EncodeError::InvalidCanonicalMetadata(format!(
                        "function {name} lacks v0x04 semantic metadata"
                    ))
                })?;
                cost.identity(
                    semantic.identity().owner(),
                    semantic.identity().name(),
                    "function identity",
                )?;
                cost.add(
                    semantic.values().len(),
                    VALUE_ROW_CHARGE,
                    "function semantic values",
                )?;
                for semantic_type in semantic.values().values() {
                    cost.semantic_type(semantic_type)?;
                }
            }
            Instr::Call {
                name,
                args,
                resolved_callee,
                ..
            } => {
                cost.string_ref(name)?;
                cost.add(args.len(), VALUE_ID_CHARGE, "call arguments")?;
                let identity = resolved_callee.as_deref().ok_or_else(|| {
                    Mic3EncodeError::InvalidCanonicalMetadata(format!(
                        "call {name} lacks a resolved identity"
                    ))
                })?;
                cost.identity(identity.owner(), identity.name(), "function identity")?;
            }
            Instr::Param { name, .. } => cost.string_ref(name)?,
            _ => {}
        }
    }
    Ok(())
}
