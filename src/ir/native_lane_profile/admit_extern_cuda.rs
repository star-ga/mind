// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! The CUDA driver rows of the closed `nbx_row` table (ruling 16b).
//!
//! The first thirteen are the `libcuda.so.1` entries `nbx_row` appends after
//! the glibc rows: the unversioned `_v2` driver names plus the plain entries,
//! enough for a native-lane program to initialise the driver, load PTX,
//! launch a kernel and read the result back through the CUDA driver API.
//! They carry no glibc version, so `main.mind`'s writer gives them no
//! Verneed bucket and a program importing only these rows emits no
//! DT_VERSYM/DT_VERNEED/DT_VERNEEDNUM at all.
//!
//! `cuLaunchKernel` is the widest row (11 parameters, the arity cap of
//! `admit_extern_fn` and `nbx_row_sig`): parameters 6..10 travel on the SysV
//! stack. The ninth argument is `CUstream` (`*mut u8`; a null pointer is the
//! default stream), the tenth is the `void **` kernel-parameter array, and
//! the eleventh is `void **extra`.
//!
//! The second block (indices 54..=86, MC-6) is the cuda.h 12.6 macro targets
//! for the legacy default stream: a `_v2` name wherever cuda.h maps the plain
//! name to it, never a `_ptsz`/`_ptds` name. The plain legacy names keep their
//! old argument layout in the driver, so they stay off the table, as do
//! `cuGetProcAddress` (an indirect call) and `cuMemAllocManaged`; each refuses
//! by name as `lane.extern.unknown_symbol`. `cuMemsetD8_v2`'s `unsigned char`
//! is `u32` here; the self-host admits that argument only as a literal or a
//! masked value in 0..=255. Mirrors `nbcu_row` in
//! `examples/mindc_mind/nbx_cuda_rows2.mind`.

use super::admit_extern::AbiTy::{
    I32, I64, PtrConstPtrConstU8, PtrConstU8, PtrMutI32, PtrMutI64, PtrMutPtrMutU8, PtrMutU8, U32,
};
use super::admit_extern::{AbiTy, Row};

/// The `libcuda.so.1` rows, in `nbx_row` order (indices 41..=86). The first
/// thirteen mirror the `// GPU loader` block of `nbx_row` in
/// `examples/mindc_mind/main.mind`; the rest mirror `nbcu_row`.
#[rustfmt::skip]
pub(super) const ROWS: &[Row] = &[
    Row {
        name: "cuInit",
        params: &[AbiTy::U32],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuDeviceGet",
        params: &[AbiTy::PtrMutI32, AbiTy::I32],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuCtxCreate_v2",
        params: &[AbiTy::PtrMutPtrMutU8, AbiTy::U32, AbiTy::I32],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuCtxDestroy_v2",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuCtxSynchronize",
        params: &[],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuModuleLoadData",
        params: &[AbiTy::PtrMutPtrMutU8, AbiTy::PtrConstU8],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuModuleGetFunction",
        params: &[AbiTy::PtrMutPtrMutU8, AbiTy::PtrMutU8, AbiTy::PtrConstU8],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuModuleUnload",
        params: &[AbiTy::PtrMutU8],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuMemAlloc_v2",
        params: &[AbiTy::PtrMutI64, AbiTy::I64],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuMemcpyHtoD_v2",
        params: &[AbiTy::I64, AbiTy::PtrConstU8, AbiTy::I64],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuMemcpyDtoH_v2",
        params: &[AbiTy::PtrMutU8, AbiTy::I64, AbiTy::I64],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuMemFree_v2",
        params: &[AbiTy::I64],
        result: AbiTy::I32,
        version: "",
    },
    Row {
        name: "cuLaunchKernel",
        params: &[
            AbiTy::PtrMutU8,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::U32,
            AbiTy::PtrMutU8,
            AbiTy::PtrMutPtrMutU8,
            AbiTy::PtrMutPtrMutU8,
        ],
        result: AbiTy::I32,
        version: "",
    },
    // MC-6: the cuda.h 12.6 macro targets, nbcu_row indices 54..=86.
    Row { name: "cuStreamBeginCapture_v2", params: &[PtrMutU8, I32], result: I32, version: "" },
    Row { name: "cuGraphInstantiateWithFlags", result: I32, version: "",
          params: &[PtrMutPtrMutU8, PtrMutU8, I64] },
    Row { name: "cuMemcpyHtoDAsync_v2", result: I32, version: "",
          params: &[I64, PtrConstU8, I64, PtrMutU8] },
    Row { name: "cuMemcpyDtoHAsync_v2", result: I32, version: "",
          params: &[PtrMutU8, I64, I64, PtrMutU8] },
    Row { name: "cuEventDestroy_v2", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuStreamDestroy_v2", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuMemHostRegister_v2", params: &[PtrMutU8, I64, U32], result: I32, version: "" },
    Row { name: "cuMemAllocHost_v2", params: &[PtrMutPtrMutU8, I64], result: I32, version: "" },
    Row { name: "cuGraphAddKernelNode_v2", result: I32, version: "",
          params: &[PtrMutPtrMutU8, PtrMutU8, PtrConstPtrConstU8, I64, PtrConstU8] },
    Row { name: "cuGraphExecUpdate_v2", result: I32, version: "",
          params: &[PtrMutU8, PtrMutU8, PtrMutU8] },
    Row { name: "cuMemsetD8_v2", params: &[I64, U32, I64], result: I32, version: "" },
    Row { name: "cuStreamGetCaptureInfo_v2", result: I32, version: "",
          params: &[PtrMutU8, PtrMutI32, PtrMutI64, PtrMutPtrMutU8, PtrMutPtrMutU8, PtrMutI64] },
    Row { name: "cuDeviceGetAttribute", params: &[PtrMutI32, I32, I32], result: I32, version: "" },
    Row { name: "cuStreamCreate", params: &[PtrMutPtrMutU8, U32], result: I32, version: "" },
    Row { name: "cuStreamSynchronize", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuMemFreeHost", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuEventCreate", params: &[PtrMutPtrMutU8, U32], result: I32, version: "" },
    Row { name: "cuEventRecord", params: &[PtrMutU8, PtrMutU8], result: I32, version: "" },
    Row { name: "cuEventSynchronize", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuEventElapsedTime", result: I32, version: "",
          params: &[PtrMutI32, PtrMutU8, PtrMutU8] },
    Row { name: "cuFuncGetAttribute", result: I32, version: "",
          params: &[PtrMutI32, I32, PtrMutU8] },
    Row { name: "cuOccupancyMaxActiveBlocksPerMultiprocessor", result: I32, version: "",
          params: &[PtrMutI32, PtrMutU8, I32, I64] },
    Row { name: "cuLaunchCooperativeKernel", result: I32, version: "",
          params: &[PtrMutU8, U32, U32, U32, U32, U32, U32, U32, PtrMutU8, PtrMutPtrMutU8] },
    Row { name: "cuGraphDestroy", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuGraphExecDestroy", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuGraphLaunch", params: &[PtrMutU8, PtrMutU8], result: I32, version: "" },
    Row { name: "cuMemcpy2D_v2", params: &[PtrConstU8], result: I32, version: "" },
    Row { name: "cuMemcpy2DAsync_v2", params: &[PtrConstU8, PtrMutU8], result: I32, version: "" },
    Row { name: "cuMemHostAlloc", params: &[PtrMutPtrMutU8, I64, U32], result: I32, version: "" },
    Row { name: "cuMemHostGetDevicePointer_v2", result: I32, version: "",
          params: &[PtrMutI64, PtrMutU8, U32] },
    Row { name: "cuStreamEndCapture", result: I32, version: "",
          params: &[PtrMutU8, PtrMutPtrMutU8] },
    Row { name: "cuStreamQuery", params: &[PtrMutU8], result: I32, version: "" },
    Row { name: "cuEventQuery", params: &[PtrMutU8], result: I32, version: "" },
];

#[cfg(test)]
mod tests {
    use super::super::admit_extern::{AbiTy, admit_extern_fn};
    use super::super::types::Ty;
    use super::*;
    use crate::ast::{CallConv, ExternFn, Param, Span, TypeAnn};

    fn param(name: &str, ty: TypeAnn) -> Param {
        Param {
            name: name.to_string(),
            ty,
            span: Span::new(0, 0),
        }
    }

    fn decl(name: &str, params: Vec<Param>, ret: Option<TypeAnn>) -> ExternFn {
        ExternFn {
            is_unsafe: false,
            name: name.to_string(),
            params,
            ret_type: ret,
            is_varargs: false,
            span: Span::new(0, 0),
        }
    }

    /// The source annotation a declaration writes for one ABI code.
    fn ann(p: AbiTy) -> TypeAnn {
        match p {
            AbiTy::I32 => TypeAnn::ScalarI32,
            AbiTy::U32 => TypeAnn::ScalarU32,
            AbiTy::I64 => TypeAnn::ScalarI64,
            AbiTy::PtrConstU8 => ptr(false, named("u8")),
            AbiTy::PtrMutU8 => ptr(true, named("u8")),
            AbiTy::PtrMutI32 => ptr(true, TypeAnn::ScalarI32),
            AbiTy::PtrConstPtrConstU8 => ptr(false, ptr(false, named("u8"))),
            AbiTy::PtrMutPtrMutU8 => ptr(true, ptr(true, named("u8"))),
            AbiTy::PtrMutI64 => ptr(true, TypeAnn::ScalarI64),
            AbiTy::Void => unreachable!("no driver row has a void parameter"),
        }
    }

    /// A declaration of `name` with `row`'s parameter list and an i32 result.
    fn decl_like(name: &str, row: &Row) -> ExternFn {
        let params = row
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| param(&format!("p{i}"), ann(*p)))
            .collect();
        decl(name, params, Some(TypeAnn::ScalarI32))
    }

    /// Every driver row admits with exactly its own spelling: the thirteen
    /// `nbx_row` appends and the 33 MC-6 rows of `nbcu_row`, no name twice and
    /// none a per-thread-stream variant.
    #[test]
    fn every_driver_row_admits_by_spelling() {
        assert_eq!(ROWS.len(), 46);
        for (i, row) in ROWS.iter().enumerate() {
            assert_eq!(row.result, AbiTy::I32, "{} returns CUresult", row.name);
            assert!(!row.name.ends_with("_ptsz") && !row.name.ends_with("_ptds"));
            assert!(
                ROWS[..i].iter().all(|r| r.name != row.name),
                "{} twice",
                row.name
            );
            let (name, sig) = admit_extern_fn(&decl_like(row.name, row), CallConv::C)
                .unwrap_or_else(|e| panic!("{} refused by its own row: {e:?}", row.name));
            assert_eq!(name, row.name);
            assert_eq!(sig.params.len(), row.params.len());
            assert_eq!(sig.result, Ty::I32);
        }
    }

    /// The names cuda.h 12.6 does NOT bind for the legacy default stream are
    /// refused by name: cuMemAllocManaged, cuGetProcAddress (both spellings),
    /// every per-thread-stream variant, and the plain legacy name of each `_v2`
    /// row (the driver keeps the old argument layout under it), each declared
    /// with the admitted row's own parameter list so only the name differs.
    #[test]
    fn off_table_driver_names_refuse_by_name() {
        let managed = decl(
            "cuMemAllocManaged",
            vec![
                param("dptr", ptr(true, TypeAnn::ScalarI64)),
                param("n", TypeAnn::ScalarI64),
                param("flags", TypeAnn::ScalarU32),
            ],
            Some(TypeAnn::ScalarI32),
        );
        let proc = decl(
            "cuGetProcAddress",
            vec![param("symbol", ptr(false, named("u8")))],
            Some(TypeAnn::ScalarI32),
        );
        let mut refused = vec![managed, proc.clone()];
        refused.push(ExternFn {
            name: "cuGetProcAddress_v2".to_string(),
            ..proc
        });
        for row in &ROWS[13..] {
            for suffix in ["_ptsz", "_ptds"] {
                refused.push(decl_like(&format!("{}{suffix}", row.name), row));
            }
            if let Some(plain) = row.name.strip_suffix("_v2") {
                refused.push(decl_like(plain, row));
            }
        }
        refused.push(decl_like("cuGraphInstantiate", &ROWS[14]));
        assert_eq!(refused.len(), 3 + 33 * 2 + 14 + 1);
        for f in &refused {
            let err = admit_extern_fn(f, CallConv::C).unwrap_err();
            assert_eq!(err.construct, "lane.extern.unknown_symbol", "{}", f.name);
        }
    }

    /// `cuMemsetD8_v2`'s `unsigned char` is the u32 code: an i32 or i64 byte
    /// parameter, or a fourth parameter, is refused.
    #[test]
    fn the_byte_argument_row_is_u32() {
        let row = ROWS.iter().find(|r| r.name == "cuMemsetD8_v2").unwrap();
        assert_eq!(row.params, &[I64, U32, I64]);
        for wrong in [TypeAnn::ScalarI32, TypeAnn::ScalarI64] {
            let mut f = decl_like(row.name, row);
            f.params[1] = param("uc", wrong);
            let err = admit_extern_fn(&f, CallConv::C).unwrap_err();
            assert_eq!(err.construct, "lane.extern.signature_mismatch");
        }
        let mut four = decl_like(row.name, row);
        four.params.push(param("extra", TypeAnn::ScalarI64));
        let err = admit_extern_fn(&four, CallConv::C).unwrap_err();
        assert_eq!(err.construct, "lane.extern.arity");
    }

    /// The 11-argument row admits; a twelfth parameter refuses by arity, and
    /// a wrong width on the widest row refuses as a signature mismatch.
    #[test]
    fn the_eleven_argument_launch_row_bounds_the_arity_cap() {
        let launch = decl(
            "cuLaunchKernel",
            vec![
                param("f", ptr(true, named("u8"))),
                param("gx", TypeAnn::ScalarU32),
                param("gy", TypeAnn::ScalarU32),
                param("gz", TypeAnn::ScalarU32),
                param("bx", TypeAnn::ScalarU32),
                param("by", TypeAnn::ScalarU32),
                param("bz", TypeAnn::ScalarU32),
                param("shm", TypeAnn::ScalarU32),
                param("stream", ptr(true, named("u8"))),
                param("kp", ptr(true, ptr(true, named("u8")))),
                param("ex", ptr(true, ptr(true, named("u8")))),
            ],
            Some(TypeAnn::ScalarI32),
        );
        assert!(admit_extern_fn(&launch, CallConv::C).is_ok());
        let mut twelve = launch.clone();
        twelve
            .params
            .push(param("one_too_many", TypeAnn::ScalarI64));
        let err = admit_extern_fn(&twelve, CallConv::C).unwrap_err();
        assert_eq!(err.construct, "lane.extern.arity");
        let mut wrong = launch;
        wrong.params[7] = param("shm", TypeAnn::ScalarI64);
        let err = admit_extern_fn(&wrong, CallConv::C).unwrap_err();
        assert_eq!(err.construct, "lane.extern.signature_mismatch");
    }

    fn named(n: &str) -> TypeAnn {
        TypeAnn::Named(n.to_string())
    }

    fn ptr(mutable: bool, pointee: TypeAnn) -> TypeAnn {
        TypeAnn::RawPtr {
            mutable,
            pointee: Box::new(pointee),
        }
    }
}
