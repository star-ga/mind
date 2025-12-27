#![allow(dead_code)]

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

#[cfg(feature = "ffi-c")]
pub mod capi {
    use std::cell::RefCell;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::os::raw::c_int;
    use std::os::raw::c_void;

    use std::ptr;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MindDType {
        I32 = 0,
        F32 = 1,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MindShape {
        pub rank: u32,
        pub dims: *const u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MindTensor {
        pub dtype: MindDType,
        pub shape: MindShape,
        pub data: *mut c_void,
        pub byte_len: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MindIO {
        pub name: *const c_char,
        pub tensor: MindTensor,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MindModelMeta {
        pub inputs_len: u32,
        pub outputs_len: u32,
        pub model_name: *const c_char,
        pub model_version: u64,
    }

    thread_local! {
        static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    }

    const MODEL_NAME: &[u8] = b"mind\0";

    fn write_error(msg: impl Into<String>) -> c_int {
        let string =
            CString::new(msg.into()).unwrap_or_else(|_| CString::new("ffi error").unwrap());
        LAST_ERROR.with(|slot| {
            *slot.borrow_mut() = Some(string);
        });
        -1
    }

    fn clear_error() {
        LAST_ERROR.with(|slot| {
            slot.borrow_mut().take();
        });
    }

    /// # Safety
    ///
    /// `meta_out` must point to a valid, properly aligned `MindModelMeta`.
    #[no_mangle]
    pub unsafe extern "C" fn mind_model_meta(meta_out: *mut MindModelMeta) -> c_int {
        clear_error();
        if meta_out.is_null() {
            return write_error("meta_out is null");
        }

        unsafe {
            (*meta_out).inputs_len = 0;
            (*meta_out).outputs_len = 0;
            (*meta_out).model_name = MODEL_NAME.as_ptr() as *const c_char;
            (*meta_out).model_version = 0;
        }
        0
    }

    #[no_mangle]
    pub extern "C" fn mind_model_io(
        inputs_out: *mut MindIO,
        cap_inputs: u32,
        outputs_out: *mut MindIO,
        cap_outputs: u32,
    ) -> c_int {
        clear_error();
        if cap_inputs > 0 && inputs_out.is_null() {
            return write_error("inputs_out is null");
        }
        if cap_outputs > 0 && outputs_out.is_null() {
            return write_error("outputs_out is null");
        }
        let _ = cap_inputs;
        let _ = cap_outputs;
        0
    }

    #[no_mangle]
    pub extern "C" fn mind_infer(
        inputs: *const MindIO,
        _inputs_len: u32,
        _outputs: *mut MindIO,
        _outputs_len: u32,
    ) -> c_int {
        clear_error();
        if inputs.is_null() {
            return write_error("inputs array is null");
        }
        write_error("mind_infer is not available in this build")
    }

    /// Allocate `size` bytes of host memory.
    ///
    /// Returns null if `size` is zero or cannot fit in `usize`. Zero-sized
    /// requests return null without setting `LAST_ERROR`; overflow cases record
    /// an error so callers can retrieve the cause via `mind_last_error()`.
    #[no_mangle]
    pub extern "C" fn mind_alloc(size: u64) -> *mut c_void {
        if size == 0 {
            return ptr::null_mut();
        }
        // Reject requests that cannot fit in the platform pointer width to avoid
        // truncating the allocation size or returning a null pointer without
        // recording an error.
        if size >= usize::MAX as u64 {
            let _ = write_error("allocation size exceeds platform pointer width");
            return ptr::null_mut();
        }
        unsafe { libc::malloc(size as usize) }
    }

    /// # Safety
    ///
    /// `ptr` must be null or a pointer previously returned by `mind_alloc`.
    #[no_mangle]
    pub unsafe extern "C" fn mind_free(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            libc::free(ptr);
        }
    }

    #[no_mangle]
    pub extern "C" fn mind_last_error() -> *const c_char {
        LAST_ERROR.with(|slot| {
            if let Some(s) = slot.borrow().as_ref() {
                s.as_ptr()
            } else {
                ptr::null()
            }
        })
    }

    pub fn last_error_as_str() -> Option<String> {
        LAST_ERROR.with(|slot| {
            slot.borrow()
                .as_ref()
                .map(|s| s.to_string_lossy().into_owned())
        })
    }

    #[allow(dead_code)]
    pub fn clear_last_error_for_tests() {
        clear_error();
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn meta_reports_defaults() {
            let mut meta = MindModelMeta {
                inputs_len: 1,
                outputs_len: 1,
                model_name: std::ptr::null(),
                model_version: 1,
            };
            // SAFETY: meta is a valid, properly aligned MindModelMeta
            assert_eq!(
                unsafe { mind_model_meta(&mut meta as *mut MindModelMeta) },
                0
            );
            assert_eq!(meta.inputs_len, 0);
            assert_eq!(meta.outputs_len, 0);
            assert!(!meta.model_name.is_null());
        }

        #[test]
        fn infer_reports_error() {
            clear_last_error_for_tests();
            let rc = mind_infer(std::ptr::null(), 0, std::ptr::null_mut(), 0);
            assert!(rc < 0);
            assert!(last_error_as_str().is_some());
        }

        #[test]
        fn alloc_rejects_oversized_request() {
            clear_last_error_for_tests();
            let ptr = mind_alloc(u64::MAX);
            assert!(ptr.is_null());
            let err = last_error_as_str().expect("error should be recorded");
            assert!(err.contains("platform pointer width"));
        }
    }
}

#[cfg(feature = "ffi-c")]
pub mod header;
