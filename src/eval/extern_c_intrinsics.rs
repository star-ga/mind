// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Extern "C" libc intrinsics dispatched by the evaluator so `mindc test`
//! can execute file-reading MIND (`std.fs.read_to_string`) without the
//! native runtime. Matches the POSIX semantics the native backend links
//! in `runtime-support/mind_intrinsics.c` by symbol.

use super::interp_mem;

/// Is `name` a runtime-support extern-C call the evaluator dispatches?
pub(crate) fn is_extern_c_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "open" | "lseek" | "read" | "close" | "read_to_string" | "string_push_byte" | "vec_zeroed"
    )
}

/// Evaluate one extern "C" libc call over the thread-local arena.
///
/// Arguments are i64 scalars (the MIND ABI); pointer arguments are arena
/// offsets. Returns the POSIX return value as i64.
pub(crate) fn eval_extern_c_intrinsic(name: &str, args: &[i64]) -> Result<i64, String> {
    match (name, args) {
        // open(path_addr, flags, mode) -> fd (or -1 on error)
        ("open", [path_addr, _flags, _mode]) => {
            let path = read_c_string(*path_addr)?;
            match std::fs::File::open(&path) {
                Ok(_) => {
                    // Bump-allocate a fake fd from a thread-local counter.
                    Ok(alloc_fake_fd(&path))
                }
                Err(_) => Ok(-1),
            }
        }
        // lseek(fd, offset, whence) -> new position (or -1 on error)
        ("lseek", [fd, offset, whence]) => lseek_fake_fd(*fd, *offset, *whence),
        // read(fd, buf_addr, count) -> bytes read (or -1 on error)
        ("read", [fd, buf_addr, count]) => read_fake_fd(*fd, *buf_addr, *count),
        // close(fd) -> 0 (or -1 on error)
        ("close", [fd]) => close_fake_fd(*fd),
        // read_to_string(path_addr) -> string record (addr, len, cap)
        // Mirrors std/fs.mind's read_to_string: reads the file into a
        // 24-byte heap record. Returns a zeroed record on error.
        ("read_to_string", [path_addr]) => {
            let path = read_c_string(*path_addr)?;
            let rec = interp_mem::eval_intrinsic("__mind_alloc", &[24])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec, 0])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 8, 0])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 16, 0])?;
            let data = std::fs::read(&path).map_err(|e| format!("read_to_string: {e}"))?;
            let n = data.len() as i64;
            let buf = interp_mem::eval_intrinsic("__mind_alloc", &[n + 1])?;
            interp_mem::store_bytes(buf, &data)?;
            interp_mem::eval_intrinsic("__mind_store_i8", &[buf + n, 0])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec, buf])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 8, n])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 16, n + 1])?;
            Ok(rec)
        }
        // string_push_byte(s, b) -> new string record
        // Mirrors runtime-support/mind_intrinsics.c string_push_byte.
        ("string_push_byte", [s, b]) => {
            let addr = interp_mem::eval_intrinsic("__mind_load_i64", &[*s])?;
            let len = interp_mem::eval_intrinsic("__mind_load_i64", &[*s + 8])?;
            let cap = interp_mem::eval_intrinsic("__mind_load_i64", &[*s + 16])?;
            let new_len = len + 1;
            let new_cap = if cap == 0 {
                16
            } else if len < cap {
                cap
            } else {
                cap * 2
            };
            let new_addr = if new_cap == cap {
                addr
            } else {
                interp_mem::eval_intrinsic("__mind_realloc", &[addr, new_cap])?
            };
            interp_mem::eval_intrinsic("__mind_store_i8", &[new_addr + len, b & 0xFF])?;
            let rec = interp_mem::eval_intrinsic("__mind_alloc", &[24])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec, new_addr])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 8, new_len])?;
            interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 16, new_cap])?;
            Ok(rec)
        }
        // vec_zeroed(n) -> base address of n zeroed i64 elements
        ("vec_zeroed", [n]) => interp_mem::eval_intrinsic("__mind_vec_zeroed", &[*n]),
        _ => Err(format!(
            "{name}: wrong argument count for runtime-support intrinsic"
        )),
    }
}

// Thread-local fake fd table mapping integer descriptors to open files.
thread_local! {
    static FAKE_FDS: std::cell::RefCell<Vec<Option<(std::path::PathBuf, u64)>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Allocate a fake file descriptor for an opened path. Returns the fd.
pub(crate) fn alloc_fake_fd(path: &str) -> i64 {
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        // Reuse a closed slot if any.
        for (i, slot) in fds.iter().enumerate() {
            if slot.is_none() {
                fds[i] = Some((std::path::PathBuf::from(path), 0));
                return i as i64;
            }
        }
        fds.push(Some((std::path::PathBuf::from(path), 0)));
        (fds.len() - 1) as i64
    })
}

/// lseek on a fake fd. `whence`: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END.
pub(crate) fn lseek_fake_fd(fd: i64, offset: i64, whence: i64) -> Result<i64, String> {
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        let slot = fds
            .get_mut(fd as usize)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| format!("lseek: bad fd {fd}"))?;
        let (_, pos) = slot;
        let file_len = std::fs::metadata(&slot.0)
            .map(|m| m.len() as i64)
            .unwrap_or(0);
        let cur = *pos as i64;
        let new_pos = match whence {
            0 => offset,
            1 => cur + offset,
            2 => file_len + offset,
            _ => return Err(format!("lseek: bad whence {whence}")),
        };
        if new_pos < 0 {
            return Ok(-1);
        }
        *pos = new_pos as u64;
        Ok(new_pos)
    })
}

/// read from a fake fd into the arena buffer.
pub(crate) fn read_fake_fd(fd: i64, buf_addr: i64, count: i64) -> Result<i64, String> {
    if count <= 0 {
        return Ok(0);
    }
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        let slot = fds
            .get_mut(fd as usize)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| format!("read: bad fd {fd}"))?;
        let (path, pos) = slot;
        let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
        let start = *pos as usize;
        if start >= data.len() {
            return Ok(0); // EOF
        }
        let end = (start + count as usize).min(data.len());
        let chunk = &data[start..end];
        interp_mem::store_bytes(buf_addr, chunk)?;
        *pos = end as u64;
        Ok(chunk.len() as i64)
    })
}

/// close a fake fd.
pub(crate) fn close_fake_fd(fd: i64) -> Result<i64, String> {
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        if fd < 0 || fd as usize >= fds.len() || fds[fd as usize].is_none() {
            return Ok(-1);
        }
        fds[fd as usize] = None;
        Ok(0)
    })
}

/// Read a NUL-terminated C string from the arena at `addr`.
pub(crate) fn read_c_string(addr: i64) -> Result<String, String> {
    let mut bytes = Vec::new();
    let mut i = 0i64;
    loop {
        let b = interp_mem::eval_intrinsic("__mind_load_i8", &[addr + i])
            .map_err(|e| format!("open: cannot read path: {e}"))?;
        if b == 0 {
            break;
        }
        bytes.push(b as u8);
        i += 1;
        if i > 4096 {
            return Err("open: path too long".to_string());
        }
    }
    String::from_utf8(bytes).map_err(|e| format!("open: invalid path encoding: {e}"))
}

/// Dispatch a runtime-support extern-C call if `callee` matches, evaluating
/// each argument through the evaluator itself. Returns `None` when `callee`
/// is not an extern-C intrinsic so the caller can continue with its own
/// dispatch chain.
pub(crate) fn dispatch(
    callee: &str,
    args: &[super::Node],
    env: &std::collections::HashMap<String, super::Value>,
    tensor_env: &std::collections::HashMap<String, super::TensorEnvEntry>,
    mode: super::ExecMode,
) -> Option<Result<super::Value, super::EvalError>> {
    if !is_extern_c_intrinsic(callee) {
        return None;
    }
    let mut int_args = Vec::with_capacity(args.len());
    for arg in args {
        match super::eval_value_expr_mode(arg, env, tensor_env, mode.clone()) {
            Ok(super::Value::Int(n)) => int_args.push(n),
            Ok(other) => {
                return Some(Err(super::EvalError::UnsupportedMsg(format!(
                    "{callee}: expected an i64 argument, got {other:?}"
                ))));
            }
            Err(e) => return Some(Err(e)),
        }
    }
    Some(
        eval_extern_c_intrinsic(callee, &int_args)
            .map(super::Value::Int)
            .map_err(super::EvalError::UnsupportedMsg),
    )
}
