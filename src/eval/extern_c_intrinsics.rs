// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Extern "C" libc intrinsics dispatched by the evaluator so `mindc test`
//! can execute file-reading MIND (`std.fs.read_to_string`) without the
//! native runtime. Matches the POSIX semantics the native backend links
//! in `runtime-support/mind_intrinsics.c` by symbol.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::interp_mem;

/// Is `name` a runtime-support extern-C call the evaluator dispatches?
pub(crate) fn is_extern_c_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "open"
            | "lseek"
            | "read"
            | "close"
            | "read_to_string"
            | "string_push_byte"
            | "vec_zeroed"
            | "is_dir_fd"
    )
}

/// Evaluate one extern "C" libc call over the thread-local arena.
///
/// Arguments are i64 scalars (the MIND ABI); pointer arguments are arena
/// offsets. Returns the POSIX return value as i64.
pub(crate) fn eval_extern_c_intrinsic(name: &str, args: &[i64]) -> Result<i64, String> {
    match (name, args) {
        ("open", [path_addr, flags, mode]) => open_rdonly(*path_addr, *flags, *mode),
        ("lseek", [fd, offset, whence]) => lseek_fake_fd(*fd, *offset, *whence),
        ("read", [fd, buf_addr, count]) => read_fake_fd(*fd, *buf_addr, *count),
        ("close", [fd]) => close_fake_fd(*fd),
        ("read_to_string", [path_addr]) => read_to_string(*path_addr),
        ("string_push_byte", [s, b]) => string_push_byte(*s, *b),
        ("vec_zeroed", [n]) => interp_mem::eval_intrinsic("__mind_vec_zeroed", &[*n]),
        ("is_dir_fd", [fd]) => Ok(if fake_fd_is_dir(*fd) { 1 } else { 0 }),
        _ => Err(format!(
            "{name}: wrong argument count for runtime-support intrinsic"
        )),
    }
}

/// Descriptor numbers start at 3. 0, 1, and 2 stay unused so a fake table
/// cannot collide with the standard streams.
const FIRST_FD: usize = 3;

struct OpenFile {
    file: File,
}

thread_local! {
    static FAKE_FDS: std::cell::RefCell<Vec<Option<OpenFile>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Drop every open file. Called from [`interp_mem::reset`] once per test.
pub(crate) fn reset_fake_fds() {
    FAKE_FDS.with(|fds| fds.borrow_mut().clear());
}

fn open_rdonly(path_addr: i64, flags: i64, _mode: i64) -> Result<i64, String> {
    // A write, create, or truncate flag is a different operation. Refuse it
    // as an evaluator error so a write cannot report success.
    if flags != 0 {
        return Err(format!(
            "open: flags {flags} refused; only O_RDONLY (0) is supported"
        ));
    }
    if path_addr == 0 {
        return Ok(-1);
    }
    let bytes = read_c_string(path_addr)?;
    let path = path_from_bytes(bytes);
    match File::open(path) {
        Ok(file) => Ok(alloc_open_file(file)),
        Err(_) => Ok(-1),
    }
}

fn alloc_open_file(file: File) -> i64 {
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        if fds.len() < FIRST_FD {
            fds.resize_with(FIRST_FD, || None);
        }
        for i in FIRST_FD..fds.len() {
            if fds[i].is_none() {
                fds[i] = Some(OpenFile { file });
                return i as i64;
            }
        }
        fds.push(Some(OpenFile { file }));
        (fds.len() - 1) as i64
    })
}

fn fd_index(fd: i64) -> Option<usize> {
    usize::try_from(fd).ok()
}

fn slot_is_open(index: usize) -> bool {
    FAKE_FDS.with(|fds| {
        fds.borrow()
            .get(index)
            .and_then(|slot| slot.as_ref())
            .is_some()
    })
}

/// fstat S_ISDIR equivalent for a fake descriptor.
fn fake_fd_is_dir(fd: i64) -> bool {
    let Some(index) = fd_index(fd) else {
        return false;
    };
    FAKE_FDS.with(|fds| {
        fds.borrow()
            .get(index)
            .and_then(|slot| slot.as_ref())
            .map(|f| f.file.metadata().map(|m| m.is_dir()).unwrap_or(false))
            .unwrap_or(false)
    })
}

/// `whence`: 0 = SEEK_SET, 1 = SEEK_CUR, 2 = SEEK_END.
/// A bad descriptor, a bad whence, or an offset that overflows returns -1.
/// The stream position is left unchanged on those failures.
fn lseek_fake_fd(fd: i64, offset: i64, whence: i64) -> Result<i64, String> {
    if !(0..=2).contains(&whence) {
        return Ok(-1);
    }
    let Some(index) = fd_index(fd) else {
        return Ok(-1);
    };
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        let Some(slot) = fds.get_mut(index).and_then(|slot| slot.as_mut()) else {
            return Ok(-1);
        };
        let file = &mut slot.file;
        let cur = match file.stream_position() {
            Ok(pos) => match i64::try_from(pos) {
                Ok(v) => v,
                Err(_) => return Ok(-1),
            },
            Err(_) => return Ok(-1),
        };
        let end_len = if whence == 2 {
            match file.seek(SeekFrom::End(0)) {
                Ok(pos) => i64::try_from(pos).ok(),
                Err(_) => None,
            }
        } else {
            None
        };
        let new_pos = match whence {
            0 => (offset >= 0).then_some(offset),
            1 => cur.checked_add(offset),
            2 => end_len.and_then(|len| len.checked_add(offset)),
            _ => None,
        };
        let restore = |file: &mut File| {
            if let Ok(back) = u64::try_from(cur) {
                let _ = file.seek(SeekFrom::Start(back));
            }
        };
        let Some(pos) = new_pos.filter(|p| *p >= 0) else {
            restore(file);
            return Ok(-1);
        };
        match file.seek(SeekFrom::Start(pos as u64)) {
            Ok(now) => Ok(i64::try_from(now).unwrap_or(-1)),
            Err(_) => {
                restore(file);
                Ok(-1)
            }
        }
    })
}

/// Read at most `count` bytes, and never more than the free bytes of the
/// allocation that contains `buf_addr`. A bad descriptor returns -1.
fn read_fake_fd(fd: i64, buf_addr: i64, count: i64) -> Result<i64, String> {
    let Some(index) = fd_index(fd) else {
        return Ok(-1);
    };
    if !slot_is_open(index) {
        return Ok(-1);
    }
    if count < 0 {
        return Ok(-1);
    }
    if count == 0 {
        return Ok(0);
    }
    let space = interp_mem::writable_suffix(buf_addr)?;
    let want = usize::try_from(count).unwrap_or(usize::MAX).min(space);
    if want == 0 {
        return Ok(0);
    }
    let read: Result<Vec<u8>, ()> = FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        let Some(slot) = fds.get_mut(index).and_then(|slot| slot.as_mut()) else {
            return Err(());
        };
        let mut tmp = vec![0u8; want];
        loop {
            match slot.file.read(&mut tmp) {
                Ok(n) => {
                    tmp.truncate(n);
                    return Ok(tmp);
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(()),
            }
        }
    });
    match read {
        Ok(bytes) => {
            if !bytes.is_empty() {
                interp_mem::store_bytes(buf_addr, &bytes)?;
            }
            Ok(bytes.len() as i64)
        }
        Err(()) => Ok(-1),
    }
}

fn close_fake_fd(fd: i64) -> Result<i64, String> {
    let Some(index) = fd_index(fd) else {
        return Ok(-1);
    };
    FAKE_FDS.with(|fds| {
        let mut fds = fds.borrow_mut();
        if index >= fds.len() || fds[index].is_none() {
            return Ok(-1);
        }
        fds[index] = None;
        Ok(0)
    })
}

/// Same record rule as `std/fs.mind` `read_to_string`: cap is `file_sz + 1`.
/// A missing file, a directory, or a read error returns the zero record.
fn read_to_string(path_addr: i64) -> Result<i64, String> {
    let rec = interp_mem::eval_intrinsic("__mind_alloc", &[24])?;
    if path_addr == 0 {
        return Ok(rec);
    }
    let fd = eval_extern_c_intrinsic("open", &[path_addr, 0, 0])?;
    if fd < 0 {
        return Ok(rec);
    }
    // A directory is refused by file type (fstat S_ISDIR), not by size:
    // filesystems disagree on the size lseek reports for a directory (ext4
    // returns a non-zero block count; others return 0), so size is not a
    // reliable signal.  The zero record (ptr 0, len 0, cap 0) is the only
    // correct answer for a directory on every filesystem.
    if fake_fd_is_dir(fd) {
        let _ = close_fake_fd(fd);
        return Ok(rec);
    }
    let file_sz = lseek_fake_fd(fd, 0, 2)?;
    if file_sz < 0 {
        let _ = close_fake_fd(fd);
        return Ok(rec);
    }
    if lseek_fake_fd(fd, 0, 0)? < 0 {
        let _ = close_fake_fd(fd);
        return Ok(rec);
    }
    let Some(cap) = file_sz.checked_add(1) else {
        let _ = close_fake_fd(fd);
        return Ok(rec);
    };
    let buf = match interp_mem::eval_intrinsic("__mind_alloc", &[cap]) {
        Ok(buf) => buf,
        Err(err) => {
            let _ = close_fake_fd(fd);
            return Err(err);
        }
    };
    if buf == 0 {
        let _ = close_fake_fd(fd);
        return Ok(rec);
    }
    let n = read_fake_fd(fd, buf, file_sz)?;
    let _ = close_fake_fd(fd);
    if n < 0 {
        return Ok(rec);
    }
    interp_mem::eval_intrinsic("__mind_store_i8", &[buf + n, 0])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec, buf])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 8, n])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 16, cap])?;
    Ok(rec)
}

/// Append one byte. Growth matches the runtime helper: cap 0 becomes 16,
/// then doubles. Overflow fails closed. The new byte is stored as one byte.
fn string_push_byte(s: i64, b: i64) -> Result<i64, String> {
    let addr = interp_mem::eval_intrinsic("__mind_load_i64", &[s])?;
    let len = interp_mem::eval_intrinsic("__mind_load_i64", &[s + 8])?;
    let cap = interp_mem::eval_intrinsic("__mind_load_i64", &[s + 16])?;
    let new_len = len
        .checked_add(1)
        .ok_or_else(|| "string_push_byte: length overflow".to_string())?;
    let (new_cap, new_addr) = if len < cap {
        (cap, addr)
    } else {
        let new_cap = if cap == 0 {
            16
        } else {
            cap.checked_mul(2)
                .ok_or_else(|| "string_push_byte: capacity overflow".to_string())?
        };
        let alloc_n = new_cap
            .checked_add(7)
            .ok_or_else(|| "string_push_byte: capacity overflow".to_string())?;
        let new_addr = interp_mem::eval_intrinsic("__mind_alloc", &[alloc_n])?;
        if len > 0 && addr != 0 {
            for i in 0..len {
                let byte = interp_mem::eval_intrinsic("__mind_load_i8", &[addr + i])?;
                interp_mem::eval_intrinsic("__mind_store_i8", &[new_addr + i, byte])?;
            }
        }
        (new_cap, new_addr)
    };
    interp_mem::eval_intrinsic("__mind_store_i8", &[new_addr + len, b & 0xFF])?;
    let rec = interp_mem::eval_intrinsic("__mind_alloc", &[24])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec, new_addr])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 8, new_len])?;
    interp_mem::eval_intrinsic("__mind_store_i64", &[rec + 16, new_cap])?;
    Ok(rec)
}

fn read_c_string(addr: i64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut i = 0i64;
    loop {
        let b = interp_mem::eval_intrinsic("__mind_load_i8", &[addr + i])?;
        if b == 0 {
            break;
        }
        bytes.push(b as u8);
        i += 1;
        if i > 4096 {
            return Err("open: path too long".to_string());
        }
    }
    Ok(bytes)
}

fn path_from_bytes(bytes: Vec<u8>) -> std::path::PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(bytes).into()
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(bytes)
            .map(std::path::PathBuf::from)
            .unwrap_or_default()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &[i64]) -> Result<i64, String> {
        eval_extern_c_intrinsic(name, args)
    }

    fn field(rec: i64, off: i64) -> i64 {
        interp_mem::eval_intrinsic("__mind_load_i64", &[rec + off]).unwrap()
    }

    fn put_bytes(bytes: &[u8]) -> i64 {
        let n = bytes.len() as i64;
        let p = interp_mem::eval_intrinsic("__mind_alloc", &[n + 1]).unwrap();
        if n > 0 {
            interp_mem::store_bytes(p, bytes).unwrap();
        }
        interp_mem::eval_intrinsic("__mind_store_i8", &[p + n, 0]).unwrap();
        p
    }

    fn expect_neg1(result: Result<i64, String>) {
        match result {
            Ok(-1) => {}
            Ok(value) => panic!("expected -1, got {value}"),
            Err(_) => panic!("expected -1"),
        }
    }

    fn scratch_file(dir: &std::path::Path, name: &str, bytes: &[u8]) -> i64 {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        put_bytes(path.to_str().unwrap().as_bytes())
    }

    #[test]
    fn reset_reclaims_descriptor_three() {
        interp_mem::reset();
        let dir = tempfile::tempdir().unwrap();
        let path = scratch_file(dir.path(), "reset-a.txt", b"hello");
        let first = call("open", &[path, 0, 0]).unwrap();
        assert_eq!(first, 3);
        let second = call("open", &[path, 0, 0]).unwrap();
        assert_eq!(second, 4);
        // Leave both open. The next test-shaped reset must hand out 3 again.
        interp_mem::reset();
        let path = scratch_file(dir.path(), "reset-b.txt", b"hello");
        assert_eq!(call("open", &[path, 0, 0]).unwrap(), 3);
    }

    #[test]
    fn open_refuses_every_flag_except_read_only() {
        interp_mem::reset();
        let err = call("open", &[0, 577, 420]).unwrap_err();
        assert!(err.contains("O_RDONLY"), "{err}");
        assert!(!err.to_ascii_lowercase().contains("os error"));
    }

    #[test]
    fn null_path_returns_minus_one_and_an_empty_record() {
        interp_mem::reset();
        expect_neg1(call("open", &[0, 0, 0]));
        let rec = call("read_to_string", &[0]).unwrap();
        assert_eq!(field(rec, 0), 0);
        assert_eq!(field(rec, 8), 0);
        assert_eq!(field(rec, 16), 0);
    }

    #[test]
    fn bad_fd_returns_minus_one() {
        interp_mem::reset();
        let buf = interp_mem::eval_intrinsic("__mind_alloc", &[8]).unwrap();
        expect_neg1(call("lseek", &[99, 0, 0]));
        expect_neg1(call("read", &[99, buf, 1]));
        expect_neg1(call("close", &[99]));
    }

    #[test]
    fn negative_fd_returns_minus_one() {
        interp_mem::reset();
        expect_neg1(call("lseek", &[-1, 0, 0]));
        expect_neg1(call("read", &[-1, 0, 1]));
        expect_neg1(call("close", &[-1]));
    }

    #[test]
    fn missing_file_and_directory_return_the_empty_record() {
        interp_mem::reset();
        let missing = put_bytes(b"/no/such/mind-file-331");
        let rec = call("read_to_string", &[missing]).unwrap();
        assert_eq!((field(rec, 0), field(rec, 8), field(rec, 16)), (0, 0, 0));

        let dir = tempfile::tempdir().unwrap();
        let rec = call(
            "read_to_string",
            &[put_bytes(dir.path().to_str().unwrap().as_bytes())],
        )
        .unwrap();
        assert_eq!((field(rec, 0), field(rec, 8), field(rec, 16)), (0, 0, 0));
    }

    #[test]
    fn read_one_byte_and_clamp_to_the_allocation() {
        interp_mem::reset();
        let mut payload = vec![0u8; 65536];
        payload[0] = b'A';
        payload[1] = b'B';
        let dir = tempfile::tempdir().unwrap();
        let path = scratch_file(dir.path(), "large.bin", &payload);
        let fd = call("open", &[path, 0, 0]).unwrap();
        let buf = interp_mem::eval_intrinsic("__mind_alloc", &[4]).unwrap();
        assert_eq!(call("read", &[fd, buf, 1]).unwrap(), 1);
        assert_eq!(
            interp_mem::eval_intrinsic("__mind_load_i8", &[buf]).unwrap(),
            i64::from(b'A')
        );
        assert_eq!(call("read", &[fd, buf + 1, 1]).unwrap(), 1);
        assert_eq!(
            interp_mem::eval_intrinsic("__mind_load_i8", &[buf + 1]).unwrap(),
            i64::from(b'B')
        );
        let tiny = interp_mem::eval_intrinsic("__mind_alloc", &[1]).unwrap();
        assert_eq!(call("read", &[fd, tiny, 100_000]).unwrap(), 1);
        assert_eq!(call("close", &[fd]).unwrap(), 0);
    }

    #[test]
    fn lseek_overflow_returns_minus_one_and_keeps_the_cursor() {
        interp_mem::reset();
        let dir = tempfile::tempdir().unwrap();
        let path = scratch_file(dir.path(), "seek.txt", b"hello");
        let fd = call("open", &[path, 0, 0]).unwrap();
        assert_eq!(call("lseek", &[fd, 1, 0]).unwrap(), 1);
        expect_neg1(call("lseek", &[fd, i64::MAX, 1]));
        assert_eq!(call("lseek", &[fd, 0, 1]).unwrap(), 1);
        expect_neg1(call("lseek", &[fd, 0, 9]));
    }

    #[test]
    fn seek_end_overflow_leaves_the_cursor_unchanged() {
        interp_mem::reset();
        let dir = tempfile::tempdir().unwrap();
        let path = scratch_file(dir.path(), "seek-end.txt", b"hello");
        let fd = call("open", &[path, 0, 0]).unwrap();
        assert_eq!(call("lseek", &[fd, 1, 0]).unwrap(), 1);
        // SEEK_END (whence 2). A non-empty file plus i64::MAX overflows.
        expect_neg1(call("lseek", &[fd, i64::MAX, 2]));
        assert_eq!(call("lseek", &[fd, 0, 1]).unwrap(), 1);
    }

    // Linux only: the macOS file system refuses a file name that is not valid UTF-8.
    #[cfg(target_os = "linux")]
    #[test]
    fn open_reads_a_non_utf8_path() {
        use std::ffi::OsString;
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        interp_mem::reset();
        let dir = tempfile::tempdir().unwrap();
        let mut raw = dir.path().as_os_str().as_bytes().to_vec();
        raw.push(b'/');
        raw.extend_from_slice(&[b'q', 0xFF, b'z']);
        let file_path = std::path::PathBuf::from(OsString::from_vec(raw.clone()));
        std::fs::write(&file_path, b"Z").unwrap();
        let addr = put_bytes(&raw);
        let fd = call("open", &[addr, 0, 0]).unwrap();
        assert!(fd >= 3, "open returned {fd}");
        let buf = interp_mem::eval_intrinsic("__mind_alloc", &[1]).unwrap();
        assert_eq!(call("read", &[fd, buf, 1]).unwrap(), 1);
        assert_eq!(
            interp_mem::eval_intrinsic("__mind_load_i8", &[buf]).unwrap(),
            i64::from(b'Z')
        );
        assert_eq!(call("close", &[fd]).unwrap(), 0);
    }

    #[test]
    fn read_to_string_cap_is_file_size_plus_one() {
        interp_mem::reset();
        let dir = tempfile::tempdir().unwrap();
        let path = scratch_file(dir.path(), "four.txt", b"abcd");
        let rec = call("read_to_string", &[path]).unwrap();
        assert_eq!(field(rec, 8), 4);
        assert_eq!(field(rec, 16), 5);
        let addr = field(rec, 0);
        assert_ne!(addr, 0);
        assert_eq!(
            interp_mem::eval_intrinsic("__mind_load_i8", &[addr]).unwrap(),
            i64::from(b'a')
        );
    }

    #[test]
    fn string_push_byte_stores_the_byte_and_fails_closed_on_overflow() {
        interp_mem::reset();
        let empty = interp_mem::eval_intrinsic("__mind_alloc", &[24]).unwrap();
        let rec = call("string_push_byte", &[empty, 65]).unwrap();
        assert_eq!(field(rec, 8), 1);
        assert_eq!(field(rec, 16), 16);
        assert_eq!(
            interp_mem::eval_intrinsic("__mind_load_i8", &[field(rec, 0)]).unwrap(),
            65
        );
        let huge = interp_mem::eval_intrinsic("__mind_alloc", &[24]).unwrap();
        let cap = (i64::MAX / 2) + 10;
        interp_mem::eval_intrinsic("__mind_store_i64", &[huge + 8, cap]).unwrap();
        interp_mem::eval_intrinsic("__mind_store_i64", &[huge + 16, cap]).unwrap();
        let err = call("string_push_byte", &[huge, 65]).unwrap_err();
        assert!(err.contains("overflow"), "{err}");
    }

    #[test]
    fn vec_zeroed_and_close_match_the_native_edges() {
        interp_mem::reset();
        assert_eq!(call("vec_zeroed", &[0]).unwrap(), 0);
        assert_eq!(call("vec_zeroed", &[-4]).unwrap(), 0);
        let base = call("vec_zeroed", &[2]).unwrap();
        assert_ne!(base, 0);
        assert_eq!(field(base, 0), 0);
        assert_eq!(field(base, 8), 0);
        expect_neg1(call("close", &[3]));
    }
}
