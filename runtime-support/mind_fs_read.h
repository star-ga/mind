// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mind_fs_read.h — the file open and the whole-file reader of the runtime-support shim.
//
// A part of mind_intrinsics.c, not a standalone header: that file includes it
// once, after the portability shim and the heap helpers this code calls
// (MIND_EXPORT, MIND_OPEN_RDONLY, MIND_READ, __mind_alloc, __mind_store_i64).
// mindc bundles it into the same single source (src/eval/mlir_build/
// runtime_src.rs), so the shim still compiles as ONE object under the fixed
// `mind_intrinsics.c` basename.
#ifndef MIND_FS_READ_H
#define MIND_FS_READ_H

#include <sys/types.h>
#include <sys/stat.h>

// __mind_open(path_addr) — open(2) with O_RDONLY on a NUL-terminated C string.
//
// The contract is fixed by std/fs.mind:274-285 and by the native-ELF self-host
// emitter's intrinsic slot 12 (`examples/mindc_mind/main.mind` nb_intrinsic_id):
// ONE argument, read-only, path is a NUL-TERMINATED C string, returns the fd, or
// a NEGATIVE value on error. Callers test `fd < 0`. This libc open folds every
// error, including a null path, into -1 and parks the code in `errno`. The raw
// `open` syscall returns `-errno`; the native lane normalizes that negative
// result to this same -1 before a program observes it. std/fs.mind also maps
// any negative fd to -1. Exact error integers are not portable evidence.
//
// Determinism: registered `Det::Pure` in `src/intrinsics.rs`, the same NAME-level
// verdict `__mind_read` carries. A path is a declared program input; the fd is an
// opaque process-local descriptor whose numeric value the std surface only ever
// compares against zero, never prints or hashes. No clock, no entropy, no pointer
// bits reach the result.
//
// There is deliberately no `__mind_close`: std/fs.mind's single-shot reader relies
// on process exit to reclaim the descriptor, and that deferral is recorded at
// std/fs.mind:284-285 with its upgrade path. Adding one here without the matching
// intrinsic row and native-ELF slot would be a half-wired third backend.
MIND_EXPORT int64_t __mind_open(int64_t path_addr) {
    if (path_addr == 0) {
        return -1;
    }
    const char *path = (const char *)(uintptr_t)path_addr;
    int fd = MIND_OPEN_RDONLY(path);
    return fd < 0 ? (int64_t)-1 : (int64_t)fd;
}

// is_dir_fd(fd) — 1 if the descriptor refers to a directory, else 0 (also 0
// when fstat fails). The test is the file type, so every filesystem agrees:
// the size lseek reports for a directory differs between filesystems (ext4
// reports a block count, others report 0) and is never consulted. Exported, so
// MIND code can ask the same question through `extern "C"`.
MIND_EXPORT int64_t is_dir_fd(int64_t fd) {
#if defined(_WIN32)
    struct _stat64 st;
    if (_fstat64((int)fd, &st) != 0) return 0;
    return (st.st_mode & _S_IFMT) == _S_IFDIR ? 1 : 0;
#else
    struct stat st;
    if (fstat((int)fd, &st) != 0) return 0;
    return S_ISDIR(st.st_mode) ? 1 : 0;
#endif
}

// read_to_string — whole file as a heap record [addr, len, cap].
//
// Same rule as std/fs.mind: size the file, allocate file_sz + 1 bytes, read
// that many bytes, store a trailing NUL, and publish cap = file_sz + 1.
// A null path, a failed open, a directory, a failed size, or a real read
// error returns the still-zero record (ptr 0, len 0, cap 0) and does not
// publish a partial buffer. EINTR is retried. MIND_EXPORT_WEAK so a linked
// std/fs object overrides this fallback; without that object this is the
// definition that links.
MIND_EXPORT_WEAK int64_t read_to_string(int64_t path_addr) {
    int64_t rec = __mind_alloc(24);
    __mind_store_i64(rec,      0);
    __mind_store_i64(rec + 8,  0);
    __mind_store_i64(rec + 16, 0);

    if (path_addr == 0) return rec;
    const char *path = (const char *)(uintptr_t)path_addr;
    int fd = MIND_OPEN_RDONLY(path);
    if (fd < 0) return rec;

    // A directory is refused by file type before any size is read (is_dir_fd).
    if (is_dir_fd(fd)) {
        MIND_CLOSE(fd);
        return rec;
    }

    int64_t file_sz = MIND_LSEEK(fd, 0, SEEK_END);
    if (file_sz < 0 || file_sz == INT64_MAX) {
        MIND_CLOSE(fd);
        return rec;
    }
    if (MIND_LSEEK(fd, 0, SEEK_SET) < 0) {
        MIND_CLOSE(fd);
        return rec;
    }

    int64_t buf = __mind_alloc(file_sz + 1);
    if (buf == 0) {
        MIND_CLOSE(fd);
        return rec;
    }

    int64_t n = 0;
    while (n < file_sz) {
        size_t chunk = (size_t)1 << 20;
        if ((int64_t)chunk > file_sz - n) chunk = (size_t)(file_sz - n);
        int64_t got = (int64_t)MIND_READ(fd, (void *)(uintptr_t)(buf + n), chunk);
        if (got < 0) {
            if (errno == EINTR) continue;
            MIND_CLOSE(fd);
            return rec;
        }
        if (got == 0) break;
        n += got;
    }
    MIND_CLOSE(fd);
    __mind_store_i8(buf + n, 0);
    __mind_store_i64(rec,      buf);
    __mind_store_i64(rec + 8,  n);
    __mind_store_i64(rec + 16, file_sz + 1);
    return rec;
}

#endif  /* MIND_FS_READ_H */
