# Native C ABI lowering: fixed-arity `extern "C"` calls

Author: STARGA Inc.

The self-host native compiler (`examples/mindc_mind/main.mind`) lowers calls to
fixed-arity `extern "C"` declarations of the program's own source when the
callee is a row of its libc table (`nbx_row`). It is selected by the
capability mark `nb_cap_extern_fixed_v1` (`nbextc01`); a compiler without the
mark keeps refusing extern declarations.

## Admitted surface

- Declarations: top-level `extern "C"` blocks in the user region, `safe fn`,
  `unsafe fn` or bare `fn`, no variadics, at most eleven parameters. Parameter
  types: `i32`, `u32`, `i64`, `*const u8`, `*mut u8`, `*mut i32`,
  `*const *const u8`, `*mut *mut u8`, `*mut i64`. Results: `i32`, `i64`,
  `*mut u8`, `*mut i32`, or no result type. Any other pointer-to-pointer
  shape stays refused (exactly one `*const *const u8` and one
  `*mut *mut u8` parameter code), and `*mut *mut u8` / `*mut i64` are
  parameter-only widths.
- Arguments: no binding, function parameter, operator or return holds an
  `i32`, `u32` or pointer value, so such an argument is a cast written at the
  call (`fd as i32`, `last as u32`, `p as *mut u8`) or the result of another
  extern call of that type. No row returns `u32`, so a `u32` argument is always
  `EXPR as u32`, which keeps the low 32 bits of an integer (`nb_mask_rax_w`;
  MLIR masks the same bits). A float source refuses.
- Symbols: the table rows, each with a fixed glibc version and the source
  signature the declaration must spell exactly. This revision binds
  `close@GLIBC_2.2.5 (i32) -> i32`, `fstat@GLIBC_2.33 (i32, *mut u8) -> i32`,
  `getuid@GLIBC_2.2.5 () -> i32`, `__errno_location@GLIBC_2.2.5 () -> *mut i32`,
  and the rows after the `// L2b: extern table rows` anchor:
  `clock_gettime@GLIBC_2.17`, `close_range@GLIBC_2.34`, `closedir@GLIBC_2.2.5`,
  `dup2@GLIBC_2.2.5`, `fchdir@GLIBC_2.2.5`, `fdopendir@GLIBC_2.4`,
  `flock@GLIBC_2.2.5`, `fork@GLIBC_2.2.5`, `fstatat@GLIBC_2.33`,
  `fsync@GLIBC_2.2.5`, `getpgid@GLIBC_2.2.5`, `getrlimit@GLIBC_2.2.5`,
  `kill@GLIBC_2.2.5`, `lseek@GLIBC_2.2.5`, `memfd_create@GLIBC_2.27`,
  `nanosleep@GLIBC_2.2.5`, `pipe2@GLIBC_2.9`, `readdir@GLIBC_2.2.5`,
  `renameat@GLIBC_2.4`, `setpgid@GLIBC_2.2.5`, `setrlimit@GLIBC_2.2.5`,
  `signal@GLIBC_2.2.5`, `umask@GLIBC_2.2.5`, `unlinkat@GLIBC_2.4`,
  `waitid@GLIBC_2.2.5`, `waitpid@GLIBC_2.2.5`,
  `_exit@GLIBC_2.2.5 (i32)` with no result type,
  `fexecve@GLIBC_2.2.5 (i32, *const *const u8, *const *const u8) -> i32`,
  and the socket, address, fill and entropy rows
  `socket@GLIBC_2.2.5 (i32, i32, i32) -> i32`,
  `connect@GLIBC_2.2.5 (i32, *const u8, i32) -> i32`,
  `send@GLIBC_2.2.5 (i32, *const u8, i64, i32) -> i64`,
  `recv@GLIBC_2.2.5 (i32, *mut u8, i64, i32) -> i64`,
  `inet_pton@GLIBC_2.2.5 (i32, *const u8, *mut u8) -> i32`,
  `memset@GLIBC_2.2.5 (*mut u8, i32, i64) -> *mut u8`,
  `setsockopt@GLIBC_2.2.5 (i32, i32, i32, *const u8, i32) -> i32`,
  `getaddrinfo@GLIBC_2.2.5 (*const u8, *const u8, *const u8, *mut u8) -> i32`
  (the result-list slot is an 8-byte `*mut u8` buffer) and
  `getrandom@GLIBC_2.25 (*mut u8, i64, i32) -> i64`. `socklen_t` and the
  getrandom flags are spelled `i32`, as std's `net`, `dns` and `entropy`
  blocks declare them; the callee reads the low 32 bits either way.
  Variadic names stay outside this table.
- The GPU loader rows (ruling 16b) bind through `libcuda.so.1` with no symbol
  version: `cuInit (u32) -> i32`,
  `cuDeviceGet (*mut i32, i32) -> i32`,
  `cuCtxCreate_v2 (*mut *mut u8, u32, i32) -> i32`,
  `cuCtxDestroy_v2 (*mut u8) -> i32`, `cuCtxSynchronize () -> i32`,
  `cuModuleLoadData (*mut *mut u8, *const u8) -> i32`,
  `cuModuleGetFunction (*mut *mut u8, *mut u8, *const u8) -> i32`,
  `cuModuleUnload (*mut u8) -> i32`,
  `cuMemAlloc_v2 (*mut i64, i64) -> i32`,
  `cuMemcpyHtoD_v2 (i64, *const u8, i64) -> i32`,
  `cuMemcpyDtoH_v2 (*mut u8, i64, i64) -> i32`,
  `cuMemFree_v2 (i64) -> i32` and
  `cuLaunchKernel (*mut u8, u32, u32, u32, u32, u32, u32, u32,
  *mut u8, *mut *mut u8, *mut *mut u8) -> i32`. That is the driver's
  entry: the ninth argument is `CUstream` (a null `*mut u8` selects the
  default stream), the tenth is the `void**` kernel-parameter array and the
  eleventh is `void** extra`. The row names (19 bytes for `cuModuleGetFunction`) fit the 24-byte
  name field of the 80-byte `nbx_row` record, and the signature's second word
  carries parameters 6..10.
- The MC-6 driver rows (indices 54..86, `nbcu_row` in
  `examples/mindc_mind/nbx_cuda_rows2.mind`, capability mark `nbcuda02`) are
  the cuda.h 12.6 macro targets for the legacy default stream, each checked
  against `grep '#define <name>'` in that header: a `_v2` name wherever cuda.h
  maps the plain name to it, never a `_ptsz`/`_ptds` name (the plain legacy
  names keep their old argument layout in the driver and stay off the table).
  Streams and capture: `cuStreamCreate (*mut *mut u8, u32)`,
  `cuStreamSynchronize (*mut u8)`, `cuStreamQuery (*mut u8)`,
  `cuStreamDestroy_v2 (*mut u8)`,
  `cuStreamBeginCapture_v2 (*mut u8, i32)`,
  `cuStreamEndCapture (*mut u8, *mut *mut u8)`,
  `cuStreamGetCaptureInfo_v2 (*mut u8, *mut i32, *mut i64, *mut *mut u8,
  *mut *mut u8, *mut i64)`. Graphs: `cuGraphInstantiateWithFlags (*mut *mut u8,
  *mut u8, i64)`, `cuGraphAddKernelNode_v2 (*mut *mut u8, *mut u8,
  *const *const u8, i64, *const u8)`, `cuGraphExecUpdate_v2 (*mut u8, *mut u8,
  *mut u8)`, `cuGraphLaunch (*mut u8, *mut u8)`, `cuGraphDestroy (*mut u8)`,
  `cuGraphExecDestroy (*mut u8)`. Memory: `cuMemcpyHtoDAsync_v2 (i64,
  *const u8, i64, *mut u8)`, `cuMemcpyDtoHAsync_v2 (*mut u8, i64, i64,
  *mut u8)`, `cuMemcpy2D_v2 (*const u8)` and `cuMemcpy2DAsync_v2 (*const u8,
  *mut u8)` (the caller builds the 128-byte `CUDA_MEMCPY2D` in its own memory),
  `cuMemHostAlloc (*mut *mut u8, i64, u32)`, `cuMemAllocHost_v2 (*mut *mut u8,
  i64)`, `cuMemHostRegister_v2 (*mut u8, i64, u32)`, `cuMemFreeHost (*mut u8)`,
  `cuMemHostGetDevicePointer_v2 (*mut i64, *mut u8, u32)`,
  `cuMemsetD8_v2 (i64, u32, i64)`. Events: `cuEventCreate (*mut *mut u8, u32)`,
  `cuEventRecord (*mut u8, *mut u8)`, `cuEventSynchronize (*mut u8)`,
  `cuEventQuery (*mut u8)`,
  `cuEventElapsedTime (*mut i32, *mut u8, *mut u8)` (the f32 milliseconds read
  as bits), `cuEventDestroy_v2 (*mut u8)`. Queries and launch:
  `cuDeviceGetAttribute (*mut i32, i32, i32)`, `cuFuncGetAttribute (*mut i32,
  i32, *mut u8)`, `cuOccupancyMaxActiveBlocksPerMultiprocessor (*mut i32,
  *mut u8, i32, i64)`, `cuLaunchCooperativeKernel (*mut u8, u32, u32, u32, u32,
  u32, u32, u32, *mut u8, *mut *mut u8)` (four stack arguments, an even count,
  so no pad). Every row returns `CUresult` as `i32`. `cuMemsetD8_v2`'s
  `unsigned char` travels as `u32`; the self-host admits that argument only
  as a decimal integer literal in 0..255 or `x & m` with a decimal literal
  mask in 0..255 (through parentheses and unsigned casts) and otherwise
  refuses the program by name (`error[extern]: extern.byte_arg_unproven:
  <argument>`, exit 3, no artifact). The fence is the self-host's: the Rust
  front end and the MLIR leg pass the `u32` as written and the driver reads
  its low byte, so `256 as u32` builds there and sets bytes to 0 (that leg
  retires at the self-host flip).
  A name longer than 24 bytes (`cuGraphInstantiateWithFlags`,
  `cuStreamGetCaptureInfo_v2`, `cuLaunchCooperativeKernel`,
  `cuMemHostGetDevicePointer_v2`,
  `cuOccupancyMaxActiveBlocksPerMultiprocessor`) keeps its six name words in
  a separate block whose address the record's first name word holds;
  `nbcu_name` reads either form for the lookup and the string table.
  `cuMemAllocManaged` and `cuGetProcAddress` have no row and refuse by name.
- Any other declaration in the user region refuses the whole module. Declared
  but uncalled symbols create no import.

## Call sequence (System V x86-64)

Arguments are evaluated into their frame slots as for a local call, then loaded
into `rdi, rsi, rdx, rcx, r8, r9` by declared width: `movsxd` for `i32`, a
32-bit `mov` (zero-extending) for `u32`, a 64-bit `mov` for `i64` and pointers
(including `*const *const u8`, `*mut *mut u8` and `*mut i64`). Up to six
arguments the call is `mov r11,rsp ; and rsp,-16 ; push r11 ; push r11 ;
call [rip+disp32] ; mov rsp,[rsp]`, so the callee sees a 16-byte aligned stack
whatever the frame shape. Seven to eleven arguments mirror the local Call arm
instead: an `sub rsp,8` pad when the overflow count is odd, the overflow
arguments pushed right-to-left (`mov rax,[slot] ; push rax`, the callee
reading argument 6 at `[rsp+8]`), the same `call [rip+disp32]`, then
`add rsp, cleanup`. An `i32` result is
sign-extended (`movsxd rax, eax`); pointer results carry pointer origin. A
void result is not sign-extended and carries no pointer origin. The call
consumes one result slot, exactly like a local call, so the count and emit
walkers stay in lockstep.

## Image

A program with a user extern block gets a static-layout dynamic ET_EXEC: the
headers, code, trace note and entry are where the static writer puts them
(`0x400000`), the arena, stack and literal segments are unchanged, and the
image adds PT_PHDR, PT_INTERP (`/lib64/ld-linux-x86-64.so.2`), DT_NEEDED
`libc.so.6` unless the image is driver-only (driver rows imported, no libc
row), DT_NEEDED `libcuda.so.1` when a driver row is imported,
a read-only metadata load (dynstr, dynsym, SysV hash, RELA, VERSYM, VERNEED)
and a read-write load with one GOT slot per import
(`R_X86_64_GLOB_DAT`), sealed by PT_GNU_RELRO, with DT_FLAGS `BIND_NOW` and
DT_FLAGS_1 `NOW`, and PT_GNU_STACK without execute. There is no build-id,
timestamp or host path in the bytes; identical sources give identical images.
The dynamic segment is sized, not fixed: fifteen fixed entries plus one
DT_NEEDED per library, less the three version tags when no imported symbol
carries a glibc version. A program importing only driver rows (a driver-only
image) therefore lists DT_NEEDED `libcuda.so.1` alone, with no `libc.so.6`
entry or string, and has no DT_VERSYM, DT_VERNEED or DT_VERNEEDNUM (their
metadata is absent too). Such an image binds only the driver rows; libc is
still loaded into the process because the driver library depends on it. A
program that imports no driver row emits exactly the bytes this lane always
emitted — dynamic segment, strsz and GOT included.

The loader initialises libc (TLS, its own initialisers) before it jumps to the
entry. The image does not run `__libc_start_main`: the program exits through
the raw exit system call, so no atexit handler or stdio buffer is flushed. The
table therefore holds only functions that do not depend on that: system call
wrappers, `__errno_location`, the pure `memset` (an indirect function the
loader resolves while it binds the GOT) and `inet_pton`, the directory stream
functions and `getaddrinfo`, which allocate from libc's own heap. The
committed tests call `getaddrinfo` with numeric hosts and services only; a
name lookup through the name service is not measured. Explicit-layout
(`--native-profile bounded`) extern programs are outside this revision.

Exit contract: on an image whose extern block declares at least one driver
row (called or not), the entry return and the runtime bounds trap (status 77)
use `exit_group` (231), like the integer-division trap (79) and the
arena-exhausted exit (70), which always use it. A driver may start helper
threads, and `exit` (60) ends only the calling thread, so a trap after driver
initialisation would otherwise leave the process running or report a wrong
status. An image that declares no driver row keeps `exit` (60) and its bytes
exactly; so does the entry stub of a module without `main`, which runs no
code before it exits (`examples/mindc_mind/nb_exit.mind`). The exit decision
follows the declared rows, which are known before any function is lowered;
the NEEDED list follows the called rows. A libc image that declares a driver
row it never calls therefore ends through `exit_group` and lists `libc.so.6`
only.

The writer is a port of `native_dynamic.mind` (`nbx_meta`, `nbx_write_elf`),
laid out around the static image instead of a separate code page. Every call
site is recorded as a call row whose callee word is `-1 - row`; the writer
checks each placeholder still reads `FF 15 00 00 00 00` before binding it.
