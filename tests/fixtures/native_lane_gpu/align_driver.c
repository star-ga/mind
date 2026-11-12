/* Alignment probe for the 11-argument cuLaunchKernel row.
 *
 * Returns the low nibble of the frame pointer. A SysV entry has rsp 8 (mod 16);
 * `push rbp; mov rbp, rsp` then leaves the frame 0 (mod 16). A missing odd-count
 * pad, or a cleanup that forgets that pad and is followed by another call in
 * the same frame, leaves this nibble at 8.
 *
 * On the aligned path the function also executes movaps into a static slot, so
 * the success path really issues the instruction. The slot is not the frame:
 * writing xmm0 through rsp would clobber the saved rbp. Compiled -O0
 * -fno-omit-frame-pointer so the prologue does not realign rsp on its own.
 */
static unsigned char align_slot[16] __attribute__((aligned(16)));

int cuLaunchKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                   unsigned int bx, unsigned int by, unsigned int bz,
                   unsigned int shm, void *stream, void **params, void **extra) {
    unsigned long fp = (unsigned long)__builtin_frame_address(0);
    unsigned mis = (unsigned)(fp & 15ul);
    (void)f;
    (void)gx;
    (void)gy;
    (void)gz;
    (void)bx;
    (void)by;
    (void)bz;
    (void)shm;
    (void)stream;
    (void)params;
    (void)extra;
    if (mis == 0) {
        __asm__ __volatile__("movaps %%xmm0, %0" : "=m"(align_slot[0]) : : "xmm0", "memory");
    }
    return (int)mis;
}

/* The 10-argument cuLaunchCooperativeKernel row: its overflow count (4) is
 * even, so it takes NO pad. A pad emitted on the even count, or a cleanup
 * that drops one it never pushed, leaves this nibble at 8. */
int cuLaunchCooperativeKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                              unsigned int bx, unsigned int by, unsigned int bz,
                              unsigned int shm, void *stream, void **params) {
    unsigned long fp = (unsigned long)__builtin_frame_address(0);
    unsigned mis = (unsigned)(fp & 15ul);
    (void)f;
    (void)gx;
    (void)gy;
    (void)gz;
    (void)bx;
    (void)by;
    (void)bz;
    (void)shm;
    (void)stream;
    (void)params;
    if (mis == 0) {
        __asm__ __volatile__("movaps %%xmm0, %0" : "=m"(align_slot[0]) : : "xmm0", "memory");
    }
    return (int)mis;
}
