/* C callees for the extern "C" ABI rows (native_libc_abi_rows.tsv).
 * Each callee exists twice: s_* (System V) and w_* (ms_abi, the Win64
 * convention). Each result depends on every argument bit, so a wrong
 * extension or a wrong register/stack slot changes the value. Each ref_*
 * function calls its callee from C with the arguments the MIND fixtures
 * use, so the expected value is computed by gcc, never written by hand. */
#include <stdbool.h>
#include <stdint.h>

#define SYSV_CC
#define WIN64_CC __attribute__((ms_abi))

/* 64-bit polynomial over the arguments, wrapping (unsigned, no UB). */
#define MIX(acc, x) ((acc) * 1000003u + (uint64_t)(int64_t)(x))

#define CALLEES(CC, P)                                                        \
    CC int8_t P##i8(int8_t a) { return a; }                                   \
    CC uint8_t P##u8(uint8_t a) { return a; }                                 \
    CC int16_t P##i16(int16_t a) { return a; }                                \
    CC uint16_t P##u16(uint16_t a) { return a; }                              \
    CC int32_t P##i32(int32_t a) { return a; }                                \
    CC uint32_t P##u32(uint32_t a) { return a; }                              \
    CC bool P##bool(bool a) { return a; }                                     \
    CC float P##f32(float a) { return a * 2.0f + 0.25f; }                     \
    CC double P##f64(double a) { return a * 3.0 - 0.5; }                      \
    CC int64_t P##ints(int8_t a, uint8_t b, int16_t c, uint16_t d, int32_t e, \
                       uint32_t f, bool g, int64_t h, int8_t i, uint16_t j) { \
        uint64_t m = 7;                                                       \
        m = MIX(m, a); m = MIX(m, b); m = MIX(m, c); m = MIX(m, d);           \
        m = MIX(m, e); m = MIX(m, f); m = MIX(m, g); m = MIX(m, h);           \
        m = MIX(m, i); m = MIX(m, j);                                         \
        return (int64_t)m;                                                    \
    }                                                                         \
    CC int64_t P##floats(double a0, float a1, double a2, float a3, double a4,  \
                        float a5, double a6, float a7, double a8, float a9) { \
        return (int64_t)(16.0 * (a0 + 2 * a1 + 3 * a2 + 4 * a3 + 5 * a4 + 6 * a5 + 7 * a6 +     \
               8 * a7 + 9 * a8 + 10 * (double)a9));                             \
    }                                                                         \
    CC int64_t P##mixed(int8_t a, float b, uint16_t c, double d, int32_t e,    \
                       float f, uint32_t g, double h, int64_t i, float j,     \
                       uint8_t k, double l, int16_t m, bool n) {              \
        return (int64_t)(16.0 * (a + 2.0 * b + 3.0 * c + 4 * d + 5.0 * e + 6.0 * f + 7.0 * g +  \
               8 * h + 9.0 * (double)i + 10.0 * j + 11.0 * k + 12 * l +       \
               13.0 * m + 14.0 * n));                                          \
    }

CALLEES(SYSV_CC, s_)
CALLEES(WIN64_CC, w_)

/* The MIND fixtures pass these exact arguments (abi_sysv.mind, abi_win64.mind). */
#define REFS(P)                                                               \
    int64_t ref_##P##i8(void) { return P##i8(-1); }                           \
    int64_t ref_##P##u8(void) { return P##u8(255); }                          \
    int64_t ref_##P##i16(void) { return P##i16(-32768); }                     \
    int64_t ref_##P##u16(void) { return P##u16(65535); }                      \
    int64_t ref_##P##i32(void) { return P##i32(INT32_MIN); }                  \
    int64_t ref_##P##u32(void) { return P##u32(0x80000000u); }                \
    int64_t ref_##P##bool(void) { return P##bool(true); }                     \
    float ref_##P##f32(void) { return P##f32(1.5f); }                         \
    double ref_##P##f64(void) { return P##f64(2.25); }                        \
    int64_t ref_##P##ints(void) {                                             \
        return P##ints(-1, 255, -32768, 65535, INT32_MIN, 0x80000000u, true,  \
                       -5, -128, 40000);                                      \
    }                                                                         \
    int64_t ref_##P##floats(void) {                                            \
        return P##floats(0.5, 1.25f, -2.0, 3.5f, 0.125, -0.75f, 6.0, 0.0625f, \
                         -9.5, 10.25f);                                       \
    }                                                                         \
    int64_t ref_##P##mixed(void) {                                             \
        return P##mixed(-1, 1.5f, 65535, -2.25, INT32_MIN, 0.75f, 0x80000000u, \
                        4.5, -5, -3.25f, 255, 0.125, -32768, true);           \
    }

REFS(s_)
REFS(w_)
