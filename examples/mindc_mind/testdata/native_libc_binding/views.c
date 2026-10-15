/* Copyright 2026 STARGA Inc. Independent test instrument, not product runtime. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    int64_t input[2];
    if (fread(input, sizeof(input), 1, stdin) != 1) return 71;
    if (getchar() != EOF) return 71;
    if (input[0] != 1 && input[0] != 2) return 72;
    uint32_t low = (uint32_t)input[1];
    int32_t signed_low;
    memcpy(&signed_low, &low, sizeof(low));
    int64_t result = input[0] == 1 ? (int64_t)low : (int64_t)signed_low;
    if (fwrite(&result, sizeof(result), 1, stdout) != 1) return 73;
    return 0;
}
