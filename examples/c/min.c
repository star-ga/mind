#include "mind.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    MindModelMeta meta;
    if (mind_model_meta(&meta) != 0) {
        fprintf(stderr, "mind_model_meta failed: %s\n", mind_last_error());
        return 1;
    }

    printf("inputs=%u outputs=%u\n", meta.inputs_len, meta.outputs_len);
    return 0;
}
