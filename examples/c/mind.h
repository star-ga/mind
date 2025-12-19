// Copyright 2025 STARGA Inc.
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

// Part of the MIND project (Machine Intelligence Native Design).

// Stub FFI header for open-core build.
// Full implementation is provided by the proprietary mind-runtime backend.

#ifndef MIND_H
#define MIND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
    uint32_t inputs_len;
    uint32_t outputs_len;
} MindModelMeta;

// Stub: returns -1 (unsupported in open-core)
static inline int mind_model_meta(MindModelMeta *meta) {
    (void)meta;
    return -1;
}

// Stub: returns static error message
static inline const char *mind_last_error(void) {
    return "FFI not available in open-core build";
}

#ifdef __cplusplus
}
#endif

#endif // MIND_H
