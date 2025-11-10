#[cfg(feature = "ffi-c")]
pub fn generate_header() -> String {
    let header = r#"#ifndef MIND_H
#define MIND_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MIND_I32=0, MIND_F32=1 } MindDType;
typedef struct { uint32_t rank; const uint64_t* dims; } MindShape;
typedef struct { MindDType dtype; MindShape shape; void* data; uint64_t byte_len; } MindTensor;
typedef struct { const char* name; MindTensor tensor; } MindIO;
typedef struct { uint32_t inputs_len; uint32_t outputs_len; const char* model_name; uint64_t model_version; } MindModelMeta;

int mind_model_meta(MindModelMeta* out);
int mind_model_io(MindIO* inputs_out, uint32_t cap_inputs, MindIO* outputs_out, uint32_t cap_outputs);
int mind_infer(const MindIO* inputs, uint32_t inputs_len, MindIO* outputs, uint32_t outputs_len);
void* mind_alloc(uint64_t size);
void mind_free(void* p);
const char* mind_last_error(void);

#ifdef __cplusplus
} // extern "C"
#endif
#endif
"#;
    header.to_string()
}
