#ifndef OPENMED_FFI_H
#define OPENMED_FFI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(OPENMED_FFI_BUILDING_LIBRARY)
#define OPENMED_FFI_API __declspec(dllexport)
#else
#define OPENMED_FFI_API __declspec(dllimport)
#endif
#else
#define OPENMED_FFI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OPENMED_FFI_ABI_VERSION 1

typedef enum openmed_status {
    OPENMED_STATUS_OK = 0,
    OPENMED_STATUS_INVALID_ARGUMENT = 1,
    OPENMED_STATUS_OUT_OF_MEMORY = 2,
    OPENMED_STATUS_RUNTIME_UNAVAILABLE = 3,
    OPENMED_STATUS_INFERENCE_FAILED = 4,
    OPENMED_STATUS_INVALID_OUTPUT = 5
} openmed_status;

typedef struct openmed_runtime openmed_runtime;

typedef struct openmed_span {
    int64_t start;
    int64_t end;
    double score;
    char *label;
} openmed_span;

typedef struct openmed_span_list {
    openmed_span *items;
    size_t count;
} openmed_span_list;

typedef struct openmed_token_batch {
    const int64_t *input_ids;
    const int64_t *attention_mask;
    const int64_t *start_offsets;
    const int64_t *end_offsets;
    size_t token_count;
} openmed_token_batch;

/*
 * Native ONNX Runtime adapters register this narrow session interface once at
 * process start. The adapter owns its session and logits allocations; the shim
 * owns decoding, span allocation, and de-identification.
 */
typedef struct openmed_session_api {
    uint32_t abi_version;
    void *(*create)(
        const char *model_path,
        const char *const *labels,
        size_t label_count,
        char *error,
        size_t error_capacity
    );
    openmed_status (*run)(
        void *session,
        const openmed_token_batch *batch,
        float **logits,
        size_t *sequence_length,
        size_t *label_count,
        char *error,
        size_t error_capacity
    );
    void (*release_logits)(void *session, float *logits);
    void (*destroy)(void *session);
} openmed_session_api;

OPENMED_FFI_API uint32_t openmed_ffi_abi_version(void);

OPENMED_FFI_API openmed_status openmed_register_session_api(
    const openmed_session_api *api
);

OPENMED_FFI_API openmed_status openmed_runtime_create(
    const char *model_path,
    const char *const *labels,
    size_t label_count,
    openmed_runtime **runtime
);

OPENMED_FFI_API void openmed_runtime_destroy(openmed_runtime *runtime);

OPENMED_FFI_API openmed_status openmed_runtime_extract_pii(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    double threshold,
    openmed_span_list *spans
);

OPENMED_FFI_API openmed_status openmed_runtime_deidentify(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    double threshold,
    char **deidentified_text,
    openmed_span_list *spans
);

OPENMED_FFI_API void openmed_span_list_free(openmed_span_list *spans);

OPENMED_FFI_API void openmed_string_free(char *value);

/* The returned message never contains source note text. */
OPENMED_FFI_API const char *openmed_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
