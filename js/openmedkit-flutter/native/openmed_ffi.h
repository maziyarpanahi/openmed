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

/*
 * Version 2 requires a verified model digest at session creation and adds
 * runtime ownership, session-kind reporting, and live-runtime accounting.
 */
#define OPENMED_FFI_ABI_VERSION 2

/* Version of the openmed_session_api vtable that native adapters register. */
#define OPENMED_FFI_SESSION_API_VERSION 1

typedef enum openmed_status {
    OPENMED_STATUS_OK = 0,
    OPENMED_STATUS_INVALID_ARGUMENT = 1,
    OPENMED_STATUS_OUT_OF_MEMORY = 2,
    OPENMED_STATUS_RUNTIME_UNAVAILABLE = 3,
    OPENMED_STATUS_INFERENCE_FAILED = 4,
    OPENMED_STATUS_INVALID_OUTPUT = 5,
    OPENMED_STATUS_INTEGRITY_FAILED = 6
} openmed_status;

/* Which session implementation backs a runtime. */
typedef enum openmed_session_kind {
    OPENMED_SESSION_KIND_NONE = 0,
    /* Synthetic logits for tests; never evidence for a real runtime claim. */
    OPENMED_SESSION_KIND_FIXTURE = 1,
    /* ONNX Runtime linked into this library. */
    OPENMED_SESSION_KIND_ONNXRUNTIME = 2,
    /* An adapter registered through openmed_register_session_api. */
    OPENMED_SESSION_KIND_REGISTERED_ADAPTER = 3
} openmed_session_kind;

/* Bits returned by openmed_ffi_capabilities. */
#define OPENMED_FFI_CAPABILITY_ONNXRUNTIME 1u
#define OPENMED_FFI_CAPABILITY_FIXTURE_SESSION 2u
#define OPENMED_FFI_CAPABILITY_REGISTERED_ADAPTER 4u

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
 * owns decoding, span allocation, and de-identification. abi_version must be
 * OPENMED_FFI_SESSION_API_VERSION.
 *
 * The shim hashes the model file and compares it with the verified digest
 * before it calls create. Adapter error text is never surfaced to callers;
 * the shim reports a fixed, value-free message instead.
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

/* Bitwise OR of the OPENMED_FFI_CAPABILITY_* flags this build supports. */
OPENMED_FFI_API uint32_t openmed_ffi_capabilities(void);

OPENMED_FFI_API openmed_status openmed_register_session_api(
    const openmed_session_api *api
);

/*
 * Creates a runtime for one verified model file.
 *
 * model_sha256 is the bundle manifest digest of the file, written as
 * "sha256:" followed by 64 lowercase hex characters. The shim hashes the file
 * before any session exists and fails with OPENMED_STATUS_INTEGRITY_FAILED on
 * a mismatch. The built-in ONNX Runtime session is created from the same bytes
 * that were hashed. owner is an identifier from openmed_runtime_owner_create,
 * or zero for an unowned runtime.
 */
OPENMED_FFI_API openmed_status openmed_runtime_create(
    const char *model_path,
    const char *model_sha256,
    const char *const *labels,
    size_t label_count,
    uint64_t owner,
    openmed_runtime **runtime
);

/*
 * Frees a runtime exactly once. Destroying a runtime that was already freed,
 * for example by openmed_runtime_owner_release, is a no-op.
 */
OPENMED_FFI_API void openmed_runtime_destroy(openmed_runtime *runtime);

/* Returns the openmed_session_kind of a runtime. */
OPENMED_FFI_API int32_t openmed_runtime_session_kind(
    const openmed_runtime *runtime
);

/* Number of runtimes that have been created and not yet freed. */
OPENMED_FFI_API uint64_t openmed_runtime_live_count(void);

/* Returns a new non-zero owner identifier for runtime ownership. */
OPENMED_FFI_API uint64_t openmed_runtime_owner_create(void);

/*
 * Frees every runtime still held by owner and returns how many were freed.
 * Call it only after every thread that used those runtimes has stopped, for
 * example from an isolate exit listener.
 */
OPENMED_FFI_API uint64_t openmed_runtime_owner_release(uint64_t owner);

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

/*
 * The returned message comes from a fixed vocabulary. It never contains
 * source note text, file paths, digests, or adapter-provided text.
 */
OPENMED_FFI_API const char *openmed_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
