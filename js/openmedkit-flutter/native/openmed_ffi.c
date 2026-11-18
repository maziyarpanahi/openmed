#include "openmed_ffi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#if defined(OPENMED_FFI_ONNXRUNTIME)
#include <onnxruntime_c_api.h>
#endif

#define OPENMED_ERROR_CAPACITY 256
#define OPENMED_SHA256_BYTES 32
#define OPENMED_READ_CHUNK 65536

#if defined(_MSC_VER)
#define OPENMED_THREAD_LOCAL __declspec(thread)
#else
#define OPENMED_THREAD_LOCAL _Thread_local
#endif

struct openmed_runtime {
    char **labels;
    size_t label_count;
    openmed_session_kind kind;
    openmed_session_api session_api;
    void *session;
    uint64_t owner;
    struct openmed_runtime *previous;
    struct openmed_runtime *next;
};

typedef struct pending_span {
    int active;
    int64_t start;
    int64_t end;
    double score_sum;
    size_t score_count;
    const char *label;
    size_t label_length;
} pending_span;

static OPENMED_THREAD_LOCAL char openmed_error[OPENMED_ERROR_CAPACITY];
static openmed_session_api registered_session_api;
static int has_registered_session_api;

/* Every live runtime is linked here so ownership release frees it once. */
static openmed_runtime *live_runtimes;
static uint64_t live_runtime_count;
static uint64_t next_owner_id;

#if defined(_WIN32)
static SRWLOCK registry_lock = SRWLOCK_INIT;
static void lock_registry(void) { AcquireSRWLockExclusive(&registry_lock); }
static void unlock_registry(void) { ReleaseSRWLockExclusive(&registry_lock); }
#else
static pthread_mutex_t registry_lock = PTHREAD_MUTEX_INITIALIZER;
static void lock_registry(void) { (void)pthread_mutex_lock(&registry_lock); }
static void unlock_registry(void) { (void)pthread_mutex_unlock(&registry_lock); }
#endif

static void set_error(const char *message) {
    if (message == NULL || message[0] == '\0') {
        openmed_error[0] = '\0';
        return;
    }
    (void)snprintf(openmed_error, sizeof(openmed_error), "%s", message);
}

/* ------------------------------------------------------------------------ */
/* SHA-256 (FIPS 180-4) for verified model loading                          */
/* ------------------------------------------------------------------------ */

typedef struct openmed_sha256 {
    uint32_t state[8];
    uint64_t length;
    unsigned char buffer[64];
    size_t buffered;
} openmed_sha256;

static const uint32_t sha256_constants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static uint32_t rotate_right(uint32_t value, unsigned int count) {
    return (value >> count) | (value << (32U - count));
}

static void sha256_block(openmed_sha256 *context, const unsigned char *block) {
    uint32_t schedule[64];
    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    size_t index;
    for (index = 0; index < 16; ++index) {
        schedule[index] = ((uint32_t)block[index * 4] << 24) |
            ((uint32_t)block[index * 4 + 1] << 16) |
            ((uint32_t)block[index * 4 + 2] << 8) |
            (uint32_t)block[index * 4 + 3];
    }
    for (index = 16; index < 64; ++index) {
        uint32_t s0 = rotate_right(schedule[index - 15], 7) ^
            rotate_right(schedule[index - 15], 18) ^ (schedule[index - 15] >> 3);
        uint32_t s1 = rotate_right(schedule[index - 2], 17) ^
            rotate_right(schedule[index - 2], 19) ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
    }
    for (index = 0; index < 64; ++index) {
        uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t first = h + s1 + choose + sha256_constants[index] + schedule[index];
        uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t second = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(openmed_sha256 *context) {
    static const uint32_t initial[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };
    memcpy(context->state, initial, sizeof(initial));
    context->length = 0;
    context->buffered = 0;
}

static void sha256_update(
    openmed_sha256 *context,
    const unsigned char *data,
    size_t length
) {
    context->length += (uint64_t)length;
    while (length > 0) {
        size_t take = 64 - context->buffered;
        if (take > length) {
            take = length;
        }
        memcpy(context->buffer + context->buffered, data, take);
        context->buffered += take;
        data += take;
        length -= take;
        if (context->buffered == 64) {
            sha256_block(context, context->buffer);
            context->buffered = 0;
        }
    }
}

static void sha256_final(
    openmed_sha256 *context,
    unsigned char digest[OPENMED_SHA256_BYTES]
) {
    uint64_t bits = context->length * 8U;
    unsigned char padding[72];
    size_t padding_length;
    size_t index;
    padding_length = context->buffered < 56
        ? 56 - context->buffered
        : 120 - context->buffered;
    memset(padding, 0, sizeof(padding));
    padding[0] = 0x80U;
    for (index = 0; index < 8; ++index) {
        padding[padding_length + index] = (unsigned char)(bits >> (56 - 8 * index));
    }
    {
        uint64_t length = context->length;
        sha256_update(context, padding, padding_length + 8);
        context->length = length;
    }
    for (index = 0; index < 8; ++index) {
        digest[index * 4] = (unsigned char)(context->state[index] >> 24);
        digest[index * 4 + 1] = (unsigned char)(context->state[index] >> 16);
        digest[index * 4 + 2] = (unsigned char)(context->state[index] >> 8);
        digest[index * 4 + 3] = (unsigned char)context->state[index];
    }
}

static int hex_value(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

/* Parses "sha256:<64 lowercase hex>" into raw digest bytes. */
static int parse_sha256(
    const char *text,
    unsigned char digest[OPENMED_SHA256_BYTES]
) {
    static const char prefix[] = "sha256:";
    size_t index;
    if (text == NULL || strncmp(text, prefix, sizeof(prefix) - 1) != 0) {
        return 0;
    }
    text += sizeof(prefix) - 1;
    if (strlen(text) != OPENMED_SHA256_BYTES * 2) {
        return 0;
    }
    for (index = 0; index < OPENMED_SHA256_BYTES; ++index) {
        int high = hex_value(text[index * 2]);
        int low = hex_value(text[index * 2 + 1]);
        if (high < 0 || low < 0) {
            return 0;
        }
        digest[index] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static int digests_equal(
    const unsigned char *left,
    const unsigned char *right
) {
    unsigned char difference = 0;
    size_t index;
    for (index = 0; index < OPENMED_SHA256_BYTES; ++index) {
        difference |= (unsigned char)(left[index] ^ right[index]);
    }
    return difference == 0;
}

static FILE *open_model_file(const char *model_path) {
#if defined(_WIN32)
    FILE *file = NULL;
    int length = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
    wchar_t *wide_path;
    if (length <= 0) {
        return NULL;
    }
    wide_path = (wchar_t *)malloc((size_t)length * sizeof(wchar_t));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wide_path, length) > 0) {
        file = _wfopen(wide_path, L"rb");
    }
    free(wide_path);
    return file;
#else
    return fopen(model_path, "rb");
#endif
}

/*
 * Hashes the model file and compares it with the verified digest. When bytes
 * is non-NULL the hashed bytes are returned so the session is created from
 * exactly the content that was verified.
 */
static openmed_status read_verified_model(
    const char *model_path,
    const unsigned char expected[OPENMED_SHA256_BYTES],
    unsigned char **bytes,
    size_t *length
) {
    openmed_sha256 hasher;
    unsigned char chunk[OPENMED_READ_CHUNK];
    unsigned char actual[OPENMED_SHA256_BYTES];
    unsigned char *buffer = NULL;
    size_t used = 0;
    size_t capacity = 0;
    size_t count;
    int read_failed;
    FILE *file = open_model_file(model_path);
    if (file == NULL) {
        set_error("the verified model file could not be read");
        return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
    }
    sha256_init(&hasher);
    while ((count = fread(chunk, 1, sizeof(chunk), file)) > 0) {
        sha256_update(&hasher, chunk, count);
        if (bytes != NULL) {
            if (used + count > capacity) {
                size_t grown = capacity == 0 ? sizeof(chunk) * 16 : capacity;
                unsigned char *resized;
                while (grown < used + count) {
                    if (grown > ((size_t)-1) / 2) {
                        grown = used + count;
                        break;
                    }
                    grown *= 2;
                }
                resized = (unsigned char *)realloc(buffer, grown);
                if (resized == NULL) {
                    free(buffer);
                    (void)fclose(file);
                    set_error("could not allocate the verified model bytes");
                    return OPENMED_STATUS_OUT_OF_MEMORY;
                }
                buffer = resized;
                capacity = grown;
            }
            memcpy(buffer + used, chunk, count);
            used += count;
        }
    }
    read_failed = ferror(file);
    (void)fclose(file);
    if (read_failed) {
        free(buffer);
        set_error("the verified model file could not be read");
        return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
    }
    sha256_final(&hasher, actual);
    if (!digests_equal(actual, expected)) {
        free(buffer);
        set_error("model bytes do not match the verified bundle digest");
        return OPENMED_STATUS_INTEGRITY_FAILED;
    }
    if (bytes != NULL) {
        *bytes = buffer;
        *length = used;
    }
    return OPENMED_STATUS_OK;
}

/* ------------------------------------------------------------------------ */
/* Built-in ONNX Runtime session                                             */
/* ------------------------------------------------------------------------ */

#if defined(OPENMED_FFI_ONNXRUNTIME)
typedef struct openmed_onnx_session {
    const OrtApi *api;
    OrtEnv *environment;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    size_t input_count;
} openmed_onnx_session;

/* ONNX Runtime messages can quote paths or graph content, so only the code
 * is reported. */
static int onnx_status_ok(const OrtApi *api, OrtStatus *status) {
    char message[96];
    if (status == NULL) {
        return 1;
    }
    (void)snprintf(
        message,
        sizeof(message),
        "ONNX Runtime operation failed (code %d)",
        (int)api->GetErrorCode(status)
    );
    api->ReleaseStatus(status);
    set_error(message);
    return 0;
}

static void destroy_onnx_session(void *opaque) {
    openmed_onnx_session *session = (openmed_onnx_session *)opaque;
    if (session == NULL) {
        return;
    }
    if (session->memory_info != NULL) {
        session->api->ReleaseMemoryInfo(session->memory_info);
    }
    if (session->session != NULL) {
        session->api->ReleaseSession(session->session);
    }
    if (session->environment != NULL) {
        session->api->ReleaseEnv(session->environment);
    }
    free(session);
}

static openmed_onnx_session *create_onnx_session(
    const unsigned char *model_bytes,
    size_t model_length
) {
    const OrtApiBase *api_base = OrtGetApiBase();
    openmed_onnx_session *created;
    OrtSessionOptions *options = NULL;
    OrtStatus *status;
    if (api_base == NULL) {
        set_error("ONNX Runtime API is unavailable");
        return NULL;
    }
    created = (openmed_onnx_session *)calloc(1, sizeof(openmed_onnx_session));
    if (created == NULL) {
        set_error("could not allocate an ONNX session");
        return NULL;
    }
    created->api = api_base->GetApi(ORT_API_VERSION);
    if (created->api == NULL) {
        set_error("ONNX Runtime ABI is incompatible");
        free(created);
        return NULL;
    }
    status = created->api->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        "openmed_flutter",
        &created->environment
    );
    if (!onnx_status_ok(created->api, status)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->CreateSessionOptions(&options);
    if (!onnx_status_ok(created->api, status)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SetIntraOpNumThreads(options, 1);
    if (!onnx_status_ok(created->api, status)) {
        created->api->ReleaseSessionOptions(options);
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SetSessionGraphOptimizationLevel(
        options,
        ORT_ENABLE_ALL
    );
    if (!onnx_status_ok(created->api, status)) {
        created->api->ReleaseSessionOptions(options);
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->CreateSessionFromArray(
        created->environment,
        model_bytes,
        model_length,
        options,
        &created->session
    );
    created->api->ReleaseSessionOptions(options);
    if (!onnx_status_ok(created->api, status)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->CreateCpuMemoryInfo(
        OrtArenaAllocator,
        OrtMemTypeDefault,
        &created->memory_info
    );
    if (!onnx_status_ok(created->api, status)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SessionGetInputCount(
        created->session,
        &created->input_count
    );
    if (!onnx_status_ok(created->api, status)) {
        destroy_onnx_session(created);
        return NULL;
    }
    if (created->input_count != 2 && created->input_count != 3) {
        set_error(
            "ONNX model must expose input_ids, attention_mask, and optional token_type_ids"
        );
        destroy_onnx_session(created);
        return NULL;
    }
    return created;
}

static int create_onnx_tensor(
    openmed_onnx_session *session,
    int64_t *values,
    size_t token_count,
    OrtValue **tensor
) {
    int64_t shape[2] = {1, (int64_t)token_count};
    OrtStatus *status = session->api->CreateTensorWithDataAsOrtValue(
        session->memory_info,
        values,
        token_count * sizeof(int64_t),
        shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        tensor
    );
    return onnx_status_ok(session->api, status);
}

static openmed_status run_onnx_session(
    openmed_onnx_session *session,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count
) {
    const char *input_names[3] = {
        "input_ids",
        "attention_mask",
        "token_type_ids"
    };
    const char *output_names[1] = {"logits"};
    OrtValue *inputs[3] = {NULL, NULL, NULL};
    const OrtValue *input_values[3];
    OrtValue *output = NULL;
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    int64_t dimensions[3] = {0, 0, 0};
    int64_t *token_types = NULL;
    ONNXTensorElementDataType element_type;
    size_t dimension_count = 0;
    size_t element_count = 0;
    void *data = NULL;
    OrtStatus *status;
    size_t index;
    openmed_status result = OPENMED_STATUS_INFERENCE_FAILED;

    if (!create_onnx_tensor(
            session,
            (int64_t *)batch->input_ids,
            batch->token_count,
            &inputs[0]) ||
        !create_onnx_tensor(
            session,
            (int64_t *)batch->attention_mask,
            batch->token_count,
            &inputs[1])) {
        goto cleanup;
    }
    if (session->input_count == 3) {
        token_types = (int64_t *)calloc(batch->token_count, sizeof(int64_t));
        if (token_types == NULL) {
            set_error("could not allocate token_type_ids");
            result = OPENMED_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        if (!create_onnx_tensor(
                session,
                token_types,
                batch->token_count,
                &inputs[2])) {
            goto cleanup;
        }
    }
    for (index = 0; index < session->input_count; ++index) {
        input_values[index] = inputs[index];
    }
    status = session->api->Run(
        session->session,
        NULL,
        input_names,
        input_values,
        session->input_count,
        output_names,
        1,
        &output
    );
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    status = session->api->GetTensorTypeAndShape(output, &shape_info);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    status = session->api->GetDimensionsCount(shape_info, &dimension_count);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    status = session->api->GetTensorElementType(shape_info, &element_type);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    if (dimension_count != 3 || element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        set_error("logits must be a rank-3 float tensor");
        result = OPENMED_STATUS_INVALID_OUTPUT;
        goto cleanup;
    }
    status = session->api->GetDimensions(shape_info, dimensions, 3);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    status = session->api->GetTensorShapeElementCount(shape_info, &element_count);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    if (dimensions[0] != 1 || dimensions[1] < 0 || dimensions[2] <= 0 ||
        (size_t)dimensions[1] != batch->token_count ||
        element_count != (size_t)(dimensions[1] * dimensions[2])) {
        set_error("logits dimensions do not match the input batch");
        result = OPENMED_STATUS_INVALID_OUTPUT;
        goto cleanup;
    }
    status = session->api->GetTensorMutableData(output, &data);
    if (!onnx_status_ok(session->api, status)) {
        goto cleanup;
    }
    *logits = (float *)malloc(element_count * sizeof(float));
    if (*logits == NULL) {
        set_error("could not allocate logits output");
        result = OPENMED_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(*logits, data, element_count * sizeof(float));
    *sequence_length = (size_t)dimensions[1];
    *label_count = (size_t)dimensions[2];
    result = OPENMED_STATUS_OK;

cleanup:
    if (shape_info != NULL) {
        session->api->ReleaseTensorTypeAndShapeInfo(shape_info);
    }
    if (output != NULL) {
        session->api->ReleaseValue(output);
    }
    for (index = 0; index < 3; ++index) {
        if (inputs[index] != NULL) {
            session->api->ReleaseValue(inputs[index]);
        }
    }
    free(token_types);
    return result;
}
#endif

/* ------------------------------------------------------------------------ */
/* Labels, offsets, and decoding                                            */
/* ------------------------------------------------------------------------ */

static char *copy_string(const char *value) {
    size_t length;
    char *copy;
    if (value == NULL) {
        return NULL;
    }
    length = strlen(value);
    copy = (char *)malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, value, length + 1);
    }
    return copy;
}

static char *copy_substring(const char *value, size_t length) {
    char *copy = (char *)malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, value, length);
        copy[length] = '\0';
    }
    return copy;
}

static void free_labels(char **labels, size_t count) {
    size_t index;
    if (labels == NULL) {
        return;
    }
    for (index = 0; index < count; ++index) {
        free(labels[index]);
    }
    free(labels);
}

static openmed_status copy_labels(
    const char *const *labels,
    size_t count,
    char ***output
) {
    char **copies;
    size_t index;
    copies = (char **)calloc(count, sizeof(char *));
    if (copies == NULL) {
        set_error("could not allocate runtime labels");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    for (index = 0; index < count; ++index) {
        if (labels[index] == NULL || labels[index][0] == '\0') {
            free_labels(copies, count);
            set_error("label values must not be empty");
            return OPENMED_STATUS_INVALID_ARGUMENT;
        }
        copies[index] = copy_string(labels[index]);
        if (copies[index] == NULL) {
            free_labels(copies, count);
            set_error("could not allocate runtime labels");
            return OPENMED_STATUS_OUT_OF_MEMORY;
        }
    }
    *output = copies;
    return OPENMED_STATUS_OK;
}

static size_t utf8_length(const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    size_t length = 0;
    while (*cursor != '\0') {
        if ((*cursor & 0xc0U) != 0x80U) {
            ++length;
        }
        ++cursor;
    }
    return length;
}

static size_t utf8_byte_offset(const char *value, size_t character_offset) {
    const unsigned char *cursor = (const unsigned char *)value;
    size_t characters = 0;
    while (*cursor != '\0' && characters < character_offset) {
        ++cursor;
        while ((*cursor & 0xc0U) == 0x80U) {
            ++cursor;
        }
        ++characters;
    }
    return (size_t)(cursor - (const unsigned char *)value);
}

static void parse_label(
    const char *raw,
    char *prefix,
    const char **label,
    size_t *label_length
) {
    size_t length = strlen(raw);
    *prefix = '\0';
    *label = raw;
    *label_length = length;
    if (length > 2 && (raw[1] == '-' || raw[1] == '_')) {
        char candidate = raw[0];
        if (candidate >= 'a' && candidate <= 'z') {
            candidate = (char)(candidate - ('a' - 'A'));
        }
        if (strchr("BIESUL", candidate) != NULL) {
            *prefix = candidate;
            *label = raw + 2;
            *label_length = length - 2;
        }
    }
}

static int label_is_outside(const char *label, size_t length) {
    return length == 1 && (label[0] == 'O' || label[0] == 'o');
}

static openmed_status append_pending(
    openmed_span_list *spans,
    pending_span *pending,
    double threshold
) {
    openmed_span *items;
    openmed_span *span;
    double score;
    if (!pending->active) {
        return OPENMED_STATUS_OK;
    }
    score = pending->score_sum / (double)pending->score_count;
    if (score < threshold) {
        pending->active = 0;
        return OPENMED_STATUS_OK;
    }
    items = (openmed_span *)realloc(
        spans->items,
        (spans->count + 1) * sizeof(openmed_span)
    );
    if (items == NULL) {
        set_error("could not allocate decoded spans");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    spans->items = items;
    span = &spans->items[spans->count];
    span->start = pending->start;
    span->end = pending->end;
    span->score = score;
    span->label = copy_substring(pending->label, pending->label_length);
    if (span->label == NULL) {
        set_error("could not allocate a span label");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    ++spans->count;
    pending->active = 0;
    return OPENMED_STATUS_OK;
}

static size_t maximum_label_index(const float *row, size_t label_count) {
    size_t best_index = 0;
    size_t index;
    for (index = 1; index < label_count; ++index) {
        if (row[index] > row[best_index]) {
            best_index = index;
        }
    }
    return best_index;
}

static double label_probability(
    const float *row,
    size_t label_count,
    size_t best_index
) {
    double denominator = 0.0;
    double maximum = (double)row[best_index];
    size_t index;
    for (index = 0; index < label_count; ++index) {
        denominator += exp((double)row[index] - maximum);
    }
    return denominator == 0.0 ? 0.0 : 1.0 / denominator;
}

static openmed_status decode_logits(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    const float *logits,
    size_t sequence_length,
    size_t output_label_count,
    double threshold,
    openmed_span_list *spans
) {
    pending_span pending = {0};
    size_t text_length = utf8_length(text);
    size_t token_index;
    if (sequence_length != batch->token_count ||
        output_label_count != runtime->label_count) {
        set_error("session logits do not match the token or label dimensions");
        return OPENMED_STATUS_INVALID_OUTPUT;
    }

    for (token_index = 0; token_index < sequence_length; ++token_index) {
        const float *row;
        size_t label_index;
        const char *label;
        size_t label_length;
        char prefix;
        int64_t start = batch->start_offsets[token_index];
        int64_t end = batch->end_offsets[token_index];
        double score;
        openmed_status status;

        if (start < 0 || end < start ||
            (uint64_t)end > (uint64_t)text_length) {
            set_error("token offsets fall outside the source text");
            return OPENMED_STATUS_INVALID_OUTPUT;
        }
        if (batch->attention_mask[token_index] == 0) {
            status = append_pending(spans, &pending, threshold);
            if (status != OPENMED_STATUS_OK) {
                return status;
            }
            continue;
        }
        row = logits + token_index * output_label_count;
        label_index = maximum_label_index(row, output_label_count);
        parse_label(
            runtime->labels[label_index],
            &prefix,
            &label,
            &label_length
        );
        if (start == end || label_is_outside(label, label_length)) {
            status = append_pending(spans, &pending, threshold);
            if (status != OPENMED_STATUS_OK) {
                return status;
            }
            continue;
        }
        score = label_probability(row, output_label_count, label_index);

        if (!pending.active || prefix == 'B' || prefix == 'S' || prefix == 'U' ||
            pending.label_length != label_length ||
            memcmp(pending.label, label, label_length) != 0 ||
            (prefix != 'I' && prefix != 'E' && prefix != 'L' &&
             start > pending.end)) {
            status = append_pending(spans, &pending, threshold);
            if (status != OPENMED_STATUS_OK) {
                return status;
            }
            pending.active = 1;
            pending.start = start;
            pending.end = end;
            pending.score_sum = score;
            pending.score_count = 1;
            pending.label = label;
            pending.label_length = label_length;
        } else {
            if (end > pending.end) {
                pending.end = end;
            }
            pending.score_sum += score;
            ++pending.score_count;
        }

        if (prefix == 'E' || prefix == 'L' || prefix == 'S' || prefix == 'U') {
            status = append_pending(spans, &pending, threshold);
            if (status != OPENMED_STATUS_OK) {
                return status;
            }
        }
    }
    return append_pending(spans, &pending, threshold);
}

/* ------------------------------------------------------------------------ */
/* Sessions                                                                 */
/* ------------------------------------------------------------------------ */

#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
/*
 * Synthetic logits for tests. A token with input id 1000 + k (0 < k <
 * label_count) selects label k; every other token selects label 0. The model
 * file is hashed like any other, but its contents are never interpreted.
 */
#define OPENMED_FIXTURE_LABEL_BASE 1000
static openmed_status fixture_logits(
    openmed_runtime *runtime,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count
) {
    size_t token_index;
    size_t label_index;
    *logits = (float *)malloc(
        batch->token_count * runtime->label_count * sizeof(float)
    );
    if (*logits == NULL) {
        set_error("could not allocate fixture logits");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    for (token_index = 0; token_index < batch->token_count; ++token_index) {
        int64_t selected = batch->input_ids[token_index] - OPENMED_FIXTURE_LABEL_BASE;
        if (selected <= 0 || (uint64_t)selected >= (uint64_t)runtime->label_count) {
            selected = 0;
        }
        for (label_index = 0; label_index < runtime->label_count; ++label_index) {
            (*logits)[token_index * runtime->label_count + label_index] =
                label_index == (size_t)selected ? 5.0f : 0.0f;
        }
    }
    *sequence_length = batch->token_count;
    *label_count = runtime->label_count;
    return OPENMED_STATUS_OK;
}
#endif

static openmed_status run_session(
    openmed_runtime *runtime,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count
) {
    switch (runtime->kind) {
#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
    case OPENMED_SESSION_KIND_FIXTURE:
        return fixture_logits(
            runtime,
            batch,
            logits,
            sequence_length,
            label_count
        );
#endif
#if defined(OPENMED_FFI_ONNXRUNTIME)
    case OPENMED_SESSION_KIND_ONNXRUNTIME:
        return run_onnx_session(
            (openmed_onnx_session *)runtime->session,
            batch,
            logits,
            sequence_length,
            label_count
        );
#endif
    case OPENMED_SESSION_KIND_REGISTERED_ADAPTER: {
        char adapter_error[OPENMED_ERROR_CAPACITY];
        openmed_status status;
        adapter_error[0] = '\0';
        status = runtime->session_api.run(
            runtime->session,
            batch,
            logits,
            sequence_length,
            label_count,
            adapter_error,
            sizeof(adapter_error)
        );
        if (status != OPENMED_STATUS_OK) {
            set_error("the registered session adapter failed");
        }
        return status;
    }
    default:
        set_error("the runtime has no session");
        return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
    }
}

static void release_logits(openmed_runtime *runtime, float *logits) {
    if (runtime->kind == OPENMED_SESSION_KIND_REGISTERED_ADAPTER) {
        runtime->session_api.release_logits(runtime->session, logits);
        return;
    }
    free(logits);
}

static void free_runtime(openmed_runtime *runtime) {
    switch (runtime->kind) {
#if defined(OPENMED_FFI_ONNXRUNTIME)
    case OPENMED_SESSION_KIND_ONNXRUNTIME:
        destroy_onnx_session(runtime->session);
        break;
#endif
    case OPENMED_SESSION_KIND_REGISTERED_ADAPTER:
        if (runtime->session != NULL) {
            runtime->session_api.destroy(runtime->session);
        }
        break;
    default:
        break;
    }
    free_labels(runtime->labels, runtime->label_count);
    free(runtime);
}

static int use_test_session(void) {
#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
    const char *value = getenv("OPENMED_FFI_TEST_SESSION");
    return value != NULL && strcmp(value, "1") == 0;
#else
    return 0;
#endif
}

static openmed_status create_session(
    openmed_runtime *runtime,
    const char *model_path,
    const unsigned char expected[OPENMED_SHA256_BYTES]
) {
    openmed_status status;
    if (use_test_session()) {
        status = read_verified_model(model_path, expected, NULL, NULL);
        if (status == OPENMED_STATUS_OK) {
            runtime->kind = OPENMED_SESSION_KIND_FIXTURE;
        }
        return status;
    }
    if (has_registered_session_api) {
        char adapter_error[OPENMED_ERROR_CAPACITY];
        status = read_verified_model(model_path, expected, NULL, NULL);
        if (status != OPENMED_STATUS_OK) {
            return status;
        }
        adapter_error[0] = '\0';
        runtime->session_api = registered_session_api;
        runtime->session = runtime->session_api.create(
            model_path,
            (const char *const *)runtime->labels,
            runtime->label_count,
            adapter_error,
            sizeof(adapter_error)
        );
        if (runtime->session == NULL) {
            set_error("the registered session adapter could not create a session");
            return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
        }
        runtime->kind = OPENMED_SESSION_KIND_REGISTERED_ADAPTER;
        return OPENMED_STATUS_OK;
    }
#if defined(OPENMED_FFI_ONNXRUNTIME)
    {
        unsigned char *model_bytes = NULL;
        size_t model_length = 0;
        status = read_verified_model(
            model_path,
            expected,
            &model_bytes,
            &model_length
        );
        if (status != OPENMED_STATUS_OK) {
            return status;
        }
        runtime->session = create_onnx_session(model_bytes, model_length);
        free(model_bytes);
        if (runtime->session == NULL) {
            if (openmed_error[0] == '\0') {
                set_error("the ONNX token-classification session could not be created");
            }
            return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
        }
        runtime->kind = OPENMED_SESSION_KIND_ONNXRUNTIME;
        return OPENMED_STATUS_OK;
    }
#else
    set_error("no ONNX token-classification session adapter is registered");
    return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
#endif
}

static openmed_status validate_inference(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    double threshold,
    openmed_span_list *spans
) {
    if (runtime == NULL || text == NULL || text[0] == '\0' || batch == NULL ||
        spans == NULL || batch->token_count == 0 || batch->input_ids == NULL ||
        batch->attention_mask == NULL || batch->start_offsets == NULL ||
        batch->end_offsets == NULL) {
        set_error("runtime, text, token batch, and span output are required");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    if (!(threshold >= 0.0 && threshold <= 1.0)) {
        set_error("threshold must be between zero and one");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    spans->items = NULL;
    spans->count = 0;
    return OPENMED_STATUS_OK;
}

/* ------------------------------------------------------------------------ */
/* Public ABI                                                               */
/* ------------------------------------------------------------------------ */

uint32_t openmed_ffi_abi_version(void) {
    return OPENMED_FFI_ABI_VERSION;
}

uint32_t openmed_ffi_capabilities(void) {
    uint32_t capabilities = 0;
#if defined(OPENMED_FFI_ONNXRUNTIME)
    capabilities |= OPENMED_FFI_CAPABILITY_ONNXRUNTIME;
#endif
#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
    capabilities |= OPENMED_FFI_CAPABILITY_FIXTURE_SESSION;
#endif
    if (has_registered_session_api) {
        capabilities |= OPENMED_FFI_CAPABILITY_REGISTERED_ADAPTER;
    }
    return capabilities;
}

openmed_status openmed_register_session_api(const openmed_session_api *api) {
    if (api == NULL || api->abi_version != OPENMED_FFI_SESSION_API_VERSION ||
        api->create == NULL || api->run == NULL ||
        api->release_logits == NULL || api->destroy == NULL) {
        set_error("session API is incomplete or has an incompatible ABI version");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    registered_session_api = *api;
    has_registered_session_api = 1;
    set_error(NULL);
    return OPENMED_STATUS_OK;
}

openmed_status openmed_runtime_create(
    const char *model_path,
    const char *model_sha256,
    const char *const *labels,
    size_t label_count,
    uint64_t owner,
    openmed_runtime **runtime
) {
    unsigned char expected[OPENMED_SHA256_BYTES];
    openmed_runtime *created;
    openmed_status status;
    if (runtime != NULL) {
        *runtime = NULL;
    }
    if (model_path == NULL || model_path[0] == '\0' || labels == NULL ||
        label_count < 2 || runtime == NULL) {
        set_error("model path, at least two labels, and runtime output are required");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    if (!parse_sha256(model_sha256, expected)) {
        set_error("a verified sha256 model digest is required");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    created = (openmed_runtime *)calloc(1, sizeof(openmed_runtime));
    if (created == NULL) {
        set_error("could not allocate the OpenMed runtime");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    status = copy_labels(labels, label_count, &created->labels);
    if (status != OPENMED_STATUS_OK) {
        free(created);
        return status;
    }
    created->label_count = label_count;
    created->owner = owner;
    status = create_session(created, model_path, expected);
    if (status != OPENMED_STATUS_OK) {
        free_runtime(created);
        return status;
    }
    lock_registry();
    created->next = live_runtimes;
    if (live_runtimes != NULL) {
        live_runtimes->previous = created;
    }
    live_runtimes = created;
    ++live_runtime_count;
    unlock_registry();
    *runtime = created;
    set_error(NULL);
    return OPENMED_STATUS_OK;
}

/* Unlinks runtime from the registry; the caller holds the lock. */
static void unlink_runtime(openmed_runtime *runtime) {
    if (runtime->previous != NULL) {
        runtime->previous->next = runtime->next;
    } else {
        live_runtimes = runtime->next;
    }
    if (runtime->next != NULL) {
        runtime->next->previous = runtime->previous;
    }
    runtime->previous = NULL;
    runtime->next = NULL;
    --live_runtime_count;
}

void openmed_runtime_destroy(openmed_runtime *runtime) {
    openmed_runtime *cursor;
    if (runtime == NULL) {
        return;
    }
    lock_registry();
    for (cursor = live_runtimes; cursor != NULL; cursor = cursor->next) {
        if (cursor == runtime) {
            unlink_runtime(cursor);
            break;
        }
    }
    unlock_registry();
    if (cursor != NULL) {
        free_runtime(cursor);
    }
}

int32_t openmed_runtime_session_kind(const openmed_runtime *runtime) {
    return runtime == NULL ? OPENMED_SESSION_KIND_NONE : (int32_t)runtime->kind;
}

uint64_t openmed_runtime_live_count(void) {
    uint64_t count;
    lock_registry();
    count = live_runtime_count;
    unlock_registry();
    return count;
}

uint64_t openmed_runtime_owner_create(void) {
    uint64_t owner;
    lock_registry();
    owner = ++next_owner_id;
    unlock_registry();
    return owner;
}

uint64_t openmed_runtime_owner_release(uint64_t owner) {
    openmed_runtime *released = NULL;
    openmed_runtime *cursor;
    uint64_t count = 0;
    if (owner == 0) {
        return 0;
    }
    lock_registry();
    cursor = live_runtimes;
    while (cursor != NULL) {
        openmed_runtime *next = cursor->next;
        if (cursor->owner == owner) {
            unlink_runtime(cursor);
            cursor->next = released;
            released = cursor;
            ++count;
        }
        cursor = next;
    }
    unlock_registry();
    while (released != NULL) {
        openmed_runtime *next = released->next;
        free_runtime(released);
        released = next;
    }
    return count;
}

openmed_status openmed_runtime_extract_pii(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    double threshold,
    openmed_span_list *spans
) {
    float *logits = NULL;
    size_t sequence_length = 0;
    size_t label_count = 0;
    openmed_status status = validate_inference(
        runtime,
        text,
        batch,
        threshold,
        spans
    );
    if (status != OPENMED_STATUS_OK) {
        return status;
    }
    openmed_error[0] = '\0';
    status = run_session(
        runtime,
        batch,
        &logits,
        &sequence_length,
        &label_count
    );
    if (status != OPENMED_STATUS_OK) {
        if (openmed_error[0] == '\0') {
            set_error("the ONNX token-classification session failed");
        }
        return status;
    }
    if (logits == NULL) {
        set_error("the ONNX token-classification session returned no logits");
        return OPENMED_STATUS_INVALID_OUTPUT;
    }
    status = decode_logits(
        runtime,
        text,
        batch,
        logits,
        sequence_length,
        label_count,
        threshold,
        spans
    );
    release_logits(runtime, logits);
    if (status != OPENMED_STATUS_OK) {
        openmed_span_list_free(spans);
        return status;
    }
    set_error(NULL);
    return OPENMED_STATUS_OK;
}

openmed_status openmed_runtime_deidentify(
    openmed_runtime *runtime,
    const char *text,
    const openmed_token_batch *batch,
    double threshold,
    char **deidentified_text,
    openmed_span_list *spans
) {
    openmed_status status;
    size_t source_bytes;
    size_t output_bytes;
    size_t prior_character = 0;
    size_t output_offset = 0;
    size_t index;
    char *output;
    if (deidentified_text == NULL) {
        set_error("de-identified text output is required");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    *deidentified_text = NULL;
    status = openmed_runtime_extract_pii(
        runtime,
        text,
        batch,
        threshold,
        spans
    );
    if (status != OPENMED_STATUS_OK) {
        return status;
    }
    source_bytes = strlen(text);
    output_bytes = source_bytes;
    for (index = 0; index < spans->count; ++index) {
        size_t start_byte;
        size_t end_byte;
        size_t replacement_bytes = strlen(spans->items[index].label) + 2;
        if ((size_t)spans->items[index].start < prior_character) {
            openmed_span_list_free(spans);
            set_error("decoded spans overlap or are not ordered");
            return OPENMED_STATUS_INVALID_OUTPUT;
        }
        start_byte = utf8_byte_offset(text, (size_t)spans->items[index].start);
        end_byte = utf8_byte_offset(text, (size_t)spans->items[index].end);
        output_bytes = output_bytes - (end_byte - start_byte) + replacement_bytes;
        prior_character = (size_t)spans->items[index].end;
    }
    output = (char *)malloc(output_bytes + 1);
    if (output == NULL) {
        openmed_span_list_free(spans);
        set_error("could not allocate de-identified output");
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    prior_character = 0;
    for (index = 0; index < spans->count; ++index) {
        size_t prior_byte = utf8_byte_offset(text, prior_character);
        size_t start_byte = utf8_byte_offset(
            text,
            (size_t)spans->items[index].start
        );
        size_t prefix_bytes = start_byte - prior_byte;
        size_t label_bytes = strlen(spans->items[index].label);
        memcpy(output + output_offset, text + prior_byte, prefix_bytes);
        output_offset += prefix_bytes;
        output[output_offset++] = '[';
        memcpy(output + output_offset, spans->items[index].label, label_bytes);
        output_offset += label_bytes;
        output[output_offset++] = ']';
        prior_character = (size_t)spans->items[index].end;
    }
    {
        size_t prior_byte = utf8_byte_offset(text, prior_character);
        size_t suffix_bytes = source_bytes - prior_byte;
        memcpy(output + output_offset, text + prior_byte, suffix_bytes);
        output_offset += suffix_bytes;
    }
    output[output_offset] = '\0';
    *deidentified_text = output;
    return OPENMED_STATUS_OK;
}

void openmed_span_list_free(openmed_span_list *spans) {
    size_t index;
    if (spans == NULL) {
        return;
    }
    for (index = 0; index < spans->count; ++index) {
        free(spans->items[index].label);
    }
    free(spans->items);
    spans->items = NULL;
    spans->count = 0;
}

void openmed_string_free(char *value) {
    free(value);
}

const char *openmed_last_error(void) {
    return openmed_error;
}
