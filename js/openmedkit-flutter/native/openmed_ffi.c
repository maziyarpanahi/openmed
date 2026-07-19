#include "openmed_ffi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(OPENMED_FFI_ONNXRUNTIME)
#include <onnxruntime_c_api.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#define OPENMED_ERROR_CAPACITY 256

#if defined(_MSC_VER)
#define OPENMED_THREAD_LOCAL __declspec(thread)
#else
#define OPENMED_THREAD_LOCAL _Thread_local
#endif

struct openmed_runtime {
    char **labels;
    size_t label_count;
    openmed_session_api session_api;
    void *session;
    int fixture_session;
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

#if defined(OPENMED_FFI_ONNXRUNTIME)
typedef struct openmed_onnx_session {
    const OrtApi *api;
    OrtEnv *environment;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    size_t input_count;
} openmed_onnx_session;

static int onnx_status_ok(
    const OrtApi *api,
    OrtStatus *status,
    char *error,
    size_t error_capacity
) {
    const char *message;
    if (status == NULL) {
        return 1;
    }
    message = api->GetErrorMessage(status);
    (void)snprintf(
        error,
        error_capacity,
        "ONNX Runtime operation failed: %s",
        message == NULL ? "unknown error" : message
    );
    api->ReleaseStatus(status);
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

#if defined(_WIN32)
static wchar_t *onnx_model_path(const char *model_path) {
    int length = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
    wchar_t *wide_path;
    if (length <= 0) {
        return NULL;
    }
    wide_path = (wchar_t *)malloc((size_t)length * sizeof(wchar_t));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wide_path, length) <= 0) {
        free(wide_path);
        return NULL;
    }
    return wide_path;
}
#endif

static void *create_onnx_session(
    const char *model_path,
    const char *const *labels,
    size_t label_count,
    char *error,
    size_t error_capacity
) {
    const OrtApiBase *api_base = OrtGetApiBase();
    openmed_onnx_session *created;
    OrtSessionOptions *options = NULL;
    OrtStatus *status;
    (void)labels;
    (void)label_count;
    if (api_base == NULL) {
        (void)snprintf(error, error_capacity, "ONNX Runtime API is unavailable");
        return NULL;
    }
    created = (openmed_onnx_session *)calloc(1, sizeof(openmed_onnx_session));
    if (created == NULL) {
        (void)snprintf(error, error_capacity, "could not allocate an ONNX session");
        return NULL;
    }
    created->api = api_base->GetApi(ORT_API_VERSION);
    if (created->api == NULL) {
        (void)snprintf(error, error_capacity, "ONNX Runtime ABI is incompatible");
        free(created);
        return NULL;
    }
    status = created->api->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        "openmed_flutter",
        &created->environment
    );
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->CreateSessionOptions(&options);
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SetIntraOpNumThreads(options, 1);
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        created->api->ReleaseSessionOptions(options);
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SetSessionGraphOptimizationLevel(
        options,
        ORT_ENABLE_ALL
    );
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        created->api->ReleaseSessionOptions(options);
        destroy_onnx_session(created);
        return NULL;
    }
#if defined(_WIN32)
    {
        wchar_t *wide_path = onnx_model_path(model_path);
        if (wide_path == NULL) {
            created->api->ReleaseSessionOptions(options);
            destroy_onnx_session(created);
            (void)snprintf(error, error_capacity, "model path is not valid UTF-8");
            return NULL;
        }
        status = created->api->CreateSession(
            created->environment,
            wide_path,
            options,
            &created->session
        );
        free(wide_path);
    }
#else
    status = created->api->CreateSession(
        created->environment,
        model_path,
        options,
        &created->session
    );
#endif
    created->api->ReleaseSessionOptions(options);
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->CreateCpuMemoryInfo(
        OrtArenaAllocator,
        OrtMemTypeDefault,
        &created->memory_info
    );
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        destroy_onnx_session(created);
        return NULL;
    }
    status = created->api->SessionGetInputCount(
        created->session,
        &created->input_count
    );
    if (!onnx_status_ok(created->api, status, error, error_capacity)) {
        destroy_onnx_session(created);
        return NULL;
    }
    if (created->input_count != 2 && created->input_count != 3) {
        (void)snprintf(
            error,
            error_capacity,
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
    OrtValue **tensor,
    char *error,
    size_t error_capacity
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
    return onnx_status_ok(session->api, status, error, error_capacity);
}

static openmed_status run_onnx_session(
    void *opaque,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count,
    char *error,
    size_t error_capacity
) {
    openmed_onnx_session *session = (openmed_onnx_session *)opaque;
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
            &inputs[0],
            error,
            error_capacity) ||
        !create_onnx_tensor(
            session,
            (int64_t *)batch->attention_mask,
            batch->token_count,
            &inputs[1],
            error,
            error_capacity)) {
        goto cleanup;
    }
    if (session->input_count == 3) {
        token_types = (int64_t *)calloc(batch->token_count, sizeof(int64_t));
        if (token_types == NULL) {
            (void)snprintf(error, error_capacity, "could not allocate token_type_ids");
            result = OPENMED_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        if (!create_onnx_tensor(
                session,
                token_types,
                batch->token_count,
                &inputs[2],
                error,
                error_capacity)) {
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
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    status = session->api->GetTensorTypeAndShape(output, &shape_info);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    status = session->api->GetDimensionsCount(shape_info, &dimension_count);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    status = session->api->GetTensorElementType(shape_info, &element_type);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    if (dimension_count != 3 || element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        (void)snprintf(error, error_capacity, "logits must be a rank-3 float tensor");
        result = OPENMED_STATUS_INVALID_OUTPUT;
        goto cleanup;
    }
    status = session->api->GetDimensions(shape_info, dimensions, 3);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    status = session->api->GetTensorShapeElementCount(shape_info, &element_count);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    if (dimensions[0] != 1 || dimensions[1] < 0 || dimensions[2] <= 0 ||
        (size_t)dimensions[1] != batch->token_count ||
        element_count != (size_t)(dimensions[1] * dimensions[2])) {
        (void)snprintf(error, error_capacity, "logits dimensions do not match the input batch");
        result = OPENMED_STATUS_INVALID_OUTPUT;
        goto cleanup;
    }
    status = session->api->GetTensorMutableData(output, &data);
    if (!onnx_status_ok(session->api, status, error, error_capacity)) {
        goto cleanup;
    }
    *logits = (float *)malloc(element_count * sizeof(float));
    if (*logits == NULL) {
        (void)snprintf(error, error_capacity, "could not allocate logits output");
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

static void release_onnx_logits(void *session, float *logits) {
    (void)session;
    free(logits);
}

static const openmed_session_api onnx_session_api = {
    OPENMED_FFI_ABI_VERSION,
    create_onnx_session,
    run_onnx_session,
    release_onnx_logits,
    destroy_onnx_session
};
#endif

static void set_error(const char *message) {
    if (message == NULL || message[0] == '\0') {
        openmed_error[0] = '\0';
        return;
    }
    (void)snprintf(openmed_error, sizeof(openmed_error), "%s", message);
}

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
        const float *row = logits + token_index * output_label_count;
        size_t label_index = maximum_label_index(row, output_label_count);
        const char *label;
        size_t label_length;
        char prefix;
        int64_t start = batch->start_offsets[token_index];
        int64_t end = batch->end_offsets[token_index];
        double score;
        openmed_status status;

        if (start < 0 || end < start || (size_t)end > text_length) {
            set_error("token offsets fall outside the source text");
            return OPENMED_STATUS_INVALID_OUTPUT;
        }
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

#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
static openmed_status fixture_logits(
    openmed_runtime *runtime,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count
) {
    size_t token_index;
    size_t label_index;
    size_t entity_index = 1;
    if (runtime->label_count < 4) {
        set_error("the synthetic fixture requires four labels");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    *logits = (float *)malloc(
        batch->token_count * runtime->label_count * sizeof(float)
    );
    if (*logits == NULL) {
        return OPENMED_STATUS_OUT_OF_MEMORY;
    }
    for (token_index = 0; token_index < batch->token_count; ++token_index) {
        size_t selected = 0;
        if (batch->start_offsets[token_index] != batch->end_offsets[token_index] &&
            entity_index < 4) {
            selected = entity_index++;
        }
        for (label_index = 0; label_index < runtime->label_count; ++label_index) {
            (*logits)[token_index * runtime->label_count + label_index] =
                label_index == selected ? 5.0f : 0.0f;
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
#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
    if (runtime->fixture_session) {
        return fixture_logits(
            runtime,
            batch,
            logits,
            sequence_length,
            label_count
        );
    }
#endif
    return runtime->session_api.run(
        runtime->session,
        batch,
        logits,
        sequence_length,
        label_count,
        openmed_error,
        sizeof(openmed_error)
    );
}

static int use_test_session(void) {
#if defined(OPENMED_FFI_ENABLE_TEST_SESSION)
    const char *value = getenv("OPENMED_FFI_TEST_SESSION");
    return value != NULL && strcmp(value, "1") == 0;
#else
    return 0;
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
    if (threshold < 0.0 || threshold > 1.0) {
        set_error("threshold must be between zero and one");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    spans->items = NULL;
    spans->count = 0;
    return OPENMED_STATUS_OK;
}

uint32_t openmed_ffi_abi_version(void) {
    return OPENMED_FFI_ABI_VERSION;
}

openmed_status openmed_register_session_api(const openmed_session_api *api) {
    if (api == NULL || api->abi_version != OPENMED_FFI_ABI_VERSION ||
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
    const char *const *labels,
    size_t label_count,
    openmed_runtime **runtime
) {
    openmed_runtime *created;
    openmed_status status;
    if (model_path == NULL || model_path[0] == '\0' || labels == NULL ||
        label_count == 0 || runtime == NULL) {
        set_error("model path, labels, and runtime output are required");
        return OPENMED_STATUS_INVALID_ARGUMENT;
    }
    *runtime = NULL;
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
    created->fixture_session = use_test_session();
    if (!created->fixture_session) {
#if defined(OPENMED_FFI_ONNXRUNTIME)
        created->session_api = has_registered_session_api
            ? registered_session_api
            : onnx_session_api;
#else
        if (has_registered_session_api) {
            created->session_api = registered_session_api;
        } else {
            free_labels(created->labels, created->label_count);
            free(created);
            set_error(
                "no ONNX token-classification session adapter is registered"
            );
            return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
        }
#endif
        created->session = created->session_api.create(
            model_path,
            (const char *const *)created->labels,
            created->label_count,
            openmed_error,
            sizeof(openmed_error)
        );
        if (created->session == NULL) {
            free_labels(created->labels, created->label_count);
            free(created);
            if (openmed_error[0] == '\0') {
                set_error("the ONNX token-classification session could not be created");
            }
            return OPENMED_STATUS_RUNTIME_UNAVAILABLE;
        }
    }
    *runtime = created;
    set_error(NULL);
    return OPENMED_STATUS_OK;
}

void openmed_runtime_destroy(openmed_runtime *runtime) {
    if (runtime == NULL) {
        return;
    }
    if (!runtime->fixture_session && runtime->session != NULL) {
        runtime->session_api.destroy(runtime->session);
    }
    free_labels(runtime->labels, runtime->label_count);
    free(runtime);
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
    if (runtime->fixture_session) {
        free(logits);
    } else {
        runtime->session_api.release_logits(runtime->session, logits);
    }
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
