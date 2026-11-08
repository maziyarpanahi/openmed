/*
 * Native checks for the OpenMed Flutter FFI shim.
 *
 * The shim is compiled into this executable with the synthetic fixture
 * session enabled, so these checks need no ONNX Runtime and no model. Every
 * model file is a synthetic placeholder written to a temporary directory.
 */
#include "../openmed_ffi.c"

#if defined(_WIN32)
#include <process.h>
#define openmed_test_getpid _getpid
#else
#include <unistd.h>
#define openmed_test_getpid getpid
#endif

static int failures;
static char model_path[1024];
static char other_path[1024];

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            ++failures;                                                      \
            (void)fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__,    \
                          __LINE__, #condition);                             \
        }                                                                    \
    } while (0)

static void set_fixture_session(int enabled) {
#if defined(_WIN32)
    (void)_putenv_s("OPENMED_FFI_TEST_SESSION", enabled ? "1" : "");
#else
    if (enabled) {
        (void)setenv("OPENMED_FFI_TEST_SESSION", "1", 1);
    } else {
        (void)unsetenv("OPENMED_FFI_TEST_SESSION");
    }
#endif
}

static void digest_text(
    const unsigned char *data,
    size_t length,
    char output[7 + OPENMED_SHA256_BYTES * 2 + 1]
) {
    static const char digits[] = "0123456789abcdef";
    openmed_sha256 hasher;
    unsigned char digest[OPENMED_SHA256_BYTES];
    size_t index;
    sha256_init(&hasher);
    sha256_update(&hasher, data, length);
    sha256_final(&hasher, digest);
    memcpy(output, "sha256:", 7);
    for (index = 0; index < OPENMED_SHA256_BYTES; ++index) {
        output[7 + index * 2] = digits[digest[index] >> 4];
        output[8 + index * 2] = digits[digest[index] & 0x0fU];
    }
    output[7 + OPENMED_SHA256_BYTES * 2] = '\0';
}

static void write_file(const char *path, const char *content) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    if (file != NULL) {
        (void)fwrite(content, 1, strlen(content), file);
        (void)fclose(file);
    }
}

static void check_sha256_vectors(void) {
    /* FIPS 180-4 examples plus a message that crosses the padding boundary. */
    static const char million_prefix[] =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    char text[7 + OPENMED_SHA256_BYTES * 2 + 1];
    unsigned char block[1000];
    openmed_sha256 hasher;
    unsigned char digest[OPENMED_SHA256_BYTES];
    unsigned char expected[OPENMED_SHA256_BYTES];
    size_t index;

    digest_text((const unsigned char *)"", 0, text);
    CHECK(strcmp(text, "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0);
    digest_text((const unsigned char *)"abc", 3, text);
    CHECK(strcmp(text, "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);
    digest_text((const unsigned char *)million_prefix, strlen(million_prefix), text);
    CHECK(strcmp(text, "sha256:248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0);

    /* One million 'a' bytes, fed in uneven chunks. */
    memset(block, 'a', sizeof(block));
    sha256_init(&hasher);
    for (index = 0; index < 1000; ++index) {
        sha256_update(&hasher, block, 997);
        sha256_update(&hasher, block, 3);
    }
    sha256_final(&hasher, digest);
    CHECK(parse_sha256(
        "sha256:cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        expected
    ));
    CHECK(digests_equal(digest, expected));

    CHECK(!parse_sha256(NULL, expected));
    CHECK(!parse_sha256("", expected));
    CHECK(!parse_sha256("md5:00", expected));
    CHECK(!parse_sha256(
        "sha256:CDC76E5C9914FB9281A1C7E284D73E67F1809A48A497200E046D39CCC7112CD0",
        expected
    ));
    CHECK(!parse_sha256(
        "sha256:cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd",
        expected
    ));
}

static const char *const labels[] = {
    "O", "B-PERSON", "B-DATE_OF_BIRTH", "B-EMAIL", "B-NAME", "I-NAME"
};

static void check_error_is_value_free(void) {
    const char *message = openmed_last_error();
    CHECK(strstr(message, model_path) == NULL);
    CHECK(strstr(message, other_path) == NULL);
    CHECK(strstr(message, "sha256:") == NULL);
    CHECK(strstr(message, "synthetic") == NULL);
    CHECK(strstr(message, "Alice") == NULL);
}

static void check_verified_creation(void) {
    char digest[7 + OPENMED_SHA256_BYTES * 2 + 1];
    char wrong[7 + OPENMED_SHA256_BYTES * 2 + 1];
    openmed_runtime *runtime = (openmed_runtime *)1;
    const char *model = "synthetic fixture model placeholder\n";

    write_file(model_path, model);
    digest_text((const unsigned char *)model, strlen(model), digest);
    digest_text((const unsigned char *)"tampered", 8, wrong);
    set_fixture_session(1);

    CHECK(openmed_runtime_create(model_path, NULL, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_INVALID_ARGUMENT);
    CHECK(runtime == NULL);
    CHECK(openmed_runtime_create(model_path, "sha256:00", labels, 6, 0, &runtime) ==
          OPENMED_STATUS_INVALID_ARGUMENT);
    CHECK(openmed_runtime_create(model_path, digest, labels, 1, 0, &runtime) ==
          OPENMED_STATUS_INVALID_ARGUMENT);
    CHECK(openmed_runtime_create(model_path, wrong, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_INTEGRITY_FAILED);
    CHECK(runtime == NULL);
    CHECK(openmed_runtime_live_count() == 0);
    check_error_is_value_free();
    CHECK(openmed_runtime_create(other_path, digest, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_RUNTIME_UNAVAILABLE);
    CHECK(runtime == NULL);
    check_error_is_value_free();

    CHECK(openmed_runtime_create(model_path, digest, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_OK);
    CHECK(runtime != NULL);
    CHECK(openmed_runtime_session_kind(runtime) == OPENMED_SESSION_KIND_FIXTURE);
    CHECK(openmed_runtime_live_count() == 1);
    openmed_runtime_destroy(runtime);
    CHECK(openmed_runtime_live_count() == 0);
    /* A second destroy of a freed handle is a no-op. */
    openmed_runtime_destroy(runtime);
    CHECK(openmed_runtime_live_count() == 0);
    openmed_runtime_destroy(NULL);
}

static void check_fixture_deidentification(void) {
    static const char text[] =
        "Patient Alice Nguyen was born on 1979-04-12. Email alice@example.org.";
    static const int64_t ids[] = {101, 1001, 1002, 1003, 102};
    static const int64_t mask[] = {1, 1, 1, 1, 1};
    static const int64_t starts[] = {0, 8, 33, 51, 0};
    static const int64_t ends[] = {0, 20, 43, 68, 0};
    char digest[7 + OPENMED_SHA256_BYTES * 2 + 1];
    const char *model = "synthetic fixture model placeholder\n";
    openmed_token_batch batch;
    openmed_span_list spans;
    openmed_runtime *runtime = NULL;
    char *output = NULL;

    digest_text((const unsigned char *)model, strlen(model), digest);
    set_fixture_session(1);
    CHECK(openmed_runtime_create(model_path, digest, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_OK);
    if (runtime == NULL) {
        return;
    }
    batch.input_ids = ids;
    batch.attention_mask = mask;
    batch.start_offsets = starts;
    batch.end_offsets = ends;
    batch.token_count = 5;
    CHECK(openmed_runtime_deidentify(runtime, text, &batch, 0.5, &output, &spans) ==
          OPENMED_STATUS_OK);
    CHECK(output != NULL && strcmp(
        output,
        "Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL]."
    ) == 0);
    CHECK(spans.count == 3);
    if (spans.count == 3) {
        CHECK(spans.items[0].start == 8 && spans.items[0].end == 20);
        CHECK(strcmp(spans.items[1].label, "DATE_OF_BIRTH") == 0);
        CHECK(spans.items[2].start == 51 && spans.items[2].end == 68);
    }
    openmed_string_free(output);
    openmed_span_list_free(&spans);

    /* Thresholds outside [0, 1], including NaN, are rejected. */
    CHECK(openmed_runtime_extract_pii(runtime, text, &batch, 1.5, &spans) ==
          OPENMED_STATUS_INVALID_ARGUMENT);
    CHECK(openmed_runtime_extract_pii(runtime, text, &batch, NAN, &spans) ==
          OPENMED_STATUS_INVALID_ARGUMENT);
    openmed_runtime_destroy(runtime);
    CHECK(openmed_runtime_live_count() == 0);
}

static void check_owner_release(void) {
    char digest[7 + OPENMED_SHA256_BYTES * 2 + 1];
    const char *model = "synthetic fixture model placeholder\n";
    uint64_t owner = openmed_runtime_owner_create();
    uint64_t other = openmed_runtime_owner_create();
    openmed_runtime *first = NULL;
    openmed_runtime *second = NULL;
    openmed_runtime *unowned = NULL;

    digest_text((const unsigned char *)model, strlen(model), digest);
    set_fixture_session(1);
    CHECK(owner != 0 && other != 0 && owner != other);
    CHECK(openmed_runtime_create(model_path, digest, labels, 6, owner, &first) ==
          OPENMED_STATUS_OK);
    CHECK(openmed_runtime_create(model_path, digest, labels, 6, owner, &second) ==
          OPENMED_STATUS_OK);
    CHECK(openmed_runtime_create(model_path, digest, labels, 6, 0, &unowned) ==
          OPENMED_STATUS_OK);
    CHECK(openmed_runtime_live_count() == 3);

    /* The owner closed one runtime itself before it went away. */
    openmed_runtime_destroy(first);
    CHECK(openmed_runtime_owner_release(other) == 0);
    CHECK(openmed_runtime_owner_release(owner) == 1);
    CHECK(openmed_runtime_owner_release(owner) == 0);
    CHECK(openmed_runtime_owner_release(0) == 0);
    CHECK(openmed_runtime_live_count() == 1);
    openmed_runtime_destroy(unowned);
    CHECK(openmed_runtime_live_count() == 0);
    (void)second;
}

static int adapter_creates;
static int adapter_destroys;
static int adapter_fail_create;

static void *adapter_create(
    const char *path,
    const char *const *adapter_labels,
    size_t count,
    char *error,
    size_t capacity
) {
    (void)adapter_labels;
    (void)count;
    if (adapter_fail_create) {
        /* Adapter text that quotes the path must never reach callers. */
        (void)snprintf(error, capacity, "cannot open %s", path);
        return NULL;
    }
    ++adapter_creates;
    return &adapter_creates;
}

static openmed_status adapter_run(
    void *session,
    const openmed_token_batch *batch,
    float **logits,
    size_t *sequence_length,
    size_t *label_count,
    char *error,
    size_t capacity
) {
    (void)session;
    (void)batch;
    (void)logits;
    (void)sequence_length;
    (void)label_count;
    (void)snprintf(error, capacity, "%s", "Alice Nguyen");
    return OPENMED_STATUS_INFERENCE_FAILED;
}

static void adapter_release(void *session, float *logits) {
    (void)session;
    free(logits);
}

static void adapter_destroy(void *session) {
    (void)session;
    ++adapter_destroys;
}

static void check_registered_adapter(void) {
    static const openmed_session_api api = {
        OPENMED_FFI_SESSION_API_VERSION,
        adapter_create,
        adapter_run,
        adapter_release,
        adapter_destroy
    };
    static const openmed_session_api stale = {
        99,
        adapter_create,
        adapter_run,
        adapter_release,
        adapter_destroy
    };
    static const char text[] = "Patient Alice Nguyen";
    static const int64_t ids[] = {1001};
    static const int64_t mask[] = {1};
    static const int64_t starts[] = {8};
    static const int64_t ends[] = {20};
    char digest[7 + OPENMED_SHA256_BYTES * 2 + 1];
    char wrong[7 + OPENMED_SHA256_BYTES * 2 + 1];
    const char *model = "synthetic fixture model placeholder\n";
    openmed_token_batch batch;
    openmed_span_list spans;
    openmed_runtime *runtime = NULL;

    digest_text((const unsigned char *)model, strlen(model), digest);
    digest_text((const unsigned char *)"tampered", 8, wrong);
    set_fixture_session(0);
    CHECK(openmed_register_session_api(&stale) == OPENMED_STATUS_INVALID_ARGUMENT);
    CHECK(openmed_register_session_api(&api) == OPENMED_STATUS_OK);
    CHECK((openmed_ffi_capabilities() & OPENMED_FFI_CAPABILITY_REGISTERED_ADAPTER) != 0);

    /* A digest mismatch is rejected before the adapter sees the path. */
    CHECK(openmed_runtime_create(model_path, wrong, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_INTEGRITY_FAILED);
    CHECK(adapter_creates == 0);

    adapter_fail_create = 1;
    CHECK(openmed_runtime_create(model_path, digest, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_RUNTIME_UNAVAILABLE);
    check_error_is_value_free();
    CHECK(openmed_runtime_live_count() == 0);
    adapter_fail_create = 0;

    CHECK(openmed_runtime_create(model_path, digest, labels, 6, 0, &runtime) ==
          OPENMED_STATUS_OK);
    CHECK(adapter_creates == 1);
    CHECK(openmed_runtime_session_kind(runtime) ==
          OPENMED_SESSION_KIND_REGISTERED_ADAPTER);
    batch.input_ids = ids;
    batch.attention_mask = mask;
    batch.start_offsets = starts;
    batch.end_offsets = ends;
    batch.token_count = 1;
    CHECK(openmed_runtime_extract_pii(runtime, text, &batch, 0.5, &spans) ==
          OPENMED_STATUS_INFERENCE_FAILED);
    CHECK(strstr(openmed_last_error(), "Alice") == NULL);
    openmed_runtime_destroy(runtime);
    openmed_runtime_destroy(runtime);
    CHECK(adapter_destroys == 1);
    CHECK(openmed_runtime_live_count() == 0);
}

int main(void) {
    const char *directory = getenv("OPENMED_FFI_TEST_TMPDIR");
#if defined(_WIN32)
    if (directory == NULL) {
        directory = getenv("TEMP");
    }
#endif
    if (directory == NULL || directory[0] == '\0') {
        directory = "/tmp";
    }
    (void)snprintf(
        model_path,
        sizeof(model_path),
        "%s/openmed-ffi-test-%d-model.onnx",
        directory,
        (int)openmed_test_getpid()
    );
    (void)snprintf(
        other_path,
        sizeof(other_path),
        "%s/openmed-ffi-test-%d-missing.onnx",
        directory,
        (int)openmed_test_getpid()
    );
    (void)remove(other_path);

    check_sha256_vectors();
    check_verified_creation();
    check_fixture_deidentification();
    check_owner_release();
    check_registered_adapter();

    (void)remove(model_path);
    if (failures != 0) {
        (void)fprintf(stderr, "%d native check(s) failed\n", failures);
        return 1;
    }
    (void)printf("openmed_ffi native checks passed\n");
    return 0;
}
