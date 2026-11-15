//! Fake-process qualification of the packaged Tauri sidecar host.
//!
//! Each test copies the synthetic `openmed-fake-sidecar` executable into a
//! temporary bundle as `openmed-sidecar`, writes a synthetic model directory,
//! pins both in a generated manifest, and drives the real host code: manifest
//! verification, process start, generation binding, bounded reads, and
//! teardown. No test loads a model or touches the network.

use std::{
    fs,
    future::Future,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use openmedkit_tauri_sidecar::{
    sidecar_command::{
        OpenMedSidecarState, SidecarCommandError, SidecarDeidentifyOptions,
        SidecarDeidentifyRequest,
    },
    sidecar_manifest::{generate_manifest, SidecarBundle, SidecarManifest, SidecarTarget},
};
use serde_json::Value;

const FAKE_SIDECAR: &str = env!("CARGO_BIN_EXE_openmed-fake-sidecar");
const PARITY_VECTORS: &str = include_str!("../../../../tests/fixtures/parity/offset_contract.json");
const SYNTHETIC_NOTE: &str = "Synthetic patient Rowan Hale called back.";

static NEXT_BUNDLE: AtomicUsize = AtomicUsize::new(0);

struct FakeBundle {
    directory: PathBuf,
    sidecar: PathBuf,
    model_root: PathBuf,
    manifest: Vec<u8>,
}

impl Drop for FakeBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

impl FakeBundle {
    fn new(mode: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "openmed-tauri-qualification-{}-{}",
            std::process::id(),
            NEXT_BUNDLE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&directory);
        let model_root = directory.join("model");
        fs::create_dir_all(model_root.join("onnx")).expect("bundle directory");
        let sidecar = directory.join(if cfg!(windows) {
            "openmed-sidecar.exe"
        } else {
            "openmed-sidecar"
        });
        fs::copy(FAKE_SIDECAR, &sidecar).expect("copy fake sidecar");
        fs::write(directory.join("fake-sidecar-mode"), mode).expect("mode");
        fs::write(
            model_root.join("config.json"),
            br#"{"model_type":"synthetic"}"#,
        )
        .expect("config");
        fs::write(model_root.join("tokenizer.json"), br#"{"synthetic":true}"#).expect("tokenizer");
        fs::write(model_root.join("onnx/model.onnx"), b"synthetic weights").expect("weights");
        let manifest = generate_manifest(
            &sidecar,
            &model_root,
            "synthetic/fake-model",
            &SidecarTarget::host(),
        )
        .expect("manifest");
        Self {
            directory,
            sidecar,
            model_root,
            manifest,
        }
    }

    fn bundle(&self) -> SidecarBundle {
        SidecarBundle::new(self.manifest.clone(), &self.sidecar, &self.model_root)
    }

    fn bundle_with(&self, edit: impl FnOnce(&mut SidecarManifest)) -> SidecarBundle {
        let mut manifest = SidecarManifest::parse(&self.manifest).expect("parse");
        edit(&mut manifest);
        SidecarBundle::new(
            manifest.to_canonical_bytes().expect("encode"),
            &self.sidecar,
            &self.model_root,
        )
    }

    fn state(&self) -> OpenMedSidecarState {
        state_for(self.bundle())
    }

    fn starts(&self) -> usize {
        fs::read_to_string(self.directory.join("fake-sidecar-starts"))
            .map(|log| log.lines().count())
            .unwrap_or(0)
    }

    fn redacted_path_fragments(&self) -> Vec<String> {
        vec![
            self.directory.display().to_string(),
            "model.onnx".to_owned(),
            "tokenizer.json".to_owned(),
        ]
    }
}

fn state_for(bundle: SidecarBundle) -> OpenMedSidecarState {
    OpenMedSidecarState::from_bundle(bundle)
        .with_request_timeout(Duration::from_secs(10))
        .with_startup_timeout(Duration::from_secs(20))
}

fn run<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(future)
}

fn note(text: &str, doc_id: Option<&str>) -> SidecarDeidentifyRequest {
    SidecarDeidentifyRequest {
        text: text.to_owned(),
        options: SidecarDeidentifyOptions {
            doc_id: doc_id.map(str::to_owned),
            ..SidecarDeidentifyOptions::default()
        },
    }
}

fn rendered(error: &SidecarCommandError) -> String {
    serde_json::to_string(error).expect("serialize error")
}

fn assert_error(error: &SidecarCommandError, code: &str, reason: Option<&str>) {
    assert_eq!(error.code, code, "{}", rendered(error));
    assert_eq!(error.reason.as_deref(), reason, "{}", rendered(error));
    let rendered = rendered(error);
    assert!(!rendered.contains("Rowan Hale"), "{rendered}");
    assert!(!rendered.contains("Traceback"), "{rendered}");
    assert!(!rendered.contains(std::path::MAIN_SEPARATOR), "{rendered}");
}

fn assert_no_bundle_values(bundle: &FakeBundle, error: &SidecarCommandError) {
    let rendered = rendered(error);
    for fragment in bundle.redacted_path_fragments() {
        assert!(!rendered.contains(&fragment), "{rendered}");
    }
    let manifest = SidecarManifest::parse(&bundle.manifest).expect("parse");
    assert!(!rendered.contains(&manifest.sidecar.sha256), "{rendered}");
    for file in manifest.model.files {
        assert!(!rendered.contains(&file.sha256), "{rendered}");
    }
}

fn refusal(bundle: &FakeBundle, state: OpenMedSidecarState) -> SidecarCommandError {
    let error = run(state.deidentify(note(SYNTHETIC_NOTE, None))).expect_err("refused");
    assert_eq!(bundle.starts(), 0, "the sidecar must not start");
    assert_no_bundle_values(bundle, &error);
    error
}

#[test]
fn qualified_bundle_starts_once_and_reports_value_free_evidence() {
    let bundle = FakeBundle::new("ok");
    let state = bundle.state();
    run(async {
        let qualification = state.qualify().await.expect("qualified");
        assert_eq!(qualification.generation.len(), 32);
        assert_eq!(
            qualification.manifest_sha256,
            bundle.bundle().manifest_sha256()
        );
        assert_eq!(qualification.files_verified, 3);
        assert_eq!(qualification.protocol_version, 1);
        assert_eq!(
            qualification.capabilities,
            ["deidentify", "ping", "shutdown"]
        );
        let evidence = serde_json::to_string(&qualification).expect("evidence");
        for fragment in bundle.redacted_path_fragments() {
            assert!(!evidence.contains(&fragment));
        }

        let ping = state.ping().await.expect("ping");
        assert!(ping.offline);
        let result = state
            .deidentify(note(SYNTHETIC_NOTE, Some("synthetic-qualification")))
            .await
            .expect("deidentify");
        assert_eq!(
            result.deidentified_text,
            "Synthetic patient [NAME] called back."
        );
        assert_eq!((result.spans[0].start, result.spans[0].end), (18, 28));
        assert_eq!(state.generation().await, Some(qualification.generation));
        assert!(state.shutdown().await.expect("shutdown").shutdown);
        assert_eq!(state.generation().await, None);
    });
    assert_eq!(bundle.starts(), 1);
}

#[test]
fn missing_binary_refuses_before_start() {
    let bundle = FakeBundle::new("ok");
    fs::remove_file(&bundle.sidecar).expect("remove sidecar");
    let error = refusal(&bundle, bundle.state());
    assert_error(&error, "SIDECAR_ARTIFACT_MISSING", Some("sidecar_missing"));
}

#[test]
fn changed_binary_refuses_before_start() {
    let bundle = FakeBundle::new("ok");
    let mut bytes = fs::read(&bundle.sidecar).expect("read sidecar");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&bundle.sidecar, &bytes).expect("flip a byte");
    let error = refusal(&bundle, bundle.state());
    assert_error(
        &error,
        "SIDECAR_ARTIFACT_MISMATCH",
        Some("sidecar_digest_mismatch"),
    );

    bytes.push(0);
    fs::write(&bundle.sidecar, &bytes).expect("grow");
    let error = refusal(&bundle, bundle.state());
    assert_error(
        &error,
        "SIDECAR_ARTIFACT_MISMATCH",
        Some("sidecar_size_mismatch"),
    );
}

#[test]
fn changed_missing_or_added_model_files_refuse_before_start() {
    type Mutation = fn(&Path);
    let cases: [(&str, Mutation, &str, &str); 5] = [
        (
            "changed",
            |root| fs::write(root.join("onnx/model.onnx"), b"synthetic weightz").unwrap(),
            "SIDECAR_ARTIFACT_MISMATCH",
            "model_file_digest_mismatch",
        ),
        (
            "truncated",
            |root| fs::write(root.join("tokenizer.json"), b"{}").unwrap(),
            "SIDECAR_ARTIFACT_MISMATCH",
            "model_file_size_mismatch",
        ),
        (
            "missing",
            |root| fs::remove_file(root.join("config.json")).unwrap(),
            "SIDECAR_ARTIFACT_MISSING",
            "model_file_missing",
        ),
        (
            "added",
            |root| fs::write(root.join("onnx/extra.onnx"), b"unlisted").unwrap(),
            "SIDECAR_ARTIFACT_MISMATCH",
            "model_file_unlisted",
        ),
        (
            "root",
            |root| fs::remove_dir_all(root).unwrap(),
            "SIDECAR_ARTIFACT_MISSING",
            "model_root_missing",
        ),
    ];
    for (label, mutate, code, reason) in cases {
        let bundle = FakeBundle::new("ok");
        mutate(&bundle.model_root);
        let error = refusal(&bundle, bundle.state());
        assert_eq!(error.reason.as_deref(), Some(reason), "{label}");
        assert_error(&error, code, Some(reason));
    }
}

#[test]
fn incompatible_or_tampered_manifests_refuse_before_start() {
    let bundle = FakeBundle::new("ok");
    let foreign = if cfg!(windows) { "linux" } else { "windows" };
    let cases: Vec<(SidecarBundle, &str, &str)> = vec![
        (
            bundle.bundle_with(|manifest| manifest.sidecar.platform = foreign.to_owned()),
            "SIDECAR_ARTIFACT_INCOMPATIBLE",
            "platform_mismatch",
        ),
        (
            bundle.bundle_with(|manifest| manifest.protocol_version = 2),
            "SIDECAR_ARTIFACT_INCOMPATIBLE",
            "protocol_unsupported",
        ),
        (
            bundle.bundle_with(|manifest| {
                manifest.capabilities.retain(|value| value != "deidentify")
            }),
            "SIDECAR_ARTIFACT_INCOMPATIBLE",
            "capability_missing",
        ),
        (
            bundle.bundle_with(|manifest| manifest.schema_version = 2),
            "SIDECAR_MANIFEST_INVALID",
            "manifest_schema_unsupported",
        ),
        (
            bundle.bundle_with(|manifest| manifest.model.files[0].sha256 = "0".repeat(64)),
            "SIDECAR_ARTIFACT_MISMATCH",
            "model_file_digest_mismatch",
        ),
        (
            SidecarBundle::new(
                serde_json::to_vec_pretty(
                    &serde_json::from_slice::<Value>(&bundle.manifest).unwrap(),
                )
                .unwrap(),
                &bundle.sidecar,
                &bundle.model_root,
            ),
            "SIDECAR_MANIFEST_INVALID",
            "manifest_not_canonical",
        ),
        (
            SidecarBundle::new(
                b"{\"schema\":".to_vec(),
                &bundle.sidecar,
                &bundle.model_root,
            ),
            "SIDECAR_MANIFEST_INVALID",
            "manifest_malformed",
        ),
        (
            bundle.bundle().with_pinned_manifest_sha256("0".repeat(64)),
            "SIDECAR_MANIFEST_INVALID",
            "manifest_digest_mismatch",
        ),
        (
            bundle.bundle_with(|manifest| {
                manifest.model.files[1].path = "../tokenizer.json".to_owned()
            }),
            "SIDECAR_MANIFEST_INVALID",
            "path_unsafe",
        ),
    ];
    for (candidate, code, reason) in cases {
        let error = refusal(&bundle, state_for(candidate));
        assert_error(&error, code, Some(reason));
    }

    let absent = SidecarBundle::from_manifest_file(
        &bundle.directory.join("absent.json"),
        &bundle.sidecar,
        &bundle.model_root,
    )
    .expect_err("absent manifest");
    assert_eq!(absent.code(), "SIDECAR_MANIFEST_INVALID");
    assert_eq!(absent.reason, "manifest_unreadable");

    let pinned = bundle.bundle().manifest_sha256();
    let state = state_for(bundle.bundle().with_pinned_manifest_sha256(pinned));
    run(state.qualify()).expect("pinned manifest qualifies");
}

#[test]
fn binary_replaced_between_generations_refuses_the_next_start() {
    let bundle = FakeBundle::new("ok");
    let state = bundle.state();
    run(async {
        state.qualify().await.expect("first generation");
        state.shutdown().await.expect("shutdown");
    });
    let mut bytes = fs::read(&bundle.sidecar).expect("read");
    bytes[0] ^= 0xff;
    fs::write(&bundle.sidecar, bytes).expect("replace");
    let error = run(state.ping()).expect_err("refused");
    assert_error(
        &error,
        "SIDECAR_ARTIFACT_MISMATCH",
        Some("sidecar_digest_mismatch"),
    );
    assert_eq!(bundle.starts(), 1);
}

#[test]
fn crash_mid_request_is_typed_and_the_next_request_gets_a_new_generation() {
    let bundle = FakeBundle::new("crash_once");
    let state = bundle.state();
    run(async {
        let first = state.qualify().await.expect("qualified").generation;
        let error = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect_err("crash");
        assert_error(&error, "SIDECAR_TERMINATED", Some("process_exited"));
        assert_eq!(state.generation().await, None);

        let result = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect("restarted");
        assert_eq!(result.spans.len(), 1);
        let second = state.generation().await.expect("running");
        assert_ne!(first, second);
    });
    assert_eq!(bundle.starts(), 2);
}

#[test]
fn partial_response_before_exit_is_terminated_not_parsed() {
    let bundle = FakeBundle::new("partial_line");
    let error = run(bundle.state().deidentify(note(SYNTHETIC_NOTE, None))).expect_err("partial");
    assert_error(&error, "SIDECAR_TERMINATED", Some("process_exited"));
}

#[test]
fn stderr_failure_maps_to_an_allow_listed_reason_without_copying_stderr() {
    let bundle = FakeBundle::new("stderr_failure");
    let error = run(bundle.state().deidentify(note(SYNTHETIC_NOTE, None))).expect_err("failure");
    assert_error(&error, "SIDECAR_TERMINATED", Some("sidecar_runtime_failed"));

    let bundle = FakeBundle::new("startup_failure");
    let error = run(bundle.state().ping()).expect_err("startup failure");
    assert_error(&error, "SIDECAR_TERMINATED", Some("sidecar_start_failed"));
    assert_eq!(bundle.starts(), 1);
}

#[test]
fn stderr_flood_is_drained_and_never_surfaced() {
    let bundle = FakeBundle::new("stderr_flood");
    let result = run(bundle.state().deidentify(note(SYNTHETIC_NOTE, None))).expect("drained");
    assert_eq!(
        result.deidentified_text,
        "Synthetic patient [NAME] called back."
    );
}

#[test]
fn stale_generation_responses_are_rejected_and_the_process_discarded() {
    let bundle = FakeBundle::new("stale_generation");
    let state = bundle.state();
    run(async {
        let error = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect_err("stale");
        assert_error(&error, "SIDECAR_PROTOCOL", Some("stale_generation"));
        assert_eq!(state.generation().await, None);
        let error = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect_err("still stale");
        assert_error(&error, "SIDECAR_PROTOCOL", Some("stale_generation"));
    });
    assert_eq!(bundle.starts(), 2);

    let bundle = FakeBundle::new("wrong_id");
    let error = run(bundle.state().deidentify(note(SYNTHETIC_NOTE, None))).expect_err("wrong id");
    assert_error(&error, "SIDECAR_PROTOCOL", Some("response_id_mismatch"));
}

#[test]
fn a_cancelled_request_never_leaks_its_late_response_to_the_next_caller() {
    let bundle = FakeBundle::new("delayed");
    let state = bundle.state();
    run(async {
        let first = state.qualify().await.expect("qualified").generation;
        let abandoned = tokio::time::timeout(
            Duration::from_millis(200),
            state.deidentify(note(SYNTHETIC_NOTE, None)),
        )
        .await;
        assert!(abandoned.is_err(), "the renderer dropped the request");

        let ping = state.ping().await.expect("fresh generation");
        assert!(ping.offline);
        let second = state.generation().await.expect("running");
        assert_ne!(first, second);
    });
    assert_eq!(bundle.starts(), 2);
}

#[test]
fn request_deadline_kills_and_restarts_the_sidecar() {
    let bundle = FakeBundle::new("slow");
    let state = state_for(bundle.bundle()).with_request_timeout(Duration::from_millis(300));
    run(async {
        let error = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect_err("deadline");
        assert_error(&error, "SIDECAR_TIMEOUT", None);
        assert_eq!(state.generation().await, None);
        state.ping().await.expect("restarted");
    });
    assert_eq!(bundle.starts(), 2);
}

#[test]
fn concurrent_requests_are_refused_while_one_is_in_flight() {
    let bundle = FakeBundle::new("delayed");
    let state = Arc::new(bundle.state());
    run(async {
        state.qualify().await.expect("qualified");
        let running = Arc::clone(&state);
        let first =
            tokio::spawn(async move { running.deidentify(note(SYNTHETIC_NOTE, None)).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let second = state.ping().await;
        first.await.expect("task").expect("first request completes");
        assert_error(&second.expect_err("busy"), "SIDECAR_BUSY", None);
    });
}

#[test]
fn terminate_interrupts_an_in_flight_request_and_kills_the_process() {
    let bundle = FakeBundle::new("slow");
    let state = Arc::new(bundle.state());
    let process_id = run(async {
        state.qualify().await.expect("qualified");
        let process_id = state.process_id().await.expect("running");
        let running = Arc::clone(&state);
        let request =
            tokio::spawn(async move { running.deidentify(note(SYNTHETIC_NOTE, None)).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        state.terminate().await;
        assert_error(
            &request.await.expect("task").expect_err("interrupted"),
            "SIDECAR_NOT_RUNNING",
            Some("host_terminated"),
        );
        assert_eq!(state.generation().await, None);
        assert_error(
            &state.ping().await.expect_err("closed"),
            "SIDECAR_NOT_RUNNING",
            Some("host_terminated"),
        );
        process_id
    });
    assert!(!process_is_alive(process_id));
    assert_eq!(bundle.starts(), 1);
}

#[test]
fn blocking_terminate_from_an_exit_handler_kills_the_process() {
    let bundle = FakeBundle::new("ok");
    let state = bundle.state();
    let process_id = tauri::async_runtime::block_on(async {
        state.qualify().await.expect("qualified");
        state.process_id().await.expect("running")
    });
    state.terminate_blocking();
    assert!(!process_is_alive(process_id));
    let error = tauri::async_runtime::block_on(state.ping()).expect_err("closed");
    assert_error(&error, "SIDECAR_NOT_RUNNING", Some("host_terminated"));
}

#[test]
fn oversized_responses_fail_closed() {
    let bundle = FakeBundle::new("oversized");
    let state = bundle.state();
    run(async {
        let error = state
            .deidentify(note(SYNTHETIC_NOTE, None))
            .await
            .expect_err("oversized");
        assert_error(&error, "SIDECAR_PROTOCOL", Some("response_too_large"));
        assert_eq!(state.generation().await, None);
    });
}

#[test]
fn malformed_responses_fail_closed() {
    for (mode, reason) in [
        ("malformed", "response_malformed"),
        ("unknown_field", "response_malformed"),
        ("invalid_spans", "result_invalid"),
    ] {
        let bundle = FakeBundle::new(mode);
        let state = bundle.state();
        run(async {
            let error = state
                .deidentify(note(SYNTHETIC_NOTE, None))
                .await
                .expect_err(mode);
            assert_error(&error, "SIDECAR_PROTOCOL", Some(reason));
            assert_eq!(state.generation().await, None, "{mode}");
        });
    }
}

#[test]
fn handshake_rejects_online_incompatible_or_noisy_sidecars() {
    for (mode, reason) in [
        ("online", "handshake_invalid"),
        ("protocol_two", "handshake_invalid"),
        ("banner", "response_malformed"),
    ] {
        let bundle = FakeBundle::new(mode);
        let state = bundle.state();
        run(async {
            let error = state.qualify().await.expect_err(mode);
            assert_error(&error, "SIDECAR_PROTOCOL", Some(reason));
            assert_eq!(state.generation().await, None, "{mode}");
        });
        assert_eq!(bundle.starts(), 1, "{mode}");
    }
}

#[test]
fn packaged_host_results_match_the_shared_parity_vectors() {
    let vectors: Value = serde_json::from_str(PARITY_VECTORS).expect("vectors");
    assert_eq!(vectors["offset_unit"], "unicode_scalar");
    let cases = vectors["cases"].as_array().expect("cases");
    assert!(cases.len() >= 40);
    let bundle = FakeBundle::new("ok");
    let state = bundle.state();
    run(async {
        for case in cases {
            let id = case["id"].as_str().expect("id");
            let doc_id = format!("parity:{id}");
            let result = state
                .deidentify(note(case["text"].as_str().expect("text"), Some(&doc_id)))
                .await
                .unwrap_or_else(|error| panic!("{id}: {}", rendered(&error)));
            assert_eq!(
                result.deidentified_text,
                case["expected_redacted"].as_str().expect("expected"),
                "{id}"
            );
            assert_eq!(result.spans.len(), 1, "{id}");
            let span = &result.spans[0];
            assert_eq!(
                (span.start as u64, span.end as u64),
                (
                    case["expected_start"].as_u64().expect("start"),
                    case["expected_end"].as_u64().expect("end")
                ),
                "{id}"
            );
            assert_eq!(
                span.replacement.as_deref(),
                case["replacement"].as_str(),
                "{id}"
            );
            assert_eq!(span.doc_id, doc_id, "{id}");
        }
    });
    assert_eq!(bundle.starts(), 1);
}

#[cfg(unix)]
fn process_is_alive(process_id: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(process_id.to_string())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
fn process_is_alive(process_id: u32) -> bool {
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {process_id}"), "/NH"])
        .output()
        .is_ok_and(|output| {
            String::from_utf8_lossy(&output.stdout).contains(&process_id.to_string())
        })
}
