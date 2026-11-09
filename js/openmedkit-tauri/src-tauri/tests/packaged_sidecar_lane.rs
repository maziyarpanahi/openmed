//! Opt-in local-artifact lane for a real packaged sidecar.
//!
//! Not run by default. The `packaged-sidecar` job in
//! `.github/workflows/tauri-sidecar.yml` builds the PyInstaller sidecar,
//! provisions a local model directory, writes the manifest with
//! `openmed-sidecar-manifest`, and then runs:
//!
//! ```text
//! OPENMED_TAURI_LANE_SIDECAR=<executable> \
//! OPENMED_TAURI_LANE_MANIFEST=<openmed-sidecar.json> \
//! OPENMED_TAURI_LANE_MODEL_ROOT=<model directory> \
//! OPENMED_SIDECAR_HASH_SECRET=sidecar-golden-secret \
//! cargo test --locked --test packaged_sidecar_lane -- --ignored
//! ```
//!
//! The lane compares the real sidecar with the shared direct reference vector
//! used by the Python stdio tests, runs one model-backed request as a leakage
//! check, and proves tampered or missing artifacts refuse before start. When
//! `OPENMED_TAURI_LANE_EVIDENCE` is set it writes value-free evidence there.

use std::{path::PathBuf, time::Duration};

use openmedkit_tauri_sidecar::{
    sidecar_command::{OpenMedSidecarState, SidecarDeidentifyOptions, SidecarDeidentifyRequest},
    sidecar_manifest::{SidecarBundle, SidecarManifest, SidecarTarget},
};
use serde_json::{json, Value};

const GOLDEN: &str =
    include_str!("../../../../tests/unit/service/fixtures/sidecar_stdio_golden.json");
const PHI_FRAGMENTS: [&str; 2] = ["425-555-0100", "rowan@example.test"];

fn lane_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must be set for the packaged sidecar lane"))
}

fn lane_state(bundle: SidecarBundle) -> OpenMedSidecarState {
    OpenMedSidecarState::from_bundle(bundle)
        .with_startup_timeout(Duration::from_secs(300))
        .with_request_timeout(Duration::from_secs(300))
}

#[test]
#[ignore = "local-artifact lane: needs a packaged PyInstaller sidecar and a local model"]
fn packaged_sidecar_matches_shared_vectors_on_this_platform() {
    let sidecar = lane_path("OPENMED_TAURI_LANE_SIDECAR");
    let model_root = lane_path("OPENMED_TAURI_LANE_MODEL_ROOT");
    let manifest_path = lane_path("OPENMED_TAURI_LANE_MANIFEST");
    let bundle = SidecarBundle::from_manifest_file(&manifest_path, &sidecar, &model_root)
        .expect("manifest readable");
    let manifest_bytes = std::fs::read(&manifest_path).expect("manifest");
    let pinned = bundle.manifest_sha256();
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden vector");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let qualification = runtime.block_on(async {
        let state = lane_state(bundle.with_pinned_manifest_sha256(pinned.clone()));
        let qualification = state.qualify().await.expect("packaged sidecar qualifies");
        assert!(state.ping().await.expect("ping").offline);

        let request = &golden["request"];
        let options = &request["options"];
        let result = state
            .deidentify(SidecarDeidentifyRequest {
                text: request["text"].as_str().expect("text").to_owned(),
                options: SidecarDeidentifyOptions {
                    policy: options["policy"].as_str().map(str::to_owned),
                    doc_id: options["doc_id"].as_str().map(str::to_owned),
                    deterministic_only: options["deterministic_only"].as_bool(),
                    ..SidecarDeidentifyOptions::default()
                },
            })
            .await
            .expect("deterministic request");
        let expected = &golden["expected"];
        assert_eq!(result.deidentified_text, expected["deidentified_text"]);
        let actual = serde_json::to_value(&result.spans).expect("spans");
        let expected_spans = expected["spans"].as_array().expect("expected spans");
        assert_eq!(actual.as_array().map(Vec::len), Some(expected_spans.len()));
        let tolerance = expected["score_tolerance"].as_f64().expect("tolerance");
        for (actual_span, expected_span) in actual.as_array().unwrap().iter().zip(expected_spans) {
            for (key, value) in expected_span.as_object().expect("span") {
                if key == "score" {
                    let delta = actual_span[key].as_f64().expect("score") - value.as_f64().unwrap();
                    assert!(delta.abs() <= tolerance, "score drift on {key}");
                } else {
                    assert_eq!(&actual_span[key], value, "span field {key}");
                }
            }
        }

        let model_backed = state
            .deidentify(SidecarDeidentifyRequest {
                text: request["text"].as_str().expect("text").to_owned(),
                options: SidecarDeidentifyOptions {
                    policy: Some("hipaa_safe_harbor".to_owned()),
                    doc_id: Some("synthetic-tauri-lane".to_owned()),
                    ..SidecarDeidentifyOptions::default()
                },
            })
            .await
            .expect("model-backed request");
        for fragment in PHI_FRAGMENTS {
            assert!(
                !model_backed.deidentified_text.contains(fragment),
                "direct identifier leaked"
            );
        }

        assert!(state.shutdown().await.expect("shutdown").shutdown);
        state.terminate().await;
        qualification
    });

    let mut manifest = SidecarManifest::parse(&manifest_bytes).expect("parse");
    manifest.model.files[0].sha256 = "0".repeat(64);
    let tampered = SidecarBundle::new(
        manifest.to_canonical_bytes().expect("encode"),
        &sidecar,
        &model_root,
    );
    let error = runtime
        .block_on(lane_state(tampered).ping())
        .expect_err("tampered model refuses");
    assert_eq!(error.code, "SIDECAR_ARTIFACT_MISMATCH");

    let missing = SidecarBundle::new(manifest_bytes.clone(), &sidecar, model_root.join("absent"));
    let error = runtime
        .block_on(lane_state(missing).ping())
        .expect_err("missing model refuses");
    assert_eq!(error.code, "SIDECAR_ARTIFACT_MISSING");

    if let Some(path) = std::env::var_os("OPENMED_TAURI_LANE_EVIDENCE") {
        let target = SidecarTarget::host();
        let evidence = json!({
            "schema": "openmed.tauri.sidecar_lane_evidence",
            "schema_version": 1,
            "platform": target.platform,
            "arch": target.arch,
            "manifest_sha256": qualification.manifest_sha256,
            "files_verified": qualification.files_verified,
            "protocol_version": qualification.protocol_version,
            "golden_vectors_passed": 1,
            "leakage_checks_passed": 1,
            "negative_controls_passed": 2,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).expect("encode"))
            .expect("write evidence");
    }
}
