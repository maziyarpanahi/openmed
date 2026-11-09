//! Synthetic stand-in for the packaged OpenMed sidecar.
//!
//! The qualification tests copy this executable into a temporary bundle as
//! `openmed-sidecar`, pin it in a manifest, and let the real host start it.
//! It speaks the JSON-lines protocol and misbehaves on purpose according to a
//! `fake-sidecar-mode` file next to the executable. Its engine is a direct
//! replay of the shared offset-contract vectors: for `doc_id`
//! `parity:<case id>` it masks the reference span by Unicode code point.
//! It handles only synthetic text and is never bundled into an application.

use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    process, thread,
    time::Duration,
};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const PARITY_VECTORS: &str = include_str!("../../../../tests/fixtures/parity/offset_contract.json");
const RESPONSE_LIMIT_BYTES: usize = 64_000_000;
const SYNTHETIC_NAME: &str = "Rowan Hale";

fn main() {
    let directory = bundle_directory();
    let mode = fs::read_to_string(directory.join("fake-sidecar-mode"))
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|_| "ok".to_owned());
    append(
        &directory.join("fake-sidecar-starts"),
        &format!("{}\n", process::id()),
    );

    match mode.as_str() {
        "startup_failure" => {
            eprintln!("Traceback: synthetic failure for {SYNTHETIC_NAME}");
            eprintln!(r#"{{"error_code":"SIDECAR_START_FAILED","event":"sidecar_failed"}}"#);
            process::exit(1);
        }
        "banner" => {
            println!("OpenMed sidecar ready");
        }
        _ => {}
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { process::exit(0) };
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = request["id"].as_str().unwrap_or_default().to_owned();
        match request["operation"].as_str() {
            Some("ping") => respond(&id, ping_result(&mode)),
            Some("shutdown") => {
                respond(&id, json!({"shutdown": true}));
                process::exit(0);
            }
            Some("deidentify") => deidentify(&directory, &mode, &id, &request),
            _ => respond_error(&id, "INVALID_REQUEST"),
        }
    }
}

fn deidentify(directory: &std::path::Path, mode: &str, id: &str, request: &Value) {
    match mode {
        "crash_on_deidentify" => process::exit(3),
        "crash_once" => {
            let marker = directory.join("fake-sidecar-crashed");
            if !marker.exists() {
                append(&marker, "crashed\n");
                process::exit(3);
            }
        }
        "partial_line" => {
            print!(r#"{{"id":"{id}","ok":tr"#);
            let _ = io::stdout().flush();
            process::exit(2);
        }
        "stderr_failure" => {
            eprintln!("Traceback: synthetic failure while redacting {SYNTHETIC_NAME}");
            eprintln!(r#"{{"error_code":"SIDECAR_RUNTIME_FAILED","event":"sidecar_failed"}}"#);
            process::exit(1);
        }
        "stderr_flood" => {
            let noise = format!(
                "synthetic diagnostic about {SYNTHETIC_NAME} {}",
                "x".repeat(1_000)
            );
            let stderr = io::stderr();
            let mut stderr = stderr.lock();
            for _ in 0..2_048 {
                let _ = writeln!(stderr, "{noise}");
            }
        }
        "slow" => thread::sleep(Duration::from_secs(30)),
        "delayed" => thread::sleep(Duration::from_millis(1_500)),
        "stale_generation" => {
            let sequence = id.rsplit(':').next().unwrap_or("0");
            let stale = format!("{}:{sequence}", "f".repeat(32));
            respond(&stale, engine(request));
            return;
        }
        "wrong_id" => {
            let generation = id.split(':').next().unwrap_or_default();
            respond(&format!("{generation}:999999999"), engine(request));
            return;
        }
        "oversized" => {
            write_oversized(id);
            return;
        }
        "malformed" => {
            println!("{{not json for {SYNTHETIC_NAME}");
            let _ = io::stdout().flush();
            return;
        }
        "unknown_field" => {
            write_line(
                &json!({"id": id, "ok": true, "result": engine(request), "debug": SYNTHETIC_NAME}),
            );
            return;
        }
        "invalid_spans" => {
            let mut result = engine(request);
            let span = result["spans"][0].clone();
            result["spans"] = json!([span.clone(), span]);
            respond(id, result);
            return;
        }
        _ => {}
    }
    respond(id, engine(request));
}

fn ping_result(mode: &str) -> Value {
    let offline = ["OPENMED_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
        .iter()
        .all(|name| std::env::var(name).as_deref() == Ok("1"))
        && std::env::var_os("PYTHONPATH").is_none()
        && std::env::var_os("OPENMED_SIDECAR_MODEL")
            .map(PathBuf::from)
            .is_some_and(|path| path.is_dir());
    let protocol_version = if mode == "protocol_two" { 2 } else { 1 };
    json!({"offline": offline && mode != "online", "protocol_version": protocol_version})
}

/// Direct reference engine: mask the shared-vector span, or the synthetic
/// name, in Unicode code-point coordinates.
fn engine(request: &Value) -> Value {
    let text = request["text"].as_str().unwrap_or_default();
    let doc_id = request["options"]["doc_id"]
        .as_str()
        .unwrap_or("synthetic-fake")
        .to_owned();
    let characters: Vec<char> = text.chars().collect();
    let found = if let Some(case_id) = doc_id.strip_prefix("parity:") {
        parity_span(case_id)
    } else {
        text.find(SYNTHETIC_NAME).map(|byte_start| {
            let start = text[..byte_start].chars().count();
            (
                start,
                start + SYNTHETIC_NAME.chars().count(),
                "[NAME]".to_owned(),
            )
        })
    };
    let Some((start, end, replacement)) =
        found.filter(|(start, end, _)| start < end && *end <= characters.len())
    else {
        return json!({"deidentified_text": text, "spans": []});
    };
    let surface: String = characters[start..end].iter().collect();
    let redacted: String = characters[..start].iter().collect::<String>()
        + &replacement
        + &characters[end..].iter().collect::<String>();
    json!({
        "deidentified_text": redacted,
        "spans": [{
            "schema_version": 1,
            "doc_id": doc_id,
            "start": start,
            "end": end,
            "text_hash": format!("hmac-sha256:{:x}", Sha256::digest(surface.as_bytes())),
            "entity_type": "person",
            "canonical_label": "NAME",
            "policy_label": "DIRECT_IDENTIFIER",
            "regulatory_tags": [],
            "score": 0.99,
            "detector": "fake-reference-engine",
            "evidence": {},
            "action": "mask",
            "replacement": replacement,
            "reversible_id": null,
            "section": null,
            "metadata": {},
        }],
    })
}

fn parity_span(case_id: &str) -> Option<(usize, usize, String)> {
    let vectors: Value = serde_json::from_str(PARITY_VECTORS).ok()?;
    let case = vectors["cases"]
        .as_array()?
        .iter()
        .find(|case| case["id"].as_str() == Some(case_id))?;
    Some((
        usize::try_from(case["expected_start"].as_u64()?).ok()?,
        usize::try_from(case["expected_end"].as_u64()?).ok()?,
        case["replacement"].as_str()?.to_owned(),
    ))
}

fn respond(id: &str, result: Value) {
    write_line(&json!({"id": id, "ok": true, "result": result}));
}

fn respond_error(id: &str, code: &str) {
    write_line(
        &json!({"id": id, "ok": false, "error": {"code": code, "message": "Invalid sidecar request."}}),
    );
}

fn write_line(value: &Value) {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    if writeln!(stdout, "{value}")
        .and_then(|()| stdout.flush())
        .is_err()
    {
        process::exit(0);
    }
}

fn write_oversized(id: &str) {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let chunk = vec![b'x'; 1 << 20];
    let mut written = 0;
    let header = format!(r#"{{"id":"{id}","ok":true,"result":{{"deidentified_text":""#);
    if stdout.write_all(header.as_bytes()).is_err() {
        process::exit(0);
    }
    while written <= RESPONSE_LIMIT_BYTES {
        if stdout.write_all(&chunk).is_err() {
            process::exit(0);
        }
        written += chunk.len();
    }
    let _ = stdout.write_all(b"\",\"spans\":[]}}\n");
    let _ = stdout.flush();
}

fn bundle_directory() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn append(path: &std::path::Path, line: &str) {
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}
