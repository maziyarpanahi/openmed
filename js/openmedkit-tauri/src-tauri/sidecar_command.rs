//! Tauri v2 command wrapper for the persistent OpenMed sidecar process.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tokio::sync::{mpsc::Receiver, Mutex};

const SIDECAR_BINARY: &str = "openmed-sidecar";

#[derive(Default)]
pub struct OpenMedSidecarState {
    process: Mutex<Option<SidecarProcess>>,
    next_request_id: AtomicU64,
}

struct SidecarProcess {
    child: CommandChild,
    events: Receiver<CommandEvent>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarDeidentifyRequest {
    pub text: String,
    #[serde(default)]
    pub options: SidecarDeidentifyOptions,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarDeidentifyOptions {
    pub model_name: Option<String>,
    pub policy: Option<String>,
    pub method: Option<String>,
    pub confidence_threshold: Option<f64>,
    pub lang: Option<String>,
    pub doc_id: Option<String>,
    pub use_smart_merging: Option<bool>,
    pub use_safety_sweep: Option<bool>,
    pub deterministic_only: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SidecarDeidentifyResult {
    #[serde(rename(serialize = "deidentifiedText", deserialize = "deidentified_text"))]
    pub deidentified_text: String,
    pub spans: Vec<OpenMedSpan>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenMedSpan {
    pub schema_version: u32,
    pub doc_id: String,
    pub start: usize,
    pub end: usize,
    pub text_hash: String,
    pub entity_type: String,
    pub canonical_label: String,
    pub policy_label: Option<String>,
    pub regulatory_tags: Vec<String>,
    pub score: Option<f64>,
    pub detector: Option<String>,
    pub evidence: Value,
    pub action: String,
    pub replacement: Option<String>,
    pub reversible_id: Option<String>,
    pub section: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SidecarPingResult {
    pub offline: bool,
    #[serde(rename(serialize = "protocolVersion", deserialize = "protocol_version"))]
    pub protocol_version: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SidecarShutdownResult {
    pub shutdown: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarCommandError {
    pub code: String,
    pub message: String,
}

impl SidecarCommandError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    fn terminated() -> Self {
        Self::new(
            "SIDECAR_TERMINATED",
            "The OpenMed sidecar terminated before responding.",
        )
    }
}

#[derive(Debug, Deserialize)]
struct ProtocolResponse {
    id: String,
    ok: bool,
    result: Option<Value>,
    error: Option<ProtocolError>,
}

#[derive(Debug, Deserialize)]
struct ProtocolError {
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ProtocolOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_smart_merging: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_safety_sweep: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deterministic_only: Option<bool>,
}

impl From<SidecarDeidentifyOptions> for ProtocolOptions {
    fn from(options: SidecarDeidentifyOptions) -> Self {
        Self {
            model_name: options.model_name,
            policy: options.policy,
            method: options.method,
            confidence_threshold: options.confidence_threshold,
            lang: options.lang,
            doc_id: options.doc_id,
            use_smart_merging: options.use_smart_merging,
            use_safety_sweep: options.use_safety_sweep,
            deterministic_only: options.deterministic_only,
        }
    }
}

#[tauri::command]
pub async fn openmed_sidecar_ping(
    app: AppHandle,
    state: State<'_, OpenMedSidecarState>,
) -> Result<SidecarPingResult, SidecarCommandError> {
    exchange(&app, &state, "ping", Value::Null).await
}

#[tauri::command]
pub async fn openmed_sidecar_deidentify(
    app: AppHandle,
    state: State<'_, OpenMedSidecarState>,
    request: SidecarDeidentifyRequest,
) -> Result<SidecarDeidentifyResult, SidecarCommandError> {
    let payload = json!({
        "text": request.text,
        "options": ProtocolOptions::from(request.options),
    });
    exchange(&app, &state, "deidentify", payload).await
}

#[tauri::command]
pub async fn openmed_sidecar_shutdown(
    app: AppHandle,
    state: State<'_, OpenMedSidecarState>,
) -> Result<SidecarShutdownResult, SidecarCommandError> {
    let result = exchange(&app, &state, "shutdown", Value::Null).await;
    state.process.lock().await.take();
    result
}

async fn exchange<T: DeserializeOwned>(
    app: &AppHandle,
    state: &State<'_, OpenMedSidecarState>,
    operation: &'static str,
    payload: Value,
) -> Result<T, SidecarCommandError> {
    let request_id = state
        .next_request_id
        .fetch_add(1, Ordering::Relaxed)
        .to_string();
    let request = protocol_request(&request_id, operation, payload);
    let mut process_guard = state.process.lock().await;
    if process_guard.is_none() {
        *process_guard = Some(spawn_sidecar(app)?);
    }

    let result = exchange_with_process(
        process_guard
            .as_mut()
            .expect("sidecar process was initialized above"),
        &request_id,
        request,
    )
    .await;
    if result.as_ref().is_err_and(should_restart_after) {
        if let Some(process) = process_guard.take() {
            let _ = process.child.kill();
        }
    }
    result
}

fn should_restart_after(error: &SidecarCommandError) -> bool {
    matches!(
        error.code.as_str(),
        "SIDECAR_IO" | "SIDECAR_PROTOCOL" | "SIDECAR_TERMINATED"
    )
}

fn protocol_request(request_id: &str, operation: &str, payload: Value) -> Value {
    let mut request = json!({"id": request_id, "operation": operation});
    if let (Some(request_object), Some(payload_object)) =
        (request.as_object_mut(), payload.as_object())
    {
        request_object.extend(payload_object.clone());
    }
    request
}

fn spawn_sidecar(app: &AppHandle) -> Result<SidecarProcess, SidecarCommandError> {
    let command = app
        .shell()
        .sidecar(SIDECAR_BINARY)
        .map_err(|_| {
            SidecarCommandError::new(
                "SIDECAR_SPAWN_FAILED",
                "The OpenMed sidecar binary is not configured.",
            )
        })?
        .env("OPENMED_OFFLINE", "1");
    let (events, child) = command.spawn().map_err(|_| {
        SidecarCommandError::new(
            "SIDECAR_SPAWN_FAILED",
            "The OpenMed sidecar process could not be started.",
        )
    })?;
    Ok(SidecarProcess { child, events })
}

async fn exchange_with_process<T: DeserializeOwned>(
    process: &mut SidecarProcess,
    request_id: &str,
    request: Value,
) -> Result<T, SidecarCommandError> {
    let mut encoded = serde_json::to_vec(&request).map_err(|_| {
        SidecarCommandError::new(
            "SIDECAR_PROTOCOL",
            "The sidecar request could not be encoded.",
        )
    })?;
    encoded.push(b'\n');
    process.child.write(&encoded).map_err(|_| {
        SidecarCommandError::new("SIDECAR_IO", "The sidecar request could not be written.")
    })?;

    while let Some(event) = process.events.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                return decode_response(request_id, &line);
            }
            CommandEvent::Stderr(_) => {
                // The sidecar owns structured operational logging. Never copy
                // stderr into a frontend error or application log.
            }
            CommandEvent::Error(_) => {
                return Err(SidecarCommandError::new(
                    "SIDECAR_IO",
                    "The OpenMed sidecar stream failed.",
                ));
            }
            CommandEvent::Terminated(_) => return Err(SidecarCommandError::terminated()),
            _ => {}
        }
    }
    Err(SidecarCommandError::terminated())
}

fn decode_response<T: DeserializeOwned>(
    request_id: &str,
    line: &[u8],
) -> Result<T, SidecarCommandError> {
    let response: ProtocolResponse = serde_json::from_slice(line).map_err(|_| {
        SidecarCommandError::new(
            "SIDECAR_PROTOCOL",
            "The sidecar returned an invalid response.",
        )
    })?;
    if response.id != request_id {
        return Err(SidecarCommandError::new(
            "SIDECAR_PROTOCOL",
            "The sidecar response did not match the request.",
        ));
    }
    if !response.ok {
        let error = response.error.unwrap_or(ProtocolError {
            code: "PROCESSING_FAILED".to_owned(),
            message: "The sidecar request failed.".to_owned(),
        });
        return Err(SidecarCommandError::new(error.code, error.message));
    }
    serde_json::from_value(response.result.unwrap_or(Value::Null)).map_err(|_| {
        SidecarCommandError::new(
            "SIDECAR_PROTOCOL",
            "The sidecar result has an invalid shape.",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_successful_ping() {
        let result: SidecarPingResult = decode_response(
            "7",
            br#"{"id":"7","ok":true,"result":{"offline":true,"protocol_version":1}}"#,
        )
        .expect("valid response");
        assert!(result.offline);
        assert_eq!(result.protocol_version, 1);
    }

    #[test]
    fn protocol_errors_are_clean_and_structured() {
        let error = decode_response::<SidecarPingResult>(
            "7",
            br#"{"id":"7","ok":false,"error":{"code":"PROCESSING_FAILED","message":"De-identification failed; verify the local model bundle."}}"#,
        )
        .expect_err("error response");
        assert_eq!(error.code, "PROCESSING_FAILED");
        assert!(!error.message.contains("Synthetic patient"));
    }
}
