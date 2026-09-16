//! Tauri v2 command wrapper for the persistent OpenMed sidecar process.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tokio::sync::{mpsc::Receiver, Mutex};
use tokio::time::timeout;

const SIDECAR_BINARY: &str = "openmed-sidecar";
const SIDECAR_MODEL_ENV: &str = "OPENMED_SIDECAR_MODEL";
const PROTOCOL_VERSION: u32 = 1;
const MAX_TEXT_CHARS: usize = 1_000_000;
const MAX_TEXT_BYTES: usize = 4_000_000;
const MAX_DEIDENTIFIED_TEXT_CHARS: usize = 8_000_000;
const MAX_DEIDENTIFIED_TEXT_BYTES: usize = 32_000_000;
const MAX_RESPONSE_LINE_BYTES: usize = 64_000_000;
const MAX_RESPONSE_SPANS: usize = 65_536;
const MAX_MODEL_NAME_CHARS: usize = 4_096;
const MAX_DOC_ID_CHARS: usize = 256;
const MAX_POLICY_CHARS: usize = 128;
const MAX_SHORT_FIELD_CHARS: usize = 128;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

pub struct OpenMedSidecarState {
    process: Mutex<Option<SidecarProcess>>,
    next_request_id: AtomicU64,
    model_name: Option<String>,
    request_timeout: Duration,
}

impl Default for OpenMedSidecarState {
    fn default() -> Self {
        Self {
            process: Mutex::new(None),
            next_request_id: AtomicU64::new(0),
            model_name: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

impl OpenMedSidecarState {
    /// Pin a cached model identifier or local path from trusted Rust setup.
    pub fn with_model(model_name: impl Into<String>) -> Result<Self, SidecarCommandError> {
        let model_name = model_name.into();
        if model_name.is_empty() || model_name.chars().count() > MAX_MODEL_NAME_CHARS {
            return Err(SidecarCommandError::new(
                "SIDECAR_CONFIGURATION",
                "The OpenMed sidecar model configuration is invalid.",
            ));
        }
        Ok(Self {
            model_name: Some(model_name),
            ..Self::default()
        })
    }
}

struct SidecarProcess {
    child: CommandChild,
    events: Receiver<CommandEvent>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct SidecarDeidentifyRequest {
    pub text: String,
    #[serde(default)]
    pub options: SidecarDeidentifyOptions,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct SidecarDeidentifyOptions {
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
#[serde(deny_unknown_fields)]
pub struct SidecarDeidentifyResult {
    #[serde(rename(serialize = "deidentifiedText", deserialize = "deidentified_text"))]
    pub deidentified_text: String,
    pub spans: Vec<OpenMedSpan>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct SidecarPingResult {
    pub offline: bool,
    #[serde(rename(serialize = "protocolVersion", deserialize = "protocol_version"))]
    pub protocol_version: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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

    fn protocol() -> Self {
        Self::new(
            "SIDECAR_PROTOCOL",
            "The OpenMed sidecar returned an invalid response.",
        )
    }

    fn timeout() -> Self {
        Self::new(
            "SIDECAR_TIMEOUT",
            "The OpenMed sidecar did not respond before the deadline.",
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolResponse {
    id: String,
    ok: bool,
    result: Option<Value>,
    error: Option<ProtocolError>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolError {
    code: String,
    #[serde(rename = "message")]
    _message: String,
}

#[derive(Debug, Serialize)]
struct ProtocolOptions {
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
    exchange(&app, &state, "ping", Value::Null, validate_ping_result).await
}

#[tauri::command]
pub async fn openmed_sidecar_deidentify(
    app: AppHandle,
    state: State<'_, OpenMedSidecarState>,
    request: SidecarDeidentifyRequest,
) -> Result<SidecarDeidentifyResult, SidecarCommandError> {
    let source_chars = validate_deidentify_request(&request)?;
    let payload = json!({
        "text": request.text,
        "options": ProtocolOptions::from(request.options),
    });
    exchange(&app, &state, "deidentify", payload, move |result| {
        validate_deidentify_result(result, source_chars)
    })
    .await
}

#[tauri::command]
pub async fn openmed_sidecar_shutdown(
    app: AppHandle,
    state: State<'_, OpenMedSidecarState>,
) -> Result<SidecarShutdownResult, SidecarCommandError> {
    let result = exchange(
        &app,
        &state,
        "shutdown",
        Value::Null,
        validate_shutdown_result,
    )
    .await;
    if result.is_ok() {
        state.process.lock().await.take();
    }
    result
}

async fn exchange<T, F>(
    app: &AppHandle,
    state: &State<'_, OpenMedSidecarState>,
    operation: &'static str,
    payload: Value,
    validate: F,
) -> Result<T, SidecarCommandError>
where
    T: DeserializeOwned,
    F: FnOnce(&T) -> Result<(), SidecarCommandError>,
{
    let request_id = state
        .next_request_id
        .fetch_add(1, Ordering::Relaxed)
        .to_string();
    let request = protocol_request(&request_id, operation, payload);
    let mut process_guard = state.process.try_lock().map_err(|_| {
        SidecarCommandError::new(
            "SIDECAR_BUSY",
            "The OpenMed sidecar is already processing a request.",
        )
    })?;
    if process_guard.is_none() {
        *process_guard = Some(spawn_sidecar(app, state.model_name.as_deref())?);
    }

    let result = match timeout(
        state.request_timeout,
        exchange_with_process(
            process_guard
                .as_mut()
                .expect("sidecar process was initialized above"),
            &request_id,
            request,
        ),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(SidecarCommandError::timeout()),
    }
    .and_then(|value| {
        validate(&value)?;
        Ok(value)
    });
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
        "SIDECAR_IO" | "SIDECAR_PROTOCOL" | "SIDECAR_TERMINATED" | "SIDECAR_TIMEOUT"
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

fn spawn_sidecar(
    app: &AppHandle,
    model_name: Option<&str>,
) -> Result<SidecarProcess, SidecarCommandError> {
    let mut command = app
        .shell()
        .sidecar(SIDECAR_BINARY)
        .map_err(|_| {
            SidecarCommandError::new(
                "SIDECAR_SPAWN_FAILED",
                "The OpenMed sidecar binary is not configured.",
            )
        })?
        .env("OPENMED_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1");
    if let Some(model_name) = model_name {
        command = command.env(SIDECAR_MODEL_ENV, model_name);
    }
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
    if line.len() > MAX_RESPONSE_LINE_BYTES {
        return Err(SidecarCommandError::protocol());
    }
    let response: ProtocolResponse =
        serde_json::from_slice(line).map_err(|_| SidecarCommandError::protocol())?;
    if response.id != request_id {
        return Err(SidecarCommandError::protocol());
    }
    if !response.ok {
        if response.result.is_some() {
            return Err(SidecarCommandError::protocol());
        }
        return match response.error.map(|error| error.code) {
            Some(code) if code == "INVALID_REQUEST" => Err(SidecarCommandError::new(
                "INVALID_REQUEST",
                "The OpenMed sidecar rejected the request.",
            )),
            Some(code) if code == "PROCESSING_FAILED" => Err(SidecarCommandError::new(
                "PROCESSING_FAILED",
                "OpenMed de-identification failed; verify the local model bundle.",
            )),
            _ => Err(SidecarCommandError::protocol()),
        };
    }
    if response.error.is_some() {
        return Err(SidecarCommandError::protocol());
    }
    serde_json::from_value(response.result.ok_or_else(SidecarCommandError::protocol)?)
        .map_err(|_| SidecarCommandError::protocol())
}

fn validate_deidentify_request(
    request: &SidecarDeidentifyRequest,
) -> Result<usize, SidecarCommandError> {
    let text_chars = request.text.chars().count();
    if text_chars == 0 || text_chars > MAX_TEXT_CHARS || request.text.len() > MAX_TEXT_BYTES {
        return Err(invalid_request());
    }
    let options = &request.options;
    if !valid_optional_string(&options.policy, MAX_POLICY_CHARS)
        || !valid_optional_string(&options.method, 32)
        || !valid_optional_string(&options.lang, 16)
        || !valid_optional_string(&options.doc_id, MAX_DOC_ID_CHARS)
        || options
            .confidence_threshold
            .is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        || options.method.as_deref().is_some_and(|method| {
            !matches!(
                method,
                "mask" | "remove" | "replace" | "hash" | "format_preserve"
            )
        })
    {
        return Err(invalid_request());
    }
    Ok(text_chars)
}

fn validate_ping_result(result: &SidecarPingResult) -> Result<(), SidecarCommandError> {
    if result.offline && result.protocol_version == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(SidecarCommandError::protocol())
    }
}

fn validate_shutdown_result(result: &SidecarShutdownResult) -> Result<(), SidecarCommandError> {
    if result.shutdown {
        Ok(())
    } else {
        Err(SidecarCommandError::protocol())
    }
}

fn validate_deidentify_result(
    result: &SidecarDeidentifyResult,
    source_chars: usize,
) -> Result<(), SidecarCommandError> {
    if result.deidentified_text.chars().count() > MAX_DEIDENTIFIED_TEXT_CHARS
        || result.deidentified_text.len() > MAX_DEIDENTIFIED_TEXT_BYTES
        || result.spans.len() > MAX_RESPONSE_SPANS
        || result.spans.len() > source_chars
    {
        return Err(SidecarCommandError::protocol());
    }

    let mut spans: Vec<&OpenMedSpan> = result.spans.iter().collect();
    spans.sort_by_key(|span| (span.start, span.end));
    let mut previous_end = 0;
    for span in spans {
        if !validate_span(span, source_chars) || span.start < previous_end {
            return Err(SidecarCommandError::protocol());
        }
        previous_end = span.end;
    }
    Ok(())
}

fn validate_span(span: &OpenMedSpan, source_chars: usize) -> bool {
    span.schema_version == PROTOCOL_VERSION
        && !span.doc_id.is_empty()
        && span.doc_id.chars().count() <= MAX_DOC_ID_CHARS
        && span.start < span.end
        && span.end <= source_chars
        && valid_text_hash(&span.text_hash)
        && valid_token(&span.entity_type, false)
        && valid_token(&span.canonical_label, true)
        && span.policy_label.as_deref().is_none_or(|label| {
            matches!(
                label,
                "DIRECT_IDENTIFIER" | "QUASI_IDENTIFIER" | "CLINICAL_CONCEPT"
            )
        })
        && span.regulatory_tags.len() <= 64
        && span
            .regulatory_tags
            .iter()
            .all(|tag| !tag.is_empty() && tag.chars().count() <= MAX_SHORT_FIELD_CHARS)
        && span
            .score
            .is_none_or(|score| score.is_finite() && (0.0..=1.0).contains(&score))
        && valid_optional_string(&span.detector, MAX_SHORT_FIELD_CHARS)
        && span.evidence.is_object()
        && matches!(
            span.action.as_str(),
            "keep" | "redact" | "replace" | "mask" | "hash" | "format_preserve"
        )
        && valid_optional_string_or_empty(&span.replacement, 4_096)
        && valid_optional_string(&span.reversible_id, 512)
        && valid_optional_string(&span.section, 256)
        && span.metadata.is_object()
}

fn valid_optional_string(value: &Option<String>, maximum: usize) -> bool {
    value
        .as_deref()
        .is_none_or(|value| !value.is_empty() && value.chars().count() <= maximum)
}

fn valid_optional_string_or_empty(value: &Option<String>, maximum: usize) -> bool {
    value
        .as_deref()
        .is_none_or(|value| value.chars().count() <= maximum)
}

fn valid_text_hash(value: &str) -> bool {
    value.strip_prefix("hmac-sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_token(value: &str, uppercase: bool) -> bool {
    !value.is_empty()
        && value.len() <= MAX_SHORT_FIELD_CHARS
        && value.bytes().all(|byte| {
            byte.is_ascii_digit()
                || if uppercase {
                    byte.is_ascii_uppercase() || byte == b'_'
                } else {
                    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'.' | b':' | b'-')
                }
        })
}

fn invalid_request() -> SidecarCommandError {
    SidecarCommandError::new("INVALID_REQUEST", "The OpenMed sidecar request is invalid.")
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
            br#"{"id":"7","ok":false,"error":{"code":"PROCESSING_FAILED","message":"Synthetic patient Rowan Hale failed."}}"#,
        )
        .expect_err("error response");
        assert_eq!(error.code, "PROCESSING_FAILED");
        assert_eq!(
            error.message,
            "OpenMed de-identification failed; verify the local model bundle."
        );
        assert!(!error.message.contains("Rowan Hale"));
    }

    #[test]
    fn rejects_ambiguous_or_unknown_protocol_envelopes() {
        for response in [
            br#"{"id":"7","ok":true,"result":{"offline":true,"protocol_version":1},"error":{"code":"PROCESSING_FAILED"}}"#.as_slice(),
            br#"{"id":"7","ok":false,"result":{},"error":{"code":"PROCESSING_FAILED"}}"#.as_slice(),
            br#"{"id":"7","ok":false,"error":{"code":"UNTRUSTED","message":"Rowan Hale"}}"#.as_slice(),
        ] {
            let error = decode_response::<SidecarPingResult>("7", response)
                .expect_err("invalid protocol envelope");
            assert_eq!(error.code, "SIDECAR_PROTOCOL");
            assert!(!error.message.contains("Rowan Hale"));
        }
    }

    #[test]
    fn validates_request_bounds_and_host_pinned_model_configuration() {
        let valid = SidecarDeidentifyRequest {
            text: "🧬Synthetic note".to_owned(),
            options: SidecarDeidentifyOptions::default(),
        };
        assert_eq!(
            validate_deidentify_request(&valid).expect("valid request"),
            15
        );

        let invalid = SidecarDeidentifyRequest {
            text: "x".repeat(MAX_TEXT_CHARS + 1),
            options: SidecarDeidentifyOptions::default(),
        };
        assert_eq!(
            validate_deidentify_request(&invalid)
                .expect_err("oversized request")
                .code,
            "INVALID_REQUEST"
        );
        assert_eq!(
            OpenMedSidecarState::with_model("")
                .err()
                .expect("empty model")
                .code,
            "SIDECAR_CONFIGURATION"
        );
    }

    #[test]
    fn validates_canonical_spans_and_rejects_overlap() {
        let mut result = SidecarDeidentifyResult {
            deidentified_text: "🧬[NAME]".to_owned(),
            spans: vec![valid_span(1, 2)],
        };
        validate_deidentify_result(&result, 2).expect("code-point offsets are valid");

        result.spans.push(valid_span(1, 2));
        assert_eq!(
            validate_deidentify_result(&result, 2)
                .expect_err("overlap must fail")
                .code,
            "SIDECAR_PROTOCOL"
        );
    }

    #[test]
    fn timeout_requires_process_restart() {
        assert!(should_restart_after(&SidecarCommandError::timeout()));
    }

    fn valid_span(start: usize, end: usize) -> OpenMedSpan {
        OpenMedSpan {
            schema_version: PROTOCOL_VERSION,
            doc_id: "synthetic-rust-test".to_owned(),
            start,
            end,
            text_hash: format!("hmac-sha256:{}", "a".repeat(64)),
            entity_type: "person".to_owned(),
            canonical_label: "NAME".to_owned(),
            policy_label: Some("DIRECT_IDENTIFIER".to_owned()),
            regulatory_tags: vec![],
            score: Some(0.9),
            detector: Some("synthetic-test".to_owned()),
            evidence: json!({}),
            action: "mask".to_owned(),
            replacement: Some("[NAME]".to_owned()),
            reversible_id: None,
            section: None,
            metadata: json!({}),
        }
    }
}
