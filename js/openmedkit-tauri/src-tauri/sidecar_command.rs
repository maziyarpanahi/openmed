//! Tauri v2 command wrapper for the persistent OpenMed sidecar process.
//!
//! The trusted host verifies the packaged sidecar and model against their
//! manifest before every process start, binds each process to a fresh
//! generation id, and accepts only responses carrying that generation. A
//! crash, timeout, stale or malformed response, or host teardown discards the
//! process and returns a typed, value-free error.

use std::{
    collections::hash_map::RandomState,
    future::Future,
    hash::{BuildHasher, Hasher},
    io,
    pin::pin,
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex as StdMutex,
    },
    task::Poll,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tauri::State;
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{watch, Mutex},
    task::JoinHandle,
    time::timeout,
};

use crate::sidecar_manifest::{
    hex, SidecarArtifactError, SidecarBundle, VerifiedSidecarBundle, SIDECAR_PROTOCOL_VERSION,
};

const SIDECAR_MODEL_ENV: &str = "OPENMED_SIDECAR_MODEL";
const PROTOCOL_VERSION: u32 = SIDECAR_PROTOCOL_VERSION;
const MAX_TEXT_CHARS: usize = 1_000_000;
const MAX_TEXT_BYTES: usize = 4_000_000;
const MAX_DEIDENTIFIED_TEXT_CHARS: usize = 8_000_000;
const MAX_DEIDENTIFIED_TEXT_BYTES: usize = 32_000_000;
const MAX_RESPONSE_LINE_BYTES: usize = 64_000_000;
const MAX_STDERR_LINE_BYTES: usize = 4_096;
const MAX_RESPONSE_SPANS: usize = 65_536;
const MAX_DOC_ID_CHARS: usize = 256;
const MAX_POLICY_CHARS: usize = 128;
const MAX_SHORT_FIELD_CHARS: usize = 128;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const EXIT_WAIT: Duration = Duration::from_secs(5);
const STDERR_SETTLE_WAIT: Duration = Duration::from_secs(2);
/// Interpreter variables that could make the verified executable load code
/// from outside its own bundle.
const BLOCKED_CHILD_ENV: [&str; 5] = [
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
];

/// Persistent sidecar state managed by the Tauri application.
pub struct OpenMedSidecarState {
    bundle: Arc<SidecarBundle>,
    process: Mutex<Option<SidecarProcess>>,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
    request_timeout: Duration,
    startup_timeout: Duration,
    closed: AtomicBool,
    terminate: watch::Sender<bool>,
}

impl OpenMedSidecarState {
    /// Manage a packaged sidecar that must match `bundle` before every start.
    ///
    /// Nothing is verified or started here; the first command, or
    /// [`Self::qualify`], verifies the manifest and starts the process.
    pub fn from_bundle(bundle: SidecarBundle) -> Self {
        Self {
            bundle: Arc::new(bundle),
            process: Mutex::new(None),
            next_request_id: AtomicU64::new(0),
            next_generation: AtomicU64::new(0),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            closed: AtomicBool::new(false),
            terminate: watch::Sender::new(false),
        }
    }

    /// Override the per-request deadline (default two minutes).
    pub fn with_request_timeout(mut self, deadline: Duration) -> Self {
        if !deadline.is_zero() {
            self.request_timeout = deadline;
        }
        self
    }

    /// Override the start-up handshake deadline (default one minute).
    pub fn with_startup_timeout(mut self, deadline: Duration) -> Self {
        if !deadline.is_zero() {
            self.startup_timeout = deadline;
        }
        self
    }

    /// Verify the bundle and start the sidecar now instead of on first use.
    pub async fn qualify(&self) -> Result<SidecarQualification, SidecarCommandError> {
        let mut guard = self.acquire()?;
        self.ensure_process(&mut guard).await?;
        let process = guard.as_ref().ok_or_else(SidecarCommandError::terminated)?;
        Ok(SidecarQualification {
            generation: process.generation.clone(),
            manifest_sha256: process.verified.manifest_sha256.clone(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: process.verified.capabilities.clone(),
            files_verified: process.verified.files_verified,
        })
    }

    /// Check that the sidecar is running offline with the expected protocol.
    pub async fn ping(&self) -> Result<SidecarPingResult, SidecarCommandError> {
        self.call("ping", Value::Null, validate_ping_result).await
    }

    /// De-identify bounded text with the pinned local model.
    pub async fn deidentify(
        &self,
        request: SidecarDeidentifyRequest,
    ) -> Result<SidecarDeidentifyResult, SidecarCommandError> {
        let source_chars = validate_deidentify_request(&request)?;
        let payload = json!({
            "text": request.text,
            "options": ProtocolOptions::from(request.options),
        });
        self.call("deidentify", payload, move |result| {
            validate_deidentify_result(result, source_chars)
        })
        .await
    }

    /// Ask a running sidecar to exit cleanly. A later command starts and
    /// verifies a new generation.
    pub async fn shutdown(&self) -> Result<SidecarShutdownResult, SidecarCommandError> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(SidecarShutdownResult { shutdown: true });
        }
        let mut guard = self.acquire()?;
        if guard.as_ref().is_none_or(|process| process.in_flight) {
            if let Some(process) = guard.take() {
                process.kill().await;
            }
            return Ok(SidecarShutdownResult { shutdown: true });
        }
        let result = self
            .exchange_locked(
                &mut guard,
                "shutdown",
                Value::Null,
                validate_shutdown_result,
            )
            .await;
        if let Some(process) = guard.take() {
            process.wait_or_kill().await;
        }
        result
    }

    /// Stop the sidecar for good, interrupting any request in progress.
    ///
    /// Call this from the application's exit handler. The interrupted request
    /// and every later command fail with `SIDECAR_NOT_RUNNING`.
    pub async fn terminate(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.terminate.send_replace(true);
        let mut guard = self.process.lock().await;
        if let Some(process) = guard.take() {
            process.kill().await;
        }
    }

    /// Blocking form of [`Self::terminate`] for synchronous Tauri run-event
    /// handlers such as `RunEvent::Exit`.
    pub fn terminate_blocking(&self) {
        tauri::async_runtime::block_on(self.terminate());
    }

    /// Opaque id of the running sidecar generation, if one is running.
    pub async fn generation(&self) -> Option<String> {
        self.process
            .lock()
            .await
            .as_ref()
            .map(|process| process.generation.clone())
    }

    /// Operating-system id of the running sidecar process, if one is running.
    pub async fn process_id(&self) -> Option<u32> {
        self.process
            .lock()
            .await
            .as_ref()
            .and_then(|process| process.child.id())
    }

    fn acquire(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, Option<SidecarProcess>>, SidecarCommandError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(SidecarCommandError::not_running());
        }
        self.process.try_lock().map_err(|_| {
            SidecarCommandError::new(
                "SIDECAR_BUSY",
                "The OpenMed sidecar is already processing a request.",
            )
        })
    }

    async fn call<T, F>(
        &self,
        operation: &'static str,
        payload: Value,
        validate: F,
    ) -> Result<T, SidecarCommandError>
    where
        T: DeserializeOwned,
        F: FnOnce(&T) -> Result<(), SidecarCommandError>,
    {
        let mut guard = self.acquire()?;
        self.ensure_process(&mut guard).await?;
        self.exchange_locked(&mut guard, operation, payload, validate)
            .await
    }

    /// Start a verified generation unless a clean one is already running.
    async fn ensure_process(
        &self,
        guard: &mut Option<SidecarProcess>,
    ) -> Result<(), SidecarCommandError> {
        if guard.as_ref().is_some_and(|process| process.in_flight) {
            // A cancelled caller left a request without reading its response.
            // Discard the generation so no later caller can receive it.
            if let Some(process) = guard.take() {
                process.kill().await;
            }
        }
        if guard.is_none() {
            let mut terminate = self.terminate.subscribe();
            match race(self.start_process(), wait_for_terminate(&mut terminate)).await {
                Race::First(process) => *guard = Some(process?),
                Race::Second(()) => return Err(SidecarCommandError::not_running()),
            }
        }
        Ok(())
    }

    async fn start_process(&self) -> Result<SidecarProcess, SidecarCommandError> {
        let bundle = Arc::clone(&self.bundle);
        let verified = tokio::task::spawn_blocking(move || bundle.verify())
            .await
            .map_err(|_| {
                SidecarCommandError::new(
                    "SIDECAR_CONFIGURATION",
                    "The OpenMed sidecar configuration is invalid.",
                )
                .with_reason("verification_aborted")
            })??;
        let generation = new_generation_id(self.next_generation.fetch_add(1, Ordering::Relaxed));
        let mut process = spawn_verified(verified, generation)?;

        let request_id = self.request_id(&process.generation);
        let request = protocol_request(&request_id, "ping", Value::Null);
        let handshake = match timeout(
            self.startup_timeout,
            process.exchange::<SidecarPingResult>(&request_id, &request),
        )
        .await
        {
            Ok(Ok(result)) if validate_ping_result(&result).is_ok() => Ok(()),
            Ok(Ok(_)) => Err(SidecarCommandError::protocol("handshake_invalid")),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(SidecarCommandError::timeout().with_reason("startup_timeout")),
        };
        match handshake {
            Ok(()) => Ok(process),
            Err(error) => {
                process.kill().await;
                Err(error)
            }
        }
    }

    async fn exchange_locked<T, F>(
        &self,
        guard: &mut Option<SidecarProcess>,
        operation: &'static str,
        payload: Value,
        validate: F,
    ) -> Result<T, SidecarCommandError>
    where
        T: DeserializeOwned,
        F: FnOnce(&T) -> Result<(), SidecarCommandError>,
    {
        let Some(process) = guard.as_mut() else {
            return Err(SidecarCommandError::terminated());
        };
        let request_id = self.request_id(&process.generation);
        let request = protocol_request(&request_id, operation, payload);
        let mut terminate = self.terminate.subscribe();
        let outcome = race(
            timeout(
                self.request_timeout,
                process.exchange::<T>(&request_id, &request),
            ),
            wait_for_terminate(&mut terminate),
        )
        .await;
        let result = match outcome {
            Race::First(Ok(result)) => result,
            Race::First(Err(_)) => Err(SidecarCommandError::timeout()),
            Race::Second(()) => Err(SidecarCommandError::not_running()),
        }
        .and_then(|value| {
            validate(&value)?;
            Ok(value)
        });
        if result.as_ref().is_err_and(should_restart_after) {
            if let Some(process) = guard.take() {
                process.kill().await;
            }
        }
        result
    }

    fn request_id(&self, generation: &str) -> String {
        let sequence = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        format!("{generation}:{sequence}")
    }
}

/// Value-free evidence that a packaged sidecar passed qualification.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarQualification {
    /// Opaque id bound to the running process; responses must carry it.
    pub generation: String,
    /// SHA-256 of the verified manifest bytes.
    pub manifest_sha256: String,
    /// Protocol version confirmed by the start-up handshake.
    pub protocol_version: u32,
    /// Commands declared by the manifest.
    pub capabilities: Vec<String>,
    /// Number of model files verified before start.
    pub files_verified: usize,
}

struct SidecarProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_task: Option<JoinHandle<()>>,
    terminal_reason: Arc<StdMutex<Option<&'static str>>>,
    generation: String,
    verified: VerifiedSidecarBundle,
    in_flight: bool,
}

impl SidecarProcess {
    async fn exchange<T: DeserializeOwned>(
        &mut self,
        request_id: &str,
        request: &Value,
    ) -> Result<T, SidecarCommandError> {
        let mut encoded = serde_json::to_vec(request).map_err(|_| {
            SidecarCommandError::new(
                "SIDECAR_PROTOCOL",
                "The sidecar request could not be encoded.",
            )
        })?;
        encoded.push(b'\n');
        self.in_flight = true;
        let written = async {
            self.stdin.write_all(&encoded).await?;
            self.stdin.flush().await
        }
        .await;
        if written.is_err() {
            return Err(self.terminated_error().await);
        }
        let line = match read_bounded_line(&mut self.stdout, MAX_RESPONSE_LINE_BYTES).await {
            Ok(BoundedLine::Line(line)) => line,
            Ok(BoundedLine::Eof) => return Err(self.terminated_error().await),
            Ok(BoundedLine::TooLong) => {
                return Err(SidecarCommandError::protocol("response_too_large"))
            }
            Err(_) => {
                return Err(SidecarCommandError::new(
                    "SIDECAR_IO",
                    "The OpenMed sidecar stream failed.",
                )
                .with_reason("stdout_failed"))
            }
        };
        self.in_flight = false;
        decode_response(&self.generation, request_id, &line)
    }

    /// Build the terminated error once stderr has had a moment to report an
    /// allow-listed terminal event. Stderr content is never copied.
    async fn terminated_error(&mut self) -> SidecarCommandError {
        let _ = timeout(STDERR_SETTLE_WAIT, self.child.wait()).await;
        if let Some(task) = self.stderr_task.take() {
            let _ = timeout(STDERR_SETTLE_WAIT, task).await;
        }
        let reason = self
            .terminal_reason
            .lock()
            .ok()
            .and_then(|reason| *reason)
            .unwrap_or("process_exited");
        SidecarCommandError::terminated().with_reason(reason)
    }

    async fn wait_or_kill(mut self) {
        drop(self.stdin);
        if timeout(EXIT_WAIT, self.child.wait()).await.is_err() {
            let _ = self.child.start_kill();
            let _ = timeout(EXIT_WAIT, self.child.wait()).await;
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }

    async fn kill(mut self) {
        let _ = self.child.start_kill();
        let _ = timeout(EXIT_WAIT, self.child.wait()).await;
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }
}

fn spawn_verified(
    verified: VerifiedSidecarBundle,
    generation: String,
) -> Result<SidecarProcess, SidecarCommandError> {
    let spawn_failed = || {
        SidecarCommandError::new(
            "SIDECAR_SPAWN_FAILED",
            "The OpenMed sidecar process could not be started.",
        )
    };
    let mut command = Command::new(verified.sidecar_path());
    for name in BLOCKED_CHILD_ENV {
        command.env_remove(name);
    }
    command
        .env("OPENMED_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env(SIDECAR_MODEL_ENV, verified.model_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command.spawn().map_err(|_| spawn_failed())?;
    let (Some(stdin), Some(stdout), Some(stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        let _ = child.start_kill();
        return Err(spawn_failed());
    };
    let terminal_reason = Arc::new(StdMutex::new(None));
    let stderr_task = tokio::spawn(drain_stderr(stderr, Arc::clone(&terminal_reason)));
    Ok(SidecarProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
        stderr_task: Some(stderr_task),
        terminal_reason,
        generation,
        verified,
        in_flight: false,
    })
}

/// Keep the stderr pipe drained and record only allow-listed terminal events.
async fn drain_stderr(stderr: ChildStderr, terminal_reason: Arc<StdMutex<Option<&'static str>>>) {
    let mut reader = BufReader::new(stderr);
    loop {
        match read_bounded_line(&mut reader, MAX_STDERR_LINE_BYTES).await {
            Ok(BoundedLine::Line(line)) => {
                if let Some(reason) = terminal_stderr_reason(&line) {
                    if let Ok(mut slot) = terminal_reason.lock() {
                        *slot = Some(reason);
                    }
                }
            }
            Ok(BoundedLine::TooLong) => {
                if discard_rest_of_line(&mut reader).await.is_err() {
                    return;
                }
            }
            Ok(BoundedLine::Eof) | Err(_) => return,
        }
    }
}

fn terminal_stderr_reason(line: &[u8]) -> Option<&'static str> {
    let record: Value = serde_json::from_slice(line).ok()?;
    if record.get("event").and_then(Value::as_str) != Some("sidecar_failed") {
        return None;
    }
    match record.get("error_code").and_then(Value::as_str)? {
        "SIDECAR_START_FAILED" => Some("sidecar_start_failed"),
        "SIDECAR_RUNTIME_FAILED" => Some("sidecar_runtime_failed"),
        _ => None,
    }
}

enum BoundedLine {
    Line(Vec<u8>),
    Eof,
    TooLong,
}

/// Read one `\n`-terminated line without buffering more than `maximum` bytes.
/// A trailing `\r` is removed; a final line without `\n` counts as EOF.
async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    maximum: usize,
) -> io::Result<BoundedLine> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(BoundedLine::Eof);
        }
        let (chunk, complete) = match available.iter().position(|byte| *byte == b'\n') {
            Some(index) => (&available[..index], Some(index + 1)),
            None => (available, None),
        };
        if line.len() + chunk.len() > maximum {
            let consumed = complete.unwrap_or(available.len());
            reader.consume(consumed);
            return Ok(BoundedLine::TooLong);
        }
        line.extend_from_slice(chunk);
        let consumed = complete.unwrap_or(available.len());
        reader.consume(consumed);
        if complete.is_some() {
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(BoundedLine::Line(line));
        }
    }
}

async fn discard_rest_of_line<R: AsyncBufRead + Unpin>(reader: &mut R) -> io::Result<()> {
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(index) = available.iter().position(|byte| *byte == b'\n') {
            reader.consume(index + 1);
            return Ok(());
        }
        let consumed = available.len();
        reader.consume(consumed);
    }
}

enum Race<A, B> {
    First(A),
    Second(B),
}

/// Poll two futures and return whichever finishes first.
async fn race<A: Future, B: Future>(first: A, second: B) -> Race<A::Output, B::Output> {
    let mut first = pin!(first);
    let mut second = pin!(second);
    std::future::poll_fn(move |context| {
        if let Poll::Ready(value) = first.as_mut().poll(context) {
            return Poll::Ready(Race::First(value));
        }
        if let Poll::Ready(value) = second.as_mut().poll(context) {
            return Poll::Ready(Race::Second(value));
        }
        Poll::Pending
    })
    .await
}

async fn wait_for_terminate(receiver: &mut watch::Receiver<bool>) {
    if receiver.wait_for(|terminated| *terminated).await.is_err() {
        std::future::pending::<()>().await;
    }
}

fn new_generation_id(sequence: u64) -> String {
    let mut digest = Sha256::new();
    for salt in [0_u64, 1] {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(sequence ^ salt);
        digest.update(hasher.finish().to_le_bytes());
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    digest.update(now.to_le_bytes());
    digest.update(std::process::id().to_le_bytes());
    digest.update(sequence.to_le_bytes());
    hex(&digest.finalize()[..16])
}

#[tauri::command]
pub async fn openmed_sidecar_ping(
    state: State<'_, OpenMedSidecarState>,
) -> Result<SidecarPingResult, SidecarCommandError> {
    state.ping().await
}

#[tauri::command]
pub async fn openmed_sidecar_deidentify(
    state: State<'_, OpenMedSidecarState>,
    request: SidecarDeidentifyRequest,
) -> Result<SidecarDeidentifyResult, SidecarCommandError> {
    state.deidentify(request).await
}

#[tauri::command]
pub async fn openmed_sidecar_shutdown(
    state: State<'_, OpenMedSidecarState>,
) -> Result<SidecarShutdownResult, SidecarCommandError> {
    state.shutdown().await
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

/// Typed command error. `code` and `message` come from a fixed table and
/// `reason`, when present, is a stable snake_case code; none of them carries
/// source text, stderr output, paths or digests.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarCommandError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl SidecarCommandError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            reason: None,
        }
    }

    fn with_reason(mut self, reason: &'static str) -> Self {
        self.reason = Some(reason.to_owned());
        self
    }

    fn terminated() -> Self {
        Self::new(
            "SIDECAR_TERMINATED",
            "The OpenMed sidecar terminated before responding.",
        )
    }

    fn not_running() -> Self {
        Self::new(
            "SIDECAR_NOT_RUNNING",
            "The OpenMed sidecar has been stopped by the host.",
        )
        .with_reason("host_terminated")
    }

    fn protocol(reason: &'static str) -> Self {
        Self::new(
            "SIDECAR_PROTOCOL",
            "The OpenMed sidecar returned an invalid response.",
        )
        .with_reason(reason)
    }

    fn timeout() -> Self {
        Self::new(
            "SIDECAR_TIMEOUT",
            "The OpenMed sidecar did not respond before the deadline.",
        )
    }
}

impl From<SidecarArtifactError> for SidecarCommandError {
    fn from(error: SidecarArtifactError) -> Self {
        Self::new(error.code(), error.message()).with_reason(error.reason)
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

fn should_restart_after(error: &SidecarCommandError) -> bool {
    matches!(
        error.code.as_str(),
        "SIDECAR_IO"
            | "SIDECAR_NOT_RUNNING"
            | "SIDECAR_PROTOCOL"
            | "SIDECAR_TERMINATED"
            | "SIDECAR_TIMEOUT"
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

fn decode_response<T: DeserializeOwned>(
    generation: &str,
    request_id: &str,
    line: &[u8],
) -> Result<T, SidecarCommandError> {
    if line.len() > MAX_RESPONSE_LINE_BYTES {
        return Err(SidecarCommandError::protocol("response_too_large"));
    }
    let response: ProtocolResponse = serde_json::from_slice(line)
        .map_err(|_| SidecarCommandError::protocol("response_malformed"))?;
    if response.id != request_id {
        let same_generation = response
            .id
            .split_once(':')
            .is_some_and(|(prefix, _)| prefix == generation);
        return Err(SidecarCommandError::protocol(if same_generation {
            "response_id_mismatch"
        } else {
            "stale_generation"
        }));
    }
    if !response.ok {
        if response.result.is_some() {
            return Err(SidecarCommandError::protocol("response_malformed"));
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
            _ => Err(SidecarCommandError::protocol("response_malformed")),
        };
    }
    if response.error.is_some() {
        return Err(SidecarCommandError::protocol("response_malformed"));
    }
    serde_json::from_value(
        response
            .result
            .ok_or_else(|| SidecarCommandError::protocol("response_malformed"))?,
    )
    .map_err(|_| SidecarCommandError::protocol("result_invalid"))
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
        Err(SidecarCommandError::protocol("result_invalid"))
    }
}

fn validate_shutdown_result(result: &SidecarShutdownResult) -> Result<(), SidecarCommandError> {
    if result.shutdown {
        Ok(())
    } else {
        Err(SidecarCommandError::protocol("result_invalid"))
    }
}

fn validate_deidentify_result(
    result: &SidecarDeidentifyResult,
    source_chars: usize,
) -> Result<(), SidecarCommandError> {
    let invalid = || SidecarCommandError::protocol("result_invalid");
    if result.deidentified_text.chars().count() > MAX_DEIDENTIFIED_TEXT_CHARS
        || result.deidentified_text.len() > MAX_DEIDENTIFIED_TEXT_BYTES
        || result.spans.len() > MAX_RESPONSE_SPANS
        || result.spans.len() > source_chars
    {
        return Err(invalid());
    }

    let mut spans: Vec<&OpenMedSpan> = result.spans.iter().collect();
    spans.sort_by_key(|span| (span.start, span.end));
    let mut previous_end = 0;
    for span in spans {
        if !validate_span(span, source_chars) || span.start < previous_end {
            return Err(invalid());
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

    const GENERATION: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn decodes_a_successful_ping() {
        let result: SidecarPingResult = decode_response(
            GENERATION,
            "0123456789abcdef0123456789abcdef:7",
            br#"{"id":"0123456789abcdef0123456789abcdef:7","ok":true,"result":{"offline":true,"protocol_version":1}}"#,
        )
        .expect("valid response");
        assert!(result.offline);
        assert_eq!(result.protocol_version, 1);
    }

    #[test]
    fn protocol_errors_are_clean_and_structured() {
        let error = decode_response::<SidecarPingResult>(
            GENERATION,
            "0123456789abcdef0123456789abcdef:7",
            br#"{"id":"0123456789abcdef0123456789abcdef:7","ok":false,"error":{"code":"PROCESSING_FAILED","message":"Synthetic patient Rowan Hale failed."}}"#,
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
            br#"{"id":"0123456789abcdef0123456789abcdef:7","ok":true,"result":{"offline":true,"protocol_version":1},"error":{"code":"PROCESSING_FAILED"}}"#.as_slice(),
            br#"{"id":"0123456789abcdef0123456789abcdef:7","ok":false,"result":{},"error":{"code":"PROCESSING_FAILED"}}"#.as_slice(),
            br#"{"id":"0123456789abcdef0123456789abcdef:7","ok":false,"error":{"code":"UNTRUSTED","message":"Rowan Hale"}}"#.as_slice(),
        ] {
            let error = decode_response::<SidecarPingResult>(
                GENERATION,
                "0123456789abcdef0123456789abcdef:7",
                response,
            )
            .expect_err("invalid protocol envelope");
            assert_eq!(error.code, "SIDECAR_PROTOCOL");
            assert!(!error.message.contains("Rowan Hale"));
        }
    }

    #[test]
    fn responses_from_another_generation_are_stale() {
        let line = br#"{"id":"ffffffffffffffffffffffffffffffff:7","ok":true,"result":{"offline":true,"protocol_version":1}}"#;
        let error = decode_response::<SidecarPingResult>(
            GENERATION,
            "0123456789abcdef0123456789abcdef:7",
            line,
        )
        .expect_err("stale generation");
        assert_eq!(error.reason.as_deref(), Some("stale_generation"));

        let line = br#"{"id":"0123456789abcdef0123456789abcdef:6","ok":true,"result":{"offline":true,"protocol_version":1}}"#;
        let error = decode_response::<SidecarPingResult>(
            GENERATION,
            "0123456789abcdef0123456789abcdef:7",
            line,
        )
        .expect_err("earlier request");
        assert_eq!(error.reason.as_deref(), Some("response_id_mismatch"));
    }

    #[test]
    fn generation_ids_are_unique_opaque_hex() {
        let first = new_generation_id(0);
        let second = new_generation_id(1);
        assert_ne!(first, second);
        assert_eq!(first.len(), 32);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn stderr_terminal_events_map_only_allow_listed_codes() {
        assert_eq!(
            terminal_stderr_reason(
                br#"{"error_code":"SIDECAR_START_FAILED","event":"sidecar_failed"}"#
            ),
            Some("sidecar_start_failed")
        );
        assert_eq!(
            terminal_stderr_reason(br#"{"error_code":"Rowan Hale","event":"sidecar_failed"}"#),
            None
        );
        assert_eq!(terminal_stderr_reason(b"Traceback: Rowan Hale"), None);
    }

    #[test]
    fn validates_request_bounds() {
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
    fn transport_failures_require_process_restart() {
        assert!(should_restart_after(&SidecarCommandError::timeout()));
        assert!(should_restart_after(&SidecarCommandError::not_running()));
        assert!(should_restart_after(&SidecarCommandError::protocol(
            "stale_generation"
        )));
        assert!(!should_restart_after(&invalid_request()));
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
