//! Versioned artifact manifest for a packaged OpenMed Tauri sidecar.
//!
//! The trusted Rust host verifies this manifest before it starts the sidecar
//! or sends it any text. The manifest pins the sidecar executable and every
//! file in the local model directory by size and SHA-256, and states the
//! platform, protocol version and commands the sidecar supports. A missing,
//! changed, added or incompatible artifact refuses with a value-free reason
//! code. Errors never contain paths, digests or file names.
//!
//! The file inventory uses the same `path`, `role`, `size_bytes` and `sha256`
//! fields as the portable model-bundle manifest, and the manifest is stored in
//! sorted-key compact JSON so its bytes have one canonical form. The two
//! formats are expected to converge once the portable bundle lands.

use std::{
    collections::BTreeSet,
    fmt, fs,
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema identifier stored in every sidecar manifest.
pub const MANIFEST_SCHEMA: &str = "openmed.tauri.sidecar";
/// Manifest schema version this host reads.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Conventional manifest file name next to the bundled sidecar.
pub const MANIFEST_FILENAME: &str = "openmed-sidecar.json";
/// Sidecar JSON-lines protocol version this host speaks.
pub const SIDECAR_PROTOCOL_VERSION: u32 = 1;
/// Commands the host calls; a manifest must declare all of them.
pub const REQUIRED_CAPABILITIES: [&str; 3] = ["deidentify", "ping", "shutdown"];
/// Stem of the bundled sidecar executable (Tauri `externalBin`).
pub const SIDECAR_BINARY_STEM: &str = "openmed-sidecar";

const MAX_MANIFEST_BYTES: usize = 1 << 20;
const MAX_MODEL_FILES: usize = 4_096;
const MAX_TREE_ENTRIES: usize = 8_192;
const MAX_TREE_DEPTH: usize = 16;
const MAX_PATH_CHARS: usize = 512;
const MAX_MODEL_ID_CHARS: usize = 256;
const MAX_ARTIFACT_BYTES: u64 = 1 << 40;
const HASH_CHUNK_BYTES: usize = 1 << 20;
const PLATFORMS: [&str; 3] = ["linux", "macos", "windows"];
const ARCHITECTURES: [&str; 3] = ["aarch64", "x86", "x86_64"];
const FILE_ROLES: [&str; 13] = [
    "calibration",
    "config",
    "decoder",
    "evidence",
    "graph",
    "labels",
    "license",
    "metadata",
    "policy",
    "provenance",
    "segmenter",
    "tokenizer",
    "weights",
];

/// Category of an artifact verification failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactErrorKind {
    /// The manifest is missing, oversized, malformed or not canonical.
    ManifestInvalid,
    /// The sidecar executable, model directory or a model file is missing.
    ArtifactMissing,
    /// An artifact differs from the manifest, or an unlisted file is present.
    ArtifactMismatch,
    /// The bundle targets another platform, protocol or command set.
    ArtifactIncompatible,
}

/// A value-free artifact verification failure.
///
/// `reason` is a stable snake_case code. Neither field ever carries a path,
/// file name, digest, model identifier or other value from the bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SidecarArtifactError {
    /// Failure category; selects the command error code.
    pub kind: ArtifactErrorKind,
    /// Stable reason code such as `model_file_digest_mismatch`.
    pub reason: &'static str,
}

impl SidecarArtifactError {
    const fn new(kind: ArtifactErrorKind, reason: &'static str) -> Self {
        Self { kind, reason }
    }

    const fn invalid(reason: &'static str) -> Self {
        Self::new(ArtifactErrorKind::ManifestInvalid, reason)
    }

    const fn missing(reason: &'static str) -> Self {
        Self::new(ArtifactErrorKind::ArtifactMissing, reason)
    }

    const fn mismatch(reason: &'static str) -> Self {
        Self::new(ArtifactErrorKind::ArtifactMismatch, reason)
    }

    const fn incompatible(reason: &'static str) -> Self {
        Self::new(ArtifactErrorKind::ArtifactIncompatible, reason)
    }

    /// Stable error code returned to the renderer.
    pub const fn code(&self) -> &'static str {
        match self.kind {
            ArtifactErrorKind::ManifestInvalid => "SIDECAR_MANIFEST_INVALID",
            ArtifactErrorKind::ArtifactMissing => "SIDECAR_ARTIFACT_MISSING",
            ArtifactErrorKind::ArtifactMismatch => "SIDECAR_ARTIFACT_MISMATCH",
            ArtifactErrorKind::ArtifactIncompatible => "SIDECAR_ARTIFACT_INCOMPATIBLE",
        }
    }

    /// Fixed, value-free message for the error code.
    pub const fn message(&self) -> &'static str {
        match self.kind {
            ArtifactErrorKind::ManifestInvalid => {
                "The OpenMed sidecar manifest is missing or invalid."
            }
            ArtifactErrorKind::ArtifactMissing => "A required OpenMed sidecar artifact is missing.",
            ArtifactErrorKind::ArtifactMismatch => {
                "An OpenMed sidecar artifact does not match its manifest."
            }
            ArtifactErrorKind::ArtifactIncompatible => {
                "The OpenMed sidecar bundle is not compatible with this host."
            }
        }
    }
}

impl fmt::Display for SidecarArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({})", self.code(), self.reason)
    }
}

impl std::error::Error for SidecarArtifactError {}

/// Operating system and CPU architecture a sidecar executable was built for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SidecarTarget {
    /// `linux`, `macos` or `windows`, as reported by `std::env::consts::OS`.
    pub platform: String,
    /// `aarch64`, `x86` or `x86_64`, as reported by `std::env::consts::ARCH`.
    pub arch: String,
}

impl SidecarTarget {
    /// Return the target of the running host process.
    pub fn host() -> Self {
        Self {
            platform: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
        }
    }
}

/// Parsed sidecar manifest. Fields are declared in sorted-key order so the
/// compact serde encoding is the canonical byte form.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SidecarManifest {
    /// Sorted, unique commands the sidecar implements.
    pub capabilities: Vec<String>,
    /// Model identity and complete file inventory.
    pub model: SidecarModelEntry,
    /// Sidecar JSON-lines protocol version.
    pub protocol_version: u32,
    /// Always [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// Always [`MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The pinned sidecar executable.
    pub sidecar: SidecarBinaryEntry,
}

/// Pinned sidecar executable.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SidecarBinaryEntry {
    /// CPU architecture the executable was built for.
    pub arch: String,
    /// Operating system the executable was built for.
    pub platform: String,
    /// Lowercase hexadecimal SHA-256 of the executable bytes.
    pub sha256: String,
    /// Executable size in bytes.
    pub size_bytes: u64,
}

/// Model identity and complete file inventory.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SidecarModelEntry {
    /// Files sorted by `path`; every file in the model directory is listed.
    pub files: Vec<SidecarFileEntry>,
    /// Public model identifier, for evidence only.
    pub id: String,
}

/// One pinned model file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SidecarFileEntry {
    /// Relative `/`-separated path inside the model directory.
    pub path: String,
    /// Portable-bundle file role such as `weights` or `tokenizer`.
    pub role: String,
    /// Lowercase hexadecimal SHA-256 of the file bytes.
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

impl SidecarManifest {
    /// Parse and structurally validate canonical manifest bytes.
    ///
    /// This checks the schema, canonical encoding, digests, sizes, paths and
    /// roles. It does not touch the file system or compare the bundle with
    /// the host; [`SidecarBundle::verify`] does both.
    pub fn parse(bytes: &[u8]) -> Result<Self, SidecarArtifactError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(SidecarArtifactError::invalid("manifest_too_large"));
        }
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|_| SidecarArtifactError::invalid("manifest_malformed"))?;
        let schema = value.get("schema").and_then(serde_json::Value::as_str);
        let schema_version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64);
        if schema != Some(MANIFEST_SCHEMA)
            || schema_version != Some(u64::from(MANIFEST_SCHEMA_VERSION))
        {
            return Err(SidecarArtifactError::invalid("manifest_schema_unsupported"));
        }
        let manifest: Self = serde_json::from_value(value)
            .map_err(|_| SidecarArtifactError::invalid("manifest_malformed"))?;
        if manifest.to_canonical_bytes()? != bytes {
            return Err(SidecarArtifactError::invalid("manifest_not_canonical"));
        }
        manifest.validate_structure()?;
        Ok(manifest)
    }

    /// Encode the manifest in its canonical sorted-key compact form.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, SidecarArtifactError> {
        serde_json::to_vec(self).map_err(|_| SidecarArtifactError::invalid("manifest_malformed"))
    }

    fn validate_structure(&self) -> Result<(), SidecarArtifactError> {
        let malformed = SidecarArtifactError::invalid("manifest_malformed");
        if self.capabilities.is_empty()
            || self.capabilities.len() > 32
            || !strictly_sorted(self.capabilities.iter().map(String::as_str))
            || !self.capabilities.iter().all(|value| valid_token(value))
        {
            return Err(malformed);
        }
        let sidecar = &self.sidecar;
        if !PLATFORMS.contains(&sidecar.platform.as_str())
            || !ARCHITECTURES.contains(&sidecar.arch.as_str())
            || !valid_sha256(&sidecar.sha256)
            || sidecar.size_bytes == 0
            || sidecar.size_bytes > MAX_ARTIFACT_BYTES
        {
            return Err(malformed);
        }
        let model = &self.model;
        if model.id.is_empty()
            || model.id.chars().count() > MAX_MODEL_ID_CHARS
            || model.id.chars().any(char::is_control)
            || model.files.is_empty()
            || model.files.len() > MAX_MODEL_FILES
        {
            return Err(malformed);
        }
        let mut folded = BTreeSet::new();
        for file in &model.files {
            if !safe_relative_path(&file.path) {
                return Err(SidecarArtifactError::invalid("path_unsafe"));
            }
            if !FILE_ROLES.contains(&file.role.as_str())
                || !valid_sha256(&file.sha256)
                || file.size_bytes > MAX_ARTIFACT_BYTES
            {
                return Err(malformed);
            }
            if !folded.insert(file.path.to_lowercase()) {
                return Err(SidecarArtifactError::invalid("duplicate_path"));
            }
        }
        if !strictly_sorted(model.files.iter().map(|file| file.path.as_str())) {
            return Err(malformed);
        }
        Ok(())
    }

    fn check_compatibility(&self, host: &SidecarTarget) -> Result<(), SidecarArtifactError> {
        if self.protocol_version != SIDECAR_PROTOCOL_VERSION {
            return Err(SidecarArtifactError::incompatible("protocol_unsupported"));
        }
        if !REQUIRED_CAPABILITIES
            .iter()
            .all(|required| self.capabilities.iter().any(|value| value == required))
        {
            return Err(SidecarArtifactError::incompatible("capability_missing"));
        }
        if self.sidecar.platform != host.platform || self.sidecar.arch != host.arch {
            return Err(SidecarArtifactError::incompatible("platform_mismatch"));
        }
        Ok(())
    }
}

/// Packaged sidecar, model directory and the manifest that binds them.
///
/// Build it only from trusted Rust setup code. Prefer embedding the manifest
/// in the application binary (`include_bytes!`) or pinning its digest with
/// [`SidecarBundle::with_pinned_manifest_sha256`], so the signed host binary
/// decides which sidecar and model it accepts.
#[derive(Clone)]
pub struct SidecarBundle {
    manifest: Vec<u8>,
    pinned_manifest_sha256: Option<String>,
    sidecar_path: PathBuf,
    model_root: PathBuf,
    target: SidecarTarget,
}

impl fmt::Debug for SidecarBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SidecarBundle")
            .field("manifest_sha256", &sha256_hex(&self.manifest))
            .field("pinned", &self.pinned_manifest_sha256.is_some())
            .finish_non_exhaustive()
    }
}

impl SidecarBundle {
    /// Bind manifest bytes to a sidecar executable and a model directory.
    pub fn new(
        manifest: impl Into<Vec<u8>>,
        sidecar_path: impl Into<PathBuf>,
        model_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            manifest: manifest.into(),
            pinned_manifest_sha256: None,
            sidecar_path: sidecar_path.into(),
            model_root: model_root.into(),
            target: SidecarTarget::host(),
        }
    }

    /// Read a manifest file with a size bound and bind it like [`Self::new`].
    pub fn from_manifest_file(
        manifest_path: &Path,
        sidecar_path: impl Into<PathBuf>,
        model_root: impl Into<PathBuf>,
    ) -> Result<Self, SidecarArtifactError> {
        let file = fs::File::open(manifest_path)
            .map_err(|_| SidecarArtifactError::invalid("manifest_unreadable"))?;
        let mut manifest = Vec::new();
        file.take(MAX_MANIFEST_BYTES as u64 + 1)
            .read_to_end(&mut manifest)
            .map_err(|_| SidecarArtifactError::invalid("manifest_unreadable"))?;
        if manifest.len() > MAX_MANIFEST_BYTES {
            return Err(SidecarArtifactError::invalid("manifest_too_large"));
        }
        Ok(Self::new(manifest, sidecar_path, model_root))
    }

    /// Path where Tauri places an `externalBin` named `openmed-sidecar`:
    /// next to the application executable, with `.exe` on Windows.
    pub fn bundled_sidecar_path() -> Result<PathBuf, SidecarArtifactError> {
        let executable = tauri::utils::platform::current_exe()
            .map_err(|_| SidecarArtifactError::missing("sidecar_missing"))?;
        let directory = executable
            .parent()
            .ok_or(SidecarArtifactError::missing("sidecar_missing"))?;
        let mut name = SIDECAR_BINARY_STEM.to_owned();
        if cfg!(windows) {
            name.push_str(".exe");
        }
        Ok(directory.join(name))
    }

    /// Refuse the bundle unless the manifest bytes have this SHA-256.
    pub fn with_pinned_manifest_sha256(mut self, digest: impl Into<String>) -> Self {
        self.pinned_manifest_sha256 = Some(digest.into());
        self
    }

    /// Lowercase hexadecimal SHA-256 of the manifest bytes.
    pub fn manifest_sha256(&self) -> String {
        sha256_hex(&self.manifest)
    }

    #[cfg(test)]
    fn with_target(mut self, target: SidecarTarget) -> Self {
        self.target = target;
        self
    }

    /// Verify the manifest, the host target and every pinned artifact.
    ///
    /// Checks run in a fixed order and stop at the first failure: manifest
    /// size, pinned digest, schema, canonical form and structure; protocol,
    /// capabilities and platform; executable presence and size; model
    /// directory presence, unlisted files, file presence and sizes; then the
    /// executable and model file digests. Nothing is started or downloaded.
    pub fn verify(&self) -> Result<VerifiedSidecarBundle, SidecarArtifactError> {
        if self.manifest.len() > MAX_MANIFEST_BYTES {
            return Err(SidecarArtifactError::invalid("manifest_too_large"));
        }
        let manifest_sha256 = sha256_hex(&self.manifest);
        if let Some(pinned) = &self.pinned_manifest_sha256 {
            if !pinned.eq_ignore_ascii_case(&manifest_sha256) {
                return Err(SidecarArtifactError::invalid("manifest_digest_mismatch"));
            }
        }
        let manifest = SidecarManifest::parse(&self.manifest)?;
        manifest.check_compatibility(&self.target)?;

        let sidecar_metadata = fs::metadata(&self.sidecar_path)
            .map_err(|_| SidecarArtifactError::missing("sidecar_missing"))?;
        if !sidecar_metadata.is_file() {
            return Err(SidecarArtifactError::missing("sidecar_missing"));
        }
        if sidecar_metadata.len() != manifest.sidecar.size_bytes {
            return Err(SidecarArtifactError::mismatch("sidecar_size_mismatch"));
        }

        let root_metadata = fs::metadata(&self.model_root)
            .map_err(|_| SidecarArtifactError::missing("model_root_missing"))?;
        if !root_metadata.is_dir() {
            return Err(SidecarArtifactError::missing("model_root_missing"));
        }
        let listed: BTreeSet<&str> = manifest
            .model
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect();
        for relative in list_tree(&self.model_root)? {
            if !listed.contains(relative.as_str()) {
                return Err(SidecarArtifactError::mismatch("model_file_unlisted"));
            }
        }
        for file in &manifest.model.files {
            let metadata = fs::metadata(join_relative(&self.model_root, &file.path))
                .map_err(|_| SidecarArtifactError::missing("model_file_missing"))?;
            if !metadata.is_file() {
                return Err(SidecarArtifactError::missing("model_file_missing"));
            }
            if metadata.len() != file.size_bytes {
                return Err(SidecarArtifactError::mismatch("model_file_size_mismatch"));
            }
        }

        if file_sha256(&self.sidecar_path)? != manifest.sidecar.sha256 {
            return Err(SidecarArtifactError::mismatch("sidecar_digest_mismatch"));
        }
        let mut bytes_verified = manifest.sidecar.size_bytes;
        for file in &manifest.model.files {
            if file_sha256(&join_relative(&self.model_root, &file.path))? != file.sha256 {
                return Err(SidecarArtifactError::mismatch("model_file_digest_mismatch"));
            }
            bytes_verified = bytes_verified.saturating_add(file.size_bytes);
        }

        Ok(VerifiedSidecarBundle {
            manifest_sha256,
            capabilities: manifest.capabilities,
            files_verified: manifest.model.files.len(),
            bytes_verified,
            sidecar_path: self.sidecar_path.clone(),
            model_root: self.model_root.clone(),
        })
    }
}

/// Result of a successful [`SidecarBundle::verify`].
///
/// Public fields hold only digests and counts; the verified paths stay
/// private so the value can be logged or reported as evidence.
#[derive(Clone)]
pub struct VerifiedSidecarBundle {
    /// SHA-256 of the manifest bytes (content identity).
    pub manifest_sha256: String,
    /// Commands declared by the manifest.
    pub capabilities: Vec<String>,
    /// Number of model files verified.
    pub files_verified: usize,
    /// Executable plus model bytes hashed.
    pub bytes_verified: u64,
    sidecar_path: PathBuf,
    model_root: PathBuf,
}

impl fmt::Debug for VerifiedSidecarBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedSidecarBundle")
            .field("manifest_sha256", &self.manifest_sha256)
            .field("capabilities", &self.capabilities)
            .field("files_verified", &self.files_verified)
            .field("bytes_verified", &self.bytes_verified)
            .finish_non_exhaustive()
    }
}

impl VerifiedSidecarBundle {
    pub(crate) fn sidecar_path(&self) -> &Path {
        &self.sidecar_path
    }

    pub(crate) fn model_root(&self) -> &Path {
        &self.model_root
    }
}

/// Build canonical manifest bytes for a sidecar executable and model directory.
///
/// Every file under `model_root` is listed with a role inferred from its
/// name. Use this at packaging time, after the sidecar has been built and the
/// model has been placed in the application resources.
pub fn generate_manifest(
    sidecar_path: &Path,
    model_root: &Path,
    model_id: &str,
    target: &SidecarTarget,
) -> Result<Vec<u8>, SidecarArtifactError> {
    let sidecar_metadata =
        fs::metadata(sidecar_path).map_err(|_| SidecarArtifactError::missing("sidecar_missing"))?;
    if !sidecar_metadata.is_file() {
        return Err(SidecarArtifactError::missing("sidecar_missing"));
    }
    if !fs::metadata(model_root).is_ok_and(|metadata| metadata.is_dir()) {
        return Err(SidecarArtifactError::missing("model_root_missing"));
    }
    let mut files = Vec::new();
    for relative in list_tree(model_root)? {
        if !safe_relative_path(&relative) {
            return Err(SidecarArtifactError::invalid("path_unsafe"));
        }
        let path = join_relative(model_root, &relative);
        let metadata =
            fs::metadata(&path).map_err(|_| SidecarArtifactError::missing("model_file_missing"))?;
        if !metadata.is_file() {
            return Err(SidecarArtifactError::missing("model_file_missing"));
        }
        files.push(SidecarFileEntry {
            role: infer_role(&relative).to_owned(),
            sha256: file_sha256(&path)?,
            size_bytes: metadata.len(),
            path: relative,
        });
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let manifest = SidecarManifest {
        capabilities: REQUIRED_CAPABILITIES
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        model: SidecarModelEntry {
            files,
            id: model_id.to_owned(),
        },
        protocol_version: SIDECAR_PROTOCOL_VERSION,
        schema: MANIFEST_SCHEMA.to_owned(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        sidecar: SidecarBinaryEntry {
            arch: target.arch.clone(),
            platform: target.platform.clone(),
            sha256: file_sha256(sidecar_path)?,
            size_bytes: sidecar_metadata.len(),
        },
    };
    let bytes = manifest.to_canonical_bytes()?;
    SidecarManifest::parse(&bytes)?;
    Ok(bytes)
}

/// Lowercase hexadecimal SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(DIGITS[usize::from(byte >> 4)] as char);
        encoded.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn file_sha256(path: &Path) -> Result<String, SidecarArtifactError> {
    let unreadable = SidecarArtifactError::missing("artifact_unreadable");
    let mut file = fs::File::open(path).map_err(|_| unreadable)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(|_| unreadable)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// List every non-directory entry under `root` as a `/`-separated relative
/// path. Directory symlinks are reported as entries, not followed, so they
/// can never satisfy the inventory. Names that are not UTF-8 fail closed.
fn list_tree(root: &Path) -> Result<Vec<String>, SidecarArtifactError> {
    let too_large = SidecarArtifactError::mismatch("model_tree_too_large");
    let unreadable = SidecarArtifactError::missing("artifact_unreadable");
    let mut entries = Vec::new();
    let mut pending = vec![(root.to_path_buf(), String::new(), 0_usize)];
    let mut visited = 0_usize;
    while let Some((directory, prefix, depth)) = pending.pop() {
        if depth > MAX_TREE_DEPTH {
            return Err(too_large);
        }
        for entry in fs::read_dir(&directory).map_err(|_| unreadable)? {
            visited += 1;
            if visited > MAX_TREE_ENTRIES {
                return Err(too_large);
            }
            let entry = entry.map_err(|_| unreadable)?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| SidecarArtifactError::mismatch("model_file_unlisted"))?;
            let relative = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            if entry.file_type().map_err(|_| unreadable)?.is_dir() {
                pending.push((entry.path(), relative, depth + 1));
            } else {
                entries.push(relative);
            }
        }
    }
    entries.sort();
    Ok(entries)
}

fn join_relative(root: &Path, relative: &str) -> PathBuf {
    relative
        .split('/')
        .fold(root.to_path_buf(), |path, segment| path.join(segment))
}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_PATH_CHARS
        && !value.chars().any(|character| {
            character.is_control()
                || matches!(character, '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
        })
        && value
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn strictly_sorted<'a>(values: impl Iterator<Item = &'a str>) -> bool {
    let mut previous: Option<&str> = None;
    for value in values {
        if previous.is_some_and(|before| before >= value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn infer_role(relative: &str) -> &'static str {
    let name = relative
        .rsplit('/')
        .next()
        .unwrap_or(relative)
        .to_ascii_lowercase();
    const WEIGHT_SUFFIXES: [&str; 12] = [
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".mlmodel",
        ".msgpack",
        ".npz",
        ".onnx",
        ".onnx_data",
        ".pt",
        ".pth",
        ".safetensors",
    ];
    if WEIGHT_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
        "weights"
    } else if name.starts_with("tokenizer")
        || name.starts_with("vocab")
        || name.starts_with("merges")
        || name.starts_with("special_tokens_map")
        || name.starts_with("added_tokens")
        || name.ends_with(".model")
    {
        "tokenizer"
    } else if name.starts_with("label") || name.starts_with("id2label") {
        "labels"
    } else if name.starts_with("license") || name.starts_with("notice") {
        "license"
    } else if name.ends_with("config.json") {
        "config"
    } else {
        "metadata"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "openmed-sidecar-manifest-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join("model/onnx")).expect("scratch directory");
        fs::write(path.join("sidecar"), b"synthetic sidecar executable").expect("sidecar");
        fs::write(path.join("model/config.json"), b"{}").expect("config");
        fs::write(path.join("model/tokenizer.json"), b"{\"synthetic\":true}").expect("tokenizer");
        fs::write(path.join("model/onnx/model.onnx"), b"synthetic weights").expect("weights");
        Scratch(path)
    }

    fn bundle(scratch: &Scratch) -> SidecarBundle {
        let manifest = generate_manifest(
            &scratch.0.join("sidecar"),
            &scratch.0.join("model"),
            "synthetic/model",
            &SidecarTarget::host(),
        )
        .expect("manifest");
        SidecarBundle::new(manifest, scratch.0.join("sidecar"), scratch.0.join("model"))
    }

    #[test]
    fn generated_manifest_is_canonical_sorted_and_verifies() {
        let scratch = scratch();
        let bundle = bundle(&scratch);
        let manifest = SidecarManifest::parse(&bundle.manifest).expect("parse");
        let paths: Vec<&str> = manifest
            .model
            .files
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert_eq!(paths, ["config.json", "onnx/model.onnx", "tokenizer.json"]);
        let roles: Vec<&str> = manifest
            .model
            .files
            .iter()
            .map(|f| f.role.as_str())
            .collect();
        assert_eq!(roles, ["config", "weights", "tokenizer"]);
        assert!(bundle
            .manifest
            .starts_with(b"{\"capabilities\":[\"deidentify\",\"ping\",\"shutdown\"]"));
        let verified = bundle.verify().expect("verified");
        assert_eq!(verified.files_verified, 3);
        assert_eq!(verified.manifest_sha256, sha256_hex(&bundle.manifest));
    }

    #[test]
    fn rejects_non_canonical_unknown_and_unsafe_manifests() {
        let scratch = scratch();
        let bundle = bundle(&scratch);
        let value: serde_json::Value = serde_json::from_slice(&bundle.manifest).unwrap();
        let pretty = serde_json::to_vec_pretty(&value).unwrap();
        assert_eq!(
            SidecarManifest::parse(&pretty).unwrap_err().reason,
            "manifest_not_canonical"
        );

        let mut extra = value.clone();
        extra["unexpected"] = serde_json::json!(true);
        assert_eq!(
            SidecarManifest::parse(&serde_json::to_vec(&extra).unwrap())
                .unwrap_err()
                .reason,
            "manifest_malformed"
        );

        let mut manifest = SidecarManifest::parse(&bundle.manifest).unwrap();
        manifest.model.files[0].path = "../escape.json".to_owned();
        assert_eq!(
            SidecarManifest::parse(&manifest.to_canonical_bytes().unwrap())
                .unwrap_err()
                .reason,
            "path_unsafe"
        );

        let mut manifest = SidecarManifest::parse(&bundle.manifest).unwrap();
        manifest.model.files.swap(0, 2);
        assert_eq!(
            SidecarManifest::parse(&manifest.to_canonical_bytes().unwrap())
                .unwrap_err()
                .reason,
            "manifest_malformed"
        );

        let mut manifest = SidecarManifest::parse(&bundle.manifest).unwrap();
        manifest.schema_version = 2;
        assert_eq!(
            SidecarManifest::parse(&manifest.to_canonical_bytes().unwrap())
                .unwrap_err()
                .reason,
            "manifest_schema_unsupported"
        );
        assert_eq!(
            SidecarManifest::parse(b"not json").unwrap_err().reason,
            "manifest_malformed"
        );
    }

    #[test]
    fn foreign_target_is_incompatible_before_any_file_is_read() {
        let scratch = scratch();
        let bundle = bundle(&scratch);
        fs::remove_file(scratch.0.join("sidecar")).unwrap();
        let foreign = SidecarTarget {
            platform: if cfg!(windows) { "linux" } else { "windows" }.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
        };
        let error = bundle.with_target(foreign).verify().unwrap_err();
        assert_eq!(error.code(), "SIDECAR_ARTIFACT_INCOMPATIBLE");
        assert_eq!(error.reason, "platform_mismatch");
    }

    #[test]
    fn errors_render_only_codes_and_reasons() {
        let scratch = scratch();
        let bundle = bundle(&scratch);
        fs::write(scratch.0.join("model/config.json"), b"[]").unwrap();
        let error = bundle.verify().unwrap_err();
        assert_eq!(error.reason, "model_file_digest_mismatch");
        let rendered = format!("{error} {error:?} {bundle:?}");
        assert!(!rendered.contains("config.json"));
        assert!(!rendered.contains(&scratch.0.display().to_string()));
    }
}
