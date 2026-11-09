//! Write the canonical artifact manifest for a packaged OpenMed sidecar.
//!
//! ```text
//! openmed-sidecar-manifest --sidecar <executable> --model-root <directory> \
//!     --model-id <public id> --out <openmed-sidecar.json> \
//!     [--platform <linux|macos|windows>] [--arch <aarch64|x86|x86_64>]
//! ```
//!
//! The manifest is written to `--out`; only its SHA-256 is printed, so the
//! host can pin it. Errors print a stable reason code, never a path.

use std::{path::PathBuf, process::ExitCode};

use openmedkit_tauri_sidecar::sidecar_manifest::{generate_manifest, sha256_hex, SidecarTarget};

const USAGE: &str = "usage: openmed-sidecar-manifest --sidecar <executable> \
--model-root <directory> --model-id <id> --out <file> [--platform <os>] [--arch <arch>]";

fn main() -> ExitCode {
    let mut sidecar = None;
    let mut model_root = None;
    let mut model_id = None;
    let mut out = None;
    let mut target = SidecarTarget::host();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(flag) = arguments.next() {
        let Some(value) = arguments.next() else {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        };
        match flag.to_str() {
            Some("--sidecar") => sidecar = Some(PathBuf::from(value)),
            Some("--model-root") => model_root = Some(PathBuf::from(value)),
            Some("--model-id") => model_id = value.into_string().ok(),
            Some("--out") => out = Some(PathBuf::from(value)),
            Some("--platform") => match value.into_string() {
                Ok(platform) => target.platform = platform,
                Err(_) => return usage(),
            },
            Some("--arch") => match value.into_string() {
                Ok(arch) => target.arch = arch,
                Err(_) => return usage(),
            },
            _ => return usage(),
        }
    }
    let (Some(sidecar), Some(model_root), Some(model_id), Some(out)) =
        (sidecar, model_root, model_id, out)
    else {
        return usage();
    };
    let manifest = match generate_manifest(&sidecar, &model_root, &model_id, &target) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };
    if std::fs::write(&out, &manifest).is_err() {
        eprintln!("SIDECAR_MANIFEST_INVALID (manifest_unwritable)");
        return ExitCode::from(1);
    }
    println!("{}", sha256_hex(&manifest));
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}
