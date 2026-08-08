use anyhow::{anyhow, Result};
use std::process::Command;

use crate::types::{WorkerErr, WorkerOk};

pub fn process_file_with_python(python: &str, script_path: &str, file_path: &str) -> Result<WorkerOk> {
    let out = Command::new(python)
        .arg(script_path)
        .arg(file_path)
        .output()?;

    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if stdout.is_empty() {
        return Err(anyhow!("Python worker returned empty output"));
    }

    // Try OK first
    if let Ok(ok) = serde_json::from_str::<WorkerOk>(&stdout) {
        if ok.status == "ok" {
            return Ok(ok);
        }
    }

    // Then error
    let err = serde_json::from_str::<WorkerErr>(&stdout)
        .map_err(|_| anyhow!("Failed to parse worker output: {}", stdout))?;

    Err(anyhow!("Worker error: {}", err.error))
}
