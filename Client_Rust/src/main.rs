mod cache;
mod scanner;
mod types;
mod worker;

use anyhow::Result;
use std::path::PathBuf;

use cache::Cache;
use worker::process_file_with_python;

fn main() -> Result<()> {
    // Adjust these two paths once for your machine.
    let python = "../venv/bin/python";
    let script = "../worker_python/process_file.py"; // run from client_rust/ directory

    // Scan roots:
    //   * Pass a folder as the first CLI argument to scan only that folder:
    //         cargo run --release -- "/Users/amans/Desktop/designs"
    //     The `--` is required so cargo hands the path to the program and
    //     not to cargo itself. This is also what the dashboard's
    //     "folder to scan" field sends.
    //   * With NO argument, scan everywhere the user can read: the whole
    //     home directory plus any mounted/external volumes (USB sticks,
    //     external SSDs, etc.).
    let roots: Vec<String> = match std::env::args().nth(1) {
        Some(folder) => vec![folder],
        None => {
            let mut r: Vec<String> = Vec::new();
            if let Ok(home) = std::env::var("HOME") {
                r.push(home);
            }
            if PathBuf::from("/Volumes").is_dir() {
                r.push("/Volumes".to_string());
            }
            if r.is_empty() {
                r.push(".".to_string()); // last-resort fallback
            }
            r
        }
    };

    let cache = Cache::open("diamond_cache.sqlite")?;

    println!("Scanning: {}", roots.join(", "));
    let files = scanner::scan_for_embroidery_files(&roots);
    println!("Found {} candidate files", files.len());

    for f in files {
        // Call python worker
        match process_file_with_python(python, script, &f) {
            Ok(r) => {
                // Skip only if this exact file path is already in the cache.
                // (We do NOT skip on UUID, because the same UUID across multiple
                // file paths is the cross-format-match signal we want to capture.)
                if cache.has_path(&r.original_path)? {
                    println!("ALREADY CACHED: {} -> {}", r.original_path, r.uuid);
                    continue;
                }

                // Check whether this UUID already exists for some other file
                // (i.e. a cross-format match) BEFORE we insert.
                let is_cross_format = cache.has_uuid(&r.uuid)?;

                cache.upsert(
                    &r.uuid,
                    &r.original_path,
                    &r.render_path,
                    &r.phash_render,
                )?;

                let label = if is_cross_format { "MATCH" } else { "NEW" };
                println!("{}: {} -> {}", label, r.original_path, r.uuid);
            }
            Err(e) => {
                eprintln!("Error processing {}: {}", f, e);
            }
        }
    }

    Ok(())
}