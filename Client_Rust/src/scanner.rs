use std::collections::HashSet;
use std::os::unix::fs::MetadataExt;
use walkdir::WalkDir;

// Embroidery extensions we look for (case-insensitive).
const EXTS: &[&str] = &["pes", "dst", "exp", "jef", "vp3", "hus", "vip", "art", "xxx"];

// Directory names we never descend into — OS/system, caches, dev noise.
// This keeps a whole-computer scan fast and skips macOS-protected and
// duplicate trees. Add or remove names here to tune coverage.
const SKIP_DIRS: &[&str] = &[
    "node_modules", ".git", ".svn", ".cache", ".npm", ".cargo",
    "Library", "System", "Applications", "private", "dev",
    ".Trash", "DerivedData", "Caches", "venv", ".venv",
    ".Spotlight-V100", ".fseventsd", ".DocumentRevisions-V100", ".TemporaryItems",
];

fn skip_dir(name: &str) -> bool {
    SKIP_DIRS.iter().any(|s| s.eq_ignore_ascii_case(name))
}

/// Recursively scan one or more roots for embroidery files.
///
/// A physical file reached through more than one path (macOS firmlinks,
/// hardlinks, symlinked mounts) is returned only once, so the downstream
/// worker never renders or caches the same file twice.
pub fn scan_for_embroidery_files(roots: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<(u64, u64)> = HashSet::new();

    for root in roots {
        let walker = WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                // Prune noise directories (and their whole subtree).
                // Files always pass through this filter.
                if e.file_type().is_dir() {
                    if let Some(name) = e.file_name().to_str() {
                        return !skip_dir(name);
                    }
                }
                true
            });

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let matches_ext = path
                .extension()
                .and_then(|x| x.to_str())
                .map(|ext| EXTS.iter().any(|want| want.eq_ignore_ascii_case(ext)))
                .unwrap_or(false);
            if !matches_ext {
                continue;
            }
            // De-dupe by (device, inode).
            if let Ok(md) = entry.metadata() {
                if !seen.insert((md.dev(), md.ino())) {
                    continue;
                }
            }
            out.push(path.to_string_lossy().to_string());
        }
    }
    out
}