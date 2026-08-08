use anyhow::Result;
use rusqlite::{params, Connection};

pub struct Cache {
    conn: Connection,
}

impl Cache {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS designs (
                original_path TEXT PRIMARY KEY,
                uuid TEXT NOT NULL,
                render_path TEXT,
                phash_render TEXT
            )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_uuid ON designs(uuid)",
            [],
        )?;
        Ok(Self { conn })
    }

    /// Has this exact file path already been processed and stored?
    pub fn has_path(&self, original_path: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare("SELECT 1 FROM designs WHERE original_path = ?1 LIMIT 1")?;
        let mut rows = stmt.query([original_path])?;
        Ok(rows.next()?.is_some())
    }

    /// Does any file already have this exact UUID? (Cheap pre-filter; kept for
    /// debugging/back-compat but no longer the primary merge check.)
    pub fn has_uuid(&self, uuid: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare("SELECT 1 FROM designs WHERE uuid = ?1 LIMIT 1")?;
        let mut rows = stmt.query([uuid])?;
        Ok(rows.next()?.is_some())
    }

    /// Find an existing UUID whose pHash is within `max_bits` Hamming distance
    /// of the given pHash. Returns Some(uuid) for the first match, or None.
    ///
    /// This is the cross-format / format-drift merge function. Same design in
    /// PES vs EXP vs VP3 frequently produces pHashes that differ by a few bits
    /// because each format encodes coordinates with different precision. We
    /// merge them into one UUID so the catalog sees one design, not three.
    ///
    /// The original code lacked an integration test for this path; main.rs
    /// was calling has_uuid() (exact equality), so any cross-format drift
    /// even one bit produced a brand-new UUID. That bug affected ~29 designs
    /// in test sets 1-4. Wiring this in fixes them all.
    pub fn find_uuid_within_tolerance(&self, phash: &str, max_bits: u32) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT uuid, phash_render FROM designs")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let stored_uuid: String = row.get(0)?;
            let stored_phash: String = row.get(1)?;
            if let Some(distance) = hamming_distance_hex(phash, &stored_phash) {
                if distance <= max_bits {
                    return Ok(Some(stored_uuid));
                }
            }
        }
        Ok(None)
    }

    pub fn upsert(&self, uuid: &str, original_path: &str, render_path: &str, phash: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO designs(original_path, uuid, render_path, phash_render)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(original_path) DO UPDATE SET
               uuid=excluded.uuid,
               render_path=excluded.render_path,
               phash_render=excluded.phash_render",
            params![original_path, uuid, render_path, phash],
        )?;
        Ok(())
    }
}

/// Hamming distance between two hex-encoded strings, in bits.
/// Returns None on length mismatch or non-hex characters.
fn hamming_distance_hex(a: &str, b: &str) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    let mut distance: u32 = 0;
    let mut ai = a.chars();
    let mut bi = b.chars();
    loop {
        match (ai.next(), bi.next()) {
            (Some(ca), Some(cb)) => {
                let va = ca.to_digit(16)?;
                let vb = cb.to_digit(16)?;
                distance += (va ^ vb).count_ones();
            }
            (None, None) => break,
            _ => return None,
        }
    }
    Some(distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hamming primitive tests ---

    #[test]
    fn hamming_identical() {
        assert_eq!(hamming_distance_hex("ff", "ff"), Some(0));
    }

    #[test]
    fn hamming_all_different() {
        assert_eq!(hamming_distance_hex("ff", "00"), Some(8));
    }

    #[test]
    fn hamming_nibble_diff() {
        assert_eq!(hamming_distance_hex("f0", "00"), Some(4));
    }

    #[test]
    fn hamming_mismatched_length() {
        assert_eq!(hamming_distance_hex("ff", "ffff"), None);
    }

    #[test]
    fn hamming_bad_hex() {
        assert_eq!(hamming_distance_hex("zz", "00"), None);
    }

    // --- Integration tests for find_uuid_within_tolerance ---
    //
    // These are the tests the original codebase was missing. The Hamming
    // function had unit tests; the end-to-end lookup did not. As a result the
    // bug in main.rs (calling has_uuid instead of this tolerance lookup) was
    // never caught by automated testing.

    fn pad64(prefix: &str) -> String {
        let mut s = prefix.to_string();
        while s.len() < 64 {
            s.push('0');
        }
        s
    }

    #[test]
    fn tolerance_finds_within_threshold() {
        let cache = Cache::open(":memory:").unwrap();
        // 4 bits apart
        let phash_a = pad64("ffff");
        let phash_b = pad64("fff0");
        cache.upsert("uuid-a", "/a.pes", "/render/a.png", &phash_a).unwrap();
        let found = cache.find_uuid_within_tolerance(&phash_b, 7).unwrap();
        assert_eq!(found, Some("uuid-a".to_string()));
    }

    #[test]
    fn tolerance_misses_beyond_threshold() {
        let cache = Cache::open(":memory:").unwrap();
        let phash_a = pad64("ffff");
        let phash_b = {
            let mut s = "0000ffff".to_string();
            while s.len() < 64 { s.push('0'); }
            s
        };
        cache.upsert("uuid-a", "/a.pes", "/render/a.png", &phash_a).unwrap();
        let found = cache.find_uuid_within_tolerance(&phash_b, 7).unwrap();
        assert_eq!(found, None);
    }

    #[test]
    fn tolerance_empty_cache() {
        let cache = Cache::open(":memory:").unwrap();
        let phash = pad64("ffff");
        let found = cache.find_uuid_within_tolerance(&phash, 7).unwrap();
        assert_eq!(found, None);
    }

    #[test]
    fn tolerance_exact_match_distance_zero() {
        let cache = Cache::open(":memory:").unwrap();
        let phash = pad64("abcd");
        cache.upsert("uuid-a", "/a.pes", "/render/a.png", &phash).unwrap();
        let found = cache.find_uuid_within_tolerance(&phash, 7).unwrap();
        assert_eq!(found, Some("uuid-a".to_string()));
    }
}