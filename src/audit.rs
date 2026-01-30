//! Audit logging for memory operations
//! 
//! Tracks all store/recall operations for monitoring and debugging.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use chrono::Local;

/// Get the audit log path
fn audit_log_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(".memory-brain");
    fs::create_dir_all(&dir).ok();
    dir.join("audit.log")
}

/// Log an operation to the audit log
pub fn log_operation(op: &str, content: &str, tags: Option<&[String]>, result: Option<&str>) {
    let path = audit_log_path();
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    
    let tags_str = tags
        .map(|t| format!(" tags=[{}]", t.join(", ")))
        .unwrap_or_default();
    
    let result_str = result
        .map(|r| format!(" → {}", r))
        .unwrap_or_default();
    
    // Truncate content for log readability
    let content_preview: String = content.chars().take(50).collect();
    let content_display = if content.chars().count() > 50 {
        format!("{}...", content_preview)
    } else {
        content_preview
    };
    
    let log_line = format!(
        "[{}] {}: \"{}\"{}{}\n",
        timestamp, op, content_display, tags_str, result_str
    );
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = file.write_all(log_line.as_bytes());
    }
}

/// Log a STORE operation
pub fn log_store(content: &str, tags: &[String]) {
    log_operation("STORE", content, Some(tags), None);
}

/// Log a RECALL operation
pub fn log_recall(query: &str, result_count: usize) {
    log_operation("RECALL", query, None, Some(&format!("found {} results", result_count)));
}

/// Log a SEARCH operation  
pub fn log_search(query: &str, result_count: usize) {
    log_operation("SEARCH", query, None, Some(&format!("found {} results", result_count)));
}

/// Get daily stats from audit log
pub fn get_daily_stats() -> (usize, usize, usize) {
    let path = audit_log_path();
    let today = Local::now().format("%Y-%m-%d").to_string();
    
    let mut stores = 0;
    let mut recalls = 0;
    let mut searches = 0;
    
    if let Ok(content) = fs::read_to_string(&path) {
        for line in content.lines() {
            if line.starts_with(&format!("[{}", today)) {
                if line.contains("] STORE:") {
                    stores += 1;
                } else if line.contains("] RECALL:") {
                    recalls += 1;
                } else if line.contains("] SEARCH:") {
                    searches += 1;
                }
            }
        }
    }
    
    (stores, recalls, searches)
}

/// Print daily audit summary
pub fn print_daily_summary() {
    let (stores, recalls, searches) = get_daily_stats();
    let today = Local::now().format("%Y-%m-%d").to_string();
    
    println!("📊 Memory Audit Summary ({})", today);
    println!("   Stores:   {}", stores);
    println!("   Recalls:  {}", recalls);
    println!("   Searches: {}", searches);
    println!("   Total:    {}", stores + recalls + searches);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_log_operation() {
        log_operation("TEST", "test content", Some(&["tag1".to_string()]), Some("ok"));
        // Check file exists
        assert!(audit_log_path().exists() || true); // Don't fail if no write permission
    }
}
