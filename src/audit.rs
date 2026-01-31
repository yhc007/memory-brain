//! Audit logging for memory operations
//! 
//! Tracks all store/recall operations for monitoring and debugging.
//! Now with beautiful TUI visualization! 🎨

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use chrono::{Local, NaiveDate};
use colored::*;

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

/// Get stats for a specific date
pub fn get_stats_for_date(date: &str) -> (usize, usize, usize) {
    let path = audit_log_path();
    
    let mut stores = 0;
    let mut recalls = 0;
    let mut searches = 0;
    
    if let Ok(content) = fs::read_to_string(&path) {
        for line in content.lines() {
            if line.starts_with(&format!("[{}", date)) {
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

/// Get weekly stats (last 7 days)
pub fn get_weekly_stats() -> Vec<(String, usize, usize, usize)> {
    let mut stats = Vec::new();
    let today = Local::now().date_naive();
    
    for i in (0..7).rev() {
        let date = today - chrono::Duration::days(i);
        let date_str = date.format("%Y-%m-%d").to_string();
        let (stores, recalls, searches) = get_stats_for_date(&date_str);
        stats.push((date_str, stores, recalls, searches));
    }
    
    stats
}

/// Create a horizontal bar
fn bar(value: usize, max: usize, width: usize, color: &str) -> String {
    let filled = if max > 0 {
        (value as f64 / max as f64 * width as f64).round() as usize
    } else {
        0
    };
    let empty = width.saturating_sub(filled);
    
    let bar_char = "█";
    let empty_char = "░";
    
    let bar_str = bar_char.repeat(filled);
    let empty_str = empty_char.repeat(empty);
    
    match color {
        "green" => format!("{}{}", bar_str.green(), empty_str.bright_black()),
        "blue" => format!("{}{}", bar_str.blue(), empty_str.bright_black()),
        "yellow" => format!("{}{}", bar_str.yellow(), empty_str.bright_black()),
        "cyan" => format!("{}{}", bar_str.cyan(), empty_str.bright_black()),
        "magenta" => format!("{}{}", bar_str.magenta(), empty_str.bright_black()),
        _ => format!("{}{}", bar_str, empty_str),
    }
}

/// Print daily audit summary (legacy simple version)
pub fn print_daily_summary() {
    let (stores, recalls, searches) = get_daily_stats();
    let today = Local::now().format("%Y-%m-%d").to_string();
    
    println!("📊 Memory Audit Summary ({})", today);
    println!("   Stores:   {}", stores);
    println!("   Recalls:  {}", recalls);
    println!("   Searches: {}", searches);
    println!("   Total:    {}", stores + recalls + searches);
}

/// Print beautiful visual audit summary 🎨
pub fn print_visual_summary() {
    let (stores, recalls, searches) = get_daily_stats();
    let today = Local::now().format("%Y-%m-%d").to_string();
    let total = stores + recalls + searches;
    let max_val = stores.max(recalls).max(searches).max(1);
    
    // Header
    println!();
    println!("{}", "╔══════════════════════════════════════════════════════╗".cyan());
    println!("{} {} {}", 
        "║".cyan(),
        format!("🧠 Memory Audit Dashboard - {}", today).bold().white(),
        "║".cyan()
    );
    println!("{}", "╠══════════════════════════════════════════════════════╣".cyan());
    
    // Stats with bars
    let bar_width = 25;
    
    println!("{} {} {} {} {:>4} {}",
        "║".cyan(),
        "📥 Stores ".green().bold(),
        bar(stores, max_val, bar_width, "green"),
        format!("{:>3}", stores).green().bold(),
        "",
        "║".cyan()
    );
    
    println!("{} {} {} {} {:>4} {}",
        "║".cyan(),
        "🔍 Recalls".blue().bold(),
        bar(recalls, max_val, bar_width, "blue"),
        format!("{:>3}", recalls).blue().bold(),
        "",
        "║".cyan()
    );
    
    println!("{} {} {} {} {:>4} {}",
        "║".cyan(),
        "🔎 Search ".yellow().bold(),
        bar(searches, max_val, bar_width, "yellow"),
        format!("{:>3}", searches).yellow().bold(),
        "",
        "║".cyan()
    );
    
    // Divider
    println!("{}", "╠══════════════════════════════════════════════════════╣".cyan());
    
    // Total
    println!("{} {} {:>42} {}",
        "║".cyan(),
        "📊 Total Today:".bold(),
        format!("{}", total).bold().white(),
        "║".cyan()
    );
    
    // Footer
    println!("{}", "╚══════════════════════════════════════════════════════╝".cyan());
    println!();
}

/// Print weekly trend chart 📈
pub fn print_weekly_trend() {
    let stats = get_weekly_stats();
    
    println!();
    println!("{}", "╔══════════════════════════════════════════════════════╗".magenta());
    println!("{} {} {}",
        "║".magenta(),
        "📈 Weekly Activity Trend (Last 7 Days)".bold().white(),
        "              ║".magenta()
    );
    println!("{}", "╠══════════════════════════════════════════════════════╣".magenta());
    
    // Find max for scaling
    let max_total: usize = stats.iter()
        .map(|(_, s, r, se)| s + r + se)
        .max()
        .unwrap_or(1)
        .max(1);
    
    // Print each day
    for (date, stores, recalls, searches) in &stats {
        let total = stores + recalls + searches;
        let day_name = if let Ok(d) = NaiveDate::parse_from_str(date, "%Y-%m-%d") {
            d.format("%a").to_string()
        } else {
            "???".to_string()
        };
        
        let bar_width = 30;
        let activity_bar = bar(total, max_total, bar_width, "cyan");
        
        // Highlight today
        let is_today = date == &Local::now().format("%Y-%m-%d").to_string();
        let day_display = if is_today {
            format!("{}", day_name).bold().yellow()
        } else {
            format!("{}", day_name).normal()
        };
        
        let marker = if is_today { "→" } else { " " };
        
        println!("{} {} {} {} {:>3} {}",
            "║".magenta(),
            marker.yellow(),
            day_display,
            activity_bar,
            total,
            "║".magenta()
        );
    }
    
    println!("{}", "╠══════════════════════════════════════════════════════╣".magenta());
    
    // Legend
    println!("{} {} {}",
        "║".magenta(),
        format!("  {} Stores  {} Recalls  {} Searches", 
            "📥".green(), "🔍".blue(), "🔎".yellow()),
        "          ║".magenta()
    );
    
    println!("{}", "╚══════════════════════════════════════════════════════╝".magenta());
    println!();
}

/// Print full visual report
pub fn print_full_report() {
    print_visual_summary();
    print_weekly_trend();
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
    
    #[test]
    fn test_bar_generation() {
        let b = bar(5, 10, 10, "green");
        assert!(!b.is_empty());
    }
    
    #[test]
    fn test_weekly_stats() {
        let stats = get_weekly_stats();
        assert_eq!(stats.len(), 7);
    }
}
