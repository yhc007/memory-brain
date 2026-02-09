//! Watch Mode - Real-time Memory Monitoring
//!
//! Monitor memory brain activity in real-time:
//! - New memories added
//! - Memory access patterns
//! - Strength changes
//! - Index statistics

use std::time::{Duration, Instant};
use std::io::{Write, stdout};
use crate::Brain;

/// Watch configuration
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Refresh interval in milliseconds
    pub interval_ms: u64,
    /// Show detailed stats
    pub detailed: bool,
    /// Clear screen on refresh
    pub clear_screen: bool,
    /// Max iterations (0 = infinite)
    pub max_iterations: usize,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            interval_ms: 1000,
            detailed: false,
            clear_screen: true,
            max_iterations: 0,
        }
    }
}

/// Memory snapshot for comparison
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MemorySnapshot {
    timestamp: Instant,
    semantic_count: usize,
    episodic_count: usize,
    procedural_count: usize,
    index_keywords: usize,
    index_docs: usize,
    bloom_items: usize,
}

impl MemorySnapshot {
    fn from_brain(brain: &Brain) -> Self {
        let index_stats = brain.keyword_index.stats();
        let bloom_stats = brain.keyword_bloom.stats();
        
        Self {
            timestamp: Instant::now(),
            semantic_count: brain.semantic.search("", 100000).map(|v| v.len()).unwrap_or(0),
            episodic_count: brain.episodic.search("", 100000).map(|v| v.len()).unwrap_or(0),
            procedural_count: brain.procedural.search("", 100000).map(|v| v.len()).unwrap_or(0),
            index_keywords: index_stats.unique_keywords,
            index_docs: index_stats.documents,
            bloom_items: bloom_stats.items_added,
        }
    }

    fn total_memories(&self) -> usize {
        self.semantic_count + self.episodic_count + self.procedural_count
    }

    fn diff(&self, other: &MemorySnapshot) -> SnapshotDiff {
        SnapshotDiff {
            semantic_delta: self.semantic_count as i64 - other.semantic_count as i64,
            episodic_delta: self.episodic_count as i64 - other.episodic_count as i64,
            procedural_delta: self.procedural_count as i64 - other.procedural_count as i64,
            keywords_delta: self.index_keywords as i64 - other.index_keywords as i64,
        }
    }
}

#[derive(Debug)]
struct SnapshotDiff {
    semantic_delta: i64,
    episodic_delta: i64,
    procedural_delta: i64,
    keywords_delta: i64,
}

impl SnapshotDiff {
    fn has_changes(&self) -> bool {
        self.semantic_delta != 0 
            || self.episodic_delta != 0 
            || self.procedural_delta != 0
            || self.keywords_delta != 0
    }
}

/// Watch runner
pub struct MemoryWatcher<'a> {
    brain: &'a Brain,
    config: WatchConfig,
    last_snapshot: Option<MemorySnapshot>,
    iteration: usize,
    start_time: Instant,
}

impl<'a> MemoryWatcher<'a> {
    pub fn new(brain: &'a Brain) -> Self {
        Self {
            brain,
            config: WatchConfig::default(),
            last_snapshot: None,
            iteration: 0,
            start_time: Instant::now(),
        }
    }

    pub fn with_config(brain: &'a Brain, config: WatchConfig) -> Self {
        Self {
            brain,
            config,
            last_snapshot: None,
            iteration: 0,
            start_time: Instant::now(),
        }
    }

    /// Run the watch loop
    pub fn run(&mut self) -> std::io::Result<()> {
        self.start_time = Instant::now();
        
        loop {
            if self.config.clear_screen {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen and move cursor to top
            }

            self.display_dashboard()?;
            
            self.iteration += 1;
            
            if self.config.max_iterations > 0 && self.iteration >= self.config.max_iterations {
                break;
            }

            std::thread::sleep(Duration::from_millis(self.config.interval_ms));
        }

        Ok(())
    }

    /// Run once (for testing or single snapshot)
    pub fn run_once(&mut self) -> std::io::Result<()> {
        self.display_dashboard()
    }

    fn display_dashboard(&mut self) -> std::io::Result<()> {
        let current = MemorySnapshot::from_brain(self.brain);
        let elapsed = self.start_time.elapsed();

        // Header
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  🧠 Memory Brain Watch                                       ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        // Time info
        println!("║  ⏱️  Uptime: {:02}:{:02}:{:02}    Refresh: {}ms                     ║",
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60,
            self.config.interval_ms
        );

        println!("╠══════════════════════════════════════════════════════════════╣");

        // Memory counts
        println!("║  📊 Memory Statistics                                        ║");
        println!("║  ├─ Semantic:   {:>6} memories                             ║", current.semantic_count);
        println!("║  ├─ Episodic:   {:>6} memories                             ║", current.episodic_count);
        println!("║  ├─ Procedural: {:>6} memories                             ║", current.procedural_count);
        println!("║  └─ Total:      {:>6} memories                             ║", current.total_memories());

        println!("╠══════════════════════════════════════════════════════════════╣");

        // Index stats
        println!("║  🔍 Index Statistics                                         ║");
        println!("║  ├─ Keywords:   {:>6}                                      ║", current.index_keywords);
        println!("║  ├─ Documents:  {:>6}                                      ║", current.index_docs);
        println!("║  └─ Bloom:      {:>6} items                                ║", current.bloom_items);

        // Show changes if we have a previous snapshot
        if let Some(ref last) = self.last_snapshot {
            let diff = current.diff(last);
            
            if diff.has_changes() {
                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║  📈 Changes Since Last Refresh                               ║");
                
                if diff.semantic_delta != 0 {
                    let _sign = if diff.semantic_delta > 0 { "+" } else { "" };
                    println!("║  ├─ Semantic:  {:>+6}                                       ║", diff.semantic_delta);
                }
                if diff.episodic_delta != 0 {
                    println!("║  ├─ Episodic:  {:>+6}                                       ║", diff.episodic_delta);
                }
                if diff.keywords_delta != 0 {
                    println!("║  └─ Keywords:  {:>+6}                                       ║", diff.keywords_delta);
                }
            }
        }

        if self.config.detailed {
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  📋 Recent Memories                                          ║");
            
            if let Ok(recent) = self.brain.semantic.search("", 5) {
                for (i, mem) in recent.iter().take(3).enumerate() {
                    let content = truncate(&mem.content, 45);
                    println!("║  {}. {}  ║", i + 1, content);
                }
            }
        }

        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\n  Press Ctrl+C to exit");

        stdout().flush()?;

        self.last_snapshot = Some(current);
        Ok(())
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        format!("{:<width$}", s, width = max_len)
    } else {
        format!("{}...", s.chars().take(max_len - 3).collect::<String>())
    }
}

/// Quick watch function
pub fn watch(brain: &Brain, interval_ms: u64, detailed: bool) -> std::io::Result<()> {
    let config = WatchConfig {
        interval_ms,
        detailed,
        ..Default::default()
    };
    
    MemoryWatcher::with_config(brain, config).run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10).trim(), "hello");
        assert_eq!(truncate("hello world test", 10), "hello w...");
    }

    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert_eq!(config.interval_ms, 1000);
        assert!(!config.detailed);
        assert!(config.clear_screen);
    }
}
