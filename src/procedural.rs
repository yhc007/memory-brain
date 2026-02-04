//! Procedural Memory - Patterns and Habits
//! 
//! "How to do things":
//! - Code patterns ("when I see X, I usually do Y")
//! - Workflow habits
//! - Problem-solving patterns
//! - Strengthens with repetition

use crate::types::{MemoryItem, MemoryType};
use crate::storage::Storage;
use serde::{Deserialize, Serialize};

/// A procedural pattern (trigger → action)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub trigger: String,      // When I see this...
    pub action: String,       // I do this
    pub success_count: u32,   // Times it worked
    pub failure_count: u32,   // Times it failed
    pub tags: Vec<String>,
}

impl Pattern {
    pub fn new(trigger: &str, action: &str) -> Self {
        Self {
            trigger: trigger.to_string(),
            action: action.to_string(),
            success_count: 1,
            failure_count: 0,
            tags: Vec::new(),
        }
    }

    /// Confidence score (0.0 - 1.0)
    pub fn confidence(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.5;
        }
        self.success_count as f32 / total as f32
    }

    /// Record success
    pub fn success(&mut self) {
        self.success_count += 1;
    }

    /// Record failure
    pub fn failure(&mut self) {
        self.failure_count += 1;
    }
}

pub struct ProceduralMemory {
    storage: Storage,
}

impl ProceduralMemory {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = Storage::new(db_path, "procedural")?;
        Ok(Self { storage })
    }

    /// Store a procedural memory (pattern)
    pub fn store(&mut self, mut item: MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        item.memory_type = MemoryType::Procedural;
        self.storage.save(&item)?;
        Ok(())
    }

    /// Learn a new pattern
    pub fn learn_pattern(&mut self, pattern: Pattern) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string(&pattern)?;
        let item = MemoryItem::new(&content, None)
            .with_type(MemoryType::Procedural)
            .with_tags(pattern.tags.clone());
        self.storage.save(&item)?;
        Ok(())
    }

    /// Search for relevant patterns
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.search(query, limit)
    }

    /// Find matching patterns for a trigger
    pub fn find_patterns(&self, trigger: &str) -> Result<Vec<Pattern>, Box<dyn std::error::Error>> {
        let items = self.storage.search(trigger, 10)?;
        let patterns: Vec<Pattern> = items
            .into_iter()
            .filter_map(|item| serde_json::from_str(&item.content).ok())
            .filter(|p: &Pattern| p.confidence() > 0.3) // Only confident patterns
            .collect();
        Ok(patterns)
    }

    /// Record feedback on a pattern
    pub fn feedback(&mut self, trigger: &str, success: bool) -> Result<(), Box<dyn std::error::Error>> {
        let items = self.storage.search(trigger, 1)?;
        if let Some(mut item) = items.into_iter().next() {
            if let Ok(mut pattern) = serde_json::from_str::<Pattern>(&item.content) {
                if success {
                    pattern.success();
                } else {
                    pattern.failure();
                }
                item.content = serde_json::to_string(&pattern)?;
                item.access();
                self.storage.update(&item)?;
            }
        }
        Ok(())
    }
}
