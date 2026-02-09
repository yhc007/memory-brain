//! Sam's Memory Storage
//!
//! Personal memory system for Sam (디지털 여우 🦊)
//! Stores conversations, learnings, and context for continuity.

use crate::{Brain, MemoryItem, Embedder, HnswIndex};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Sam's memory categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SamMemoryType {
    /// Conversation with Paul
    Conversation,
    /// Something learned (facts, preferences)
    Learning,
    /// Project/task context
    Project,
    /// Decision or choice made
    Decision,
    /// Lesson learned from mistake
    Lesson,
    /// Paul's preference
    Preference,
    /// Scheduled reminder or task
    Task,
}

impl std::fmt::Display for SamMemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamMemoryType::Conversation => write!(f, "💬"),
            SamMemoryType::Learning => write!(f, "📚"),
            SamMemoryType::Project => write!(f, "🔧"),
            SamMemoryType::Decision => write!(f, "⚖️"),
            SamMemoryType::Lesson => write!(f, "💡"),
            SamMemoryType::Preference => write!(f, "❤️"),
            SamMemoryType::Task => write!(f, "📋"),
        }
    }
}

/// A memory item specific to Sam
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamMemory {
    pub id: Uuid,
    pub content: String,
    pub memory_type: SamMemoryType,
    pub source: String, // "chat", "heartbeat", "cron", etc.
    pub channel: Option<String>, // "imessage", "discord", etc.
    pub importance: u8, // 1-5
    pub created_at: DateTime<Utc>,
    pub tags: Vec<String>,
}

impl SamMemory {
    pub fn new(content: &str, memory_type: SamMemoryType) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.to_string(),
            memory_type,
            source: "manual".to_string(),
            channel: None,
            importance: 3,
            created_at: Utc::now(),
            tags: Vec::new(),
        }
    }

    pub fn conversation(content: &str, channel: &str) -> Self {
        Self {
            memory_type: SamMemoryType::Conversation,
            source: "chat".to_string(),
            channel: Some(channel.to_string()),
            ..Self::new(content, SamMemoryType::Conversation)
        }
    }

    pub fn learning(content: &str) -> Self {
        Self {
            memory_type: SamMemoryType::Learning,
            importance: 4,
            ..Self::new(content, SamMemoryType::Learning)
        }
    }

    pub fn preference(content: &str) -> Self {
        Self {
            memory_type: SamMemoryType::Preference,
            importance: 5,
            tags: vec!["paul".to_string()],
            ..Self::new(content, SamMemoryType::Preference)
        }
    }

    pub fn lesson(content: &str) -> Self {
        Self {
            memory_type: SamMemoryType::Lesson,
            importance: 5,
            ..Self::new(content, SamMemoryType::Lesson)
        }
    }

    pub fn project(name: &str, details: &str) -> Self {
        Self {
            content: format!("{}: {}", name, details),
            memory_type: SamMemoryType::Project,
            tags: vec![name.to_lowercase()],
            ..Self::new(details, SamMemoryType::Project)
        }
    }

    pub fn with_importance(mut self, importance: u8) -> Self {
        self.importance = importance.min(5).max(1);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }
}

/// Sam's Brain - wrapper around memory-brain for Sam-specific operations
pub struct SamBrain {
    brain: Brain,
    hnsw: HnswIndex,
    embedder: Arc<dyn Embedder>,
}

impl SamBrain {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let brain = Brain::new(db_path)?;
        let embedder = brain.embedder().clone();
        let dim = embedder.dimension();
        
        Ok(Self {
            brain,
            hnsw: HnswIndex::new(dim),
            embedder,
        })
    }

    /// Store a Sam memory
    pub fn remember(&mut self, memory: SamMemory) -> Result<Uuid, Box<dyn std::error::Error>> {
        let id = memory.id;
        
        // Convert to MemoryItem
        let embedding = self.embedder.embed(&memory.content);
        let mut item = MemoryItem::new(&memory.content, Some(&format!("{}", memory.memory_type)));
        item.id = memory.id;
        item.tags = memory.tags.clone();
        item.tags.push(format!("sam:{:?}", memory.memory_type).to_lowercase());
        item.tags.push(format!("importance:{}", memory.importance));
        item.embedding = Some(embedding.clone());
        
        // Store in brain
        self.brain.semantic.store(item)?;
        
        // Add to HNSW
        let _ = self.hnsw.add(id, embedding);
        
        Ok(id)
    }

    /// Remember a conversation
    pub fn remember_conversation(&mut self, content: &str, channel: &str) -> Result<Uuid, Box<dyn std::error::Error>> {
        let memory = SamMemory::conversation(content, channel);
        self.remember(memory)
    }

    /// Remember something learned
    pub fn remember_learning(&mut self, content: &str) -> Result<Uuid, Box<dyn std::error::Error>> {
        let memory = SamMemory::learning(content);
        self.remember(memory)
    }

    /// Remember Paul's preference
    pub fn remember_preference(&mut self, content: &str) -> Result<Uuid, Box<dyn std::error::Error>> {
        let memory = SamMemory::preference(content);
        self.remember(memory)
    }

    /// Remember a lesson learned
    pub fn remember_lesson(&mut self, content: &str) -> Result<Uuid, Box<dyn std::error::Error>> {
        let memory = SamMemory::lesson(content);
        self.remember(memory)
    }

    /// Recall memories related to a query
    pub fn recall(&mut self, query: &str, limit: usize) -> Vec<MemoryItem> {
        self.brain.recall(query, limit)
    }

    /// Fast recall using HNSW
    pub fn fast_recall(&self, query: &str, limit: usize) -> Vec<(Uuid, f32)> {
        let query_embedding = self.embedder.embed(query);
        self.hnsw.search(&query_embedding, limit)
    }

    /// Get memories by type
    pub fn recall_by_type(&self, memory_type: SamMemoryType, _limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let tag = format!("sam:{:?}", memory_type).to_lowercase();
        self.brain.semantic.get_by_tag(&tag)
    }

    /// Get all preferences
    pub fn get_preferences(&self) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.recall_by_type(SamMemoryType::Preference, 100)
    }

    /// Get all lessons
    pub fn get_lessons(&self) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.recall_by_type(SamMemoryType::Lesson, 100)
    }

    /// Get stats
    pub fn stats(&self) -> SamBrainStats {
        let hnsw_stats = self.hnsw.stats();
        SamBrainStats {
            total_memories: hnsw_stats.count,
            embedding_dim: hnsw_stats.dimension,
        }
    }
}

#[derive(Debug)]
pub struct SamBrainStats {
    pub total_memories: usize,
    pub embedding_dim: usize,
}

impl std::fmt::Display for SamBrainStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "🧠 Sam's Brain: {} memories ({}d embeddings)", 
            self.total_memories, self.embedding_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sam_memory_creation() {
        let mem = SamMemory::learning("Rust uses ownership for memory safety");
        assert_eq!(mem.memory_type, SamMemoryType::Learning);
        assert_eq!(mem.importance, 4);
    }

    #[test]
    fn test_sam_brain() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("sam.db");
        let mut brain = SamBrain::new(db_path.to_str().unwrap()).unwrap();

        // Store some memories
        brain.remember_learning("Paul prefers 반말").unwrap();
        brain.remember_preference("Paul likes Rust").unwrap();
        brain.remember_lesson("Always commit before big changes").unwrap();

        // Recall
        let results = brain.recall("Paul", 5);
        assert!(!results.is_empty());
    }
}
