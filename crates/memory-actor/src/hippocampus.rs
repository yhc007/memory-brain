//! HippocampusActor - Fast Episodic Memory
//!
//! Implements the hippocampus role in CLS:
//! - Fast storage of new experiences
//! - High plasticity
//! - Episodic memory with temporal context

use std::collections::{HashMap, VecDeque};
use chrono::Utc;
use tracing::{info, debug};

use crate::messages::*;

/// Configuration for HippocampusActor
#[derive(Debug, Clone)]
pub struct HippocampusConfig {
    /// Maximum size of working memory (recent memories)
    pub working_memory_size: usize,
    /// Default decay rate for forgetting
    pub default_decay_rate: f32,
    /// Minimum strength before memory is forgotten
    pub min_strength: f32,
}

impl Default for HippocampusConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 100,
            default_decay_rate: 0.1,
            min_strength: 0.01,
        }
    }
}

/// HippocampusActor - manages fast episodic memory
pub struct HippocampusActor {
    config: HippocampusConfig,
    /// All stored memories
    memories: HashMap<MemoryId, Memory>,
    /// Recent memories (working memory) - FIFO queue
    working_memory: VecDeque<MemoryId>,
    // TODO: embedding_client: Option<EmbeddingClient>,
}

impl HippocampusActor {
    pub fn new(config: HippocampusConfig) -> Self {
        Self {
            config,
            memories: HashMap::new(),
            working_memory: VecDeque::new(),
        }
    }

    /// Store a new memory
    pub fn store(&mut self, content: String, context: MemoryContext) -> MemoryId {
        let memory = Memory::new(content, context);
        let id = memory.id;
        
        // Add to main storage
        self.memories.insert(id, memory);
        
        // Add to working memory
        self.working_memory.push_front(id);
        if self.working_memory.len() > self.config.working_memory_size {
            self.working_memory.pop_back();
        }
        
        info!("Stored memory: {}", id);
        id
    }

    /// Recall memories by semantic similarity
    /// NOTE: This is a simplified version. Real implementation needs embeddings.
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        // Simple keyword matching for now
        // TODO: Use embedding similarity with CoreVecDB
        let query_lower = query.to_lowercase();
        
        let mut results: Vec<RecallResult> = self.memories
            .values_mut()
            .filter_map(|memory| {
                let content_lower = memory.content.to_lowercase();
                if content_lower.contains(&query_lower) {
                    // Update access info
                    memory.access_count += 1;
                    memory.last_accessed = Utc::now();
                    // Reinforce on access
                    memory.strength = (memory.strength + 0.1).min(1.0);
                    
                    Some(RecallResult {
                        memory: memory.clone(),
                        similarity: 1.0, // TODO: Real similarity
                    })
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by strength * similarity
        results.sort_by(|a, b| {
            let score_a = a.similarity * a.memory.strength;
            let score_b = b.similarity * b.memory.strength;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        results.truncate(k);
        debug!("Recalled {} memories for query: {}", results.len(), query);
        results
    }

    /// Get a specific memory by ID
    pub fn get(&self, id: &MemoryId) -> Option<Memory> {
        self.memories.get(id).cloned()
    }

    /// Reinforce a memory (increase strength)
    pub fn reinforce(&mut self, id: &MemoryId, delta: f32) -> Option<f32> {
        if let Some(memory) = self.memories.get_mut(id) {
            memory.strength = (memory.strength + delta).clamp(0.0, 1.0);
            memory.last_accessed = Utc::now();
            memory.access_count += 1;
            debug!("Reinforced memory {}: new strength = {}", id, memory.strength);
            Some(memory.strength)
        } else {
            None
        }
    }

    /// Apply forgetting curve to all memories
    /// Uses Ebbinghaus forgetting curve: R = e^(-t/S)
    pub fn apply_forgetting(&mut self, decay_rate: f32) -> usize {
        let now = Utc::now();
        let mut to_remove = Vec::new();
        
        for (id, memory) in self.memories.iter_mut() {
            let duration = now.signed_duration_since(memory.last_accessed);
            let hours_since_access = (duration.num_minutes() as f32 / 60.0).max(0.0);
            
            // Decay based on time since last access
            let decay = (-hours_since_access * decay_rate).exp();
            memory.strength *= decay;
            
            if memory.strength < self.config.min_strength {
                to_remove.push(*id);
            }
        }
        
        let removed_count = to_remove.len();
        for id in &to_remove {
            self.memories.remove(id);
            self.working_memory.retain(|mid| mid != id);
        }
        
        if removed_count > 0 {
            info!("Forgetting applied: {} memories removed", removed_count);
        }
        removed_count
    }

    /// Get recent memories (working memory)
    pub fn get_recent(&self, limit: usize) -> Vec<Memory> {
        self.working_memory
            .iter()
            .take(limit)
            .filter_map(|id| self.memories.get(id).cloned())
            .collect()
    }

    /// Link two memories
    pub fn link(&mut self, source: &MemoryId, target: &MemoryId) -> bool {
        if let Some(memory) = self.memories.get_mut(source) {
            if !memory.links.contains(target) {
                memory.links.push(*target);
                debug!("Linked {} -> {}", source, target);
                return true;
            }
        }
        false
    }

    /// Get all memories (for consolidation)
    pub fn all_memories(&self) -> Vec<Memory> {
        self.memories.values().cloned().collect()
    }

    /// Get memory count
    pub fn count(&self) -> usize {
        self.memories.len()
    }

    /// Process a message and return response
    pub fn handle(&mut self, msg: HippocampusMessage) -> HippocampusResponse {
        match msg {
            HippocampusMessage::Store { content, context } => {
                let id = self.store(content, context);
                HippocampusResponse::Stored { id }
            }
            HippocampusMessage::Recall { query, k } => {
                let results = self.recall(&query, k);
                HippocampusResponse::Recalled { results }
            }
            HippocampusMessage::Get { id } => {
                let memory = self.get(&id);
                HippocampusResponse::Found { memory }
            }
            HippocampusMessage::Reinforce { id, delta } => {
                match self.reinforce(&id, delta) {
                    Some(new_strength) => HippocampusResponse::Reinforced { new_strength },
                    None => HippocampusResponse::Error { 
                        message: format!("Memory not found: {}", id) 
                    },
                }
            }
            HippocampusMessage::ApplyForgetting { decay_rate } => {
                let affected = self.apply_forgetting(decay_rate);
                HippocampusResponse::ForgettingApplied { affected }
            }
            HippocampusMessage::GetRecent { limit } => {
                let memories = self.get_recent(limit);
                HippocampusResponse::RecentMemories { memories }
            }
            HippocampusMessage::Link { source, target } => {
                if self.link(&source, &target) {
                    HippocampusResponse::Linked
                } else {
                    HippocampusResponse::Error {
                        message: "Failed to link memories".to_string()
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() {
        let mut actor = HippocampusActor::new(HippocampusConfig::default());
        
        let id = actor.store(
            "Rust is a systems programming language".to_string(),
            MemoryContext::default(),
        );
        
        assert!(actor.get(&id).is_some());
        
        let results = actor.recall("Rust", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, id);
    }

    #[test]
    fn test_working_memory_limit() {
        let config = HippocampusConfig {
            working_memory_size: 3,
            ..Default::default()
        };
        let mut actor = HippocampusActor::new(config);
        
        for i in 0..5 {
            actor.store(format!("Memory {}", i), MemoryContext::default());
        }
        
        // Working memory should only have 3 most recent
        assert_eq!(actor.working_memory.len(), 3);
        let recent = actor.get_recent(10);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_forgetting() {
        let mut actor = HippocampusActor::new(HippocampusConfig::default());
        
        let id = actor.store("Test memory".to_string(), MemoryContext::default());
        
        // Artificially lower strength
        if let Some(memory) = actor.memories.get_mut(&id) {
            memory.strength = 0.001;
        }
        
        let removed = actor.apply_forgetting(0.1);
        assert_eq!(removed, 1);
        assert!(actor.get(&id).is_none());
    }

    #[test]
    fn test_reinforcement() {
        let mut actor = HippocampusActor::new(HippocampusConfig::default());
        
        let id = actor.store("Test".to_string(), MemoryContext::default());
        
        // Lower strength first
        if let Some(memory) = actor.memories.get_mut(&id) {
            memory.strength = 0.5;
        }
        
        let new_strength = actor.reinforce(&id, 0.3);
        assert_eq!(new_strength, Some(0.8));
    }
}
