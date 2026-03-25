//! NeocortexActor - Slow Semantic Memory
//!
//! Implements the neocortex role in CLS:
//! - Slow consolidation of semantic knowledge
//! - Pattern extraction and generalization
//! - Low plasticity, robust representations

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{info, debug};

use crate::messages::*;

/// A semantic concept/knowledge unit
#[derive(Debug, Clone)]
pub struct Concept {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Source memories that contributed to this concept
    pub source_memories: Vec<MemoryId>,
    /// Related concepts
    pub relations: Vec<(String, f32)>, // (concept_id, strength)
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// How well established this concept is (0.0 - 1.0)
    pub consolidation_level: f32,
}

impl Concept {
    pub fn new(name: String, description: String, source_memories: Vec<MemoryId>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            source_memories,
            relations: vec![],
            created_at: now,
            updated_at: now,
            consolidation_level: 0.1, // Starts weak
        }
    }
}

/// Configuration for NeocortexActor
#[derive(Debug, Clone)]
pub struct NeocortexConfig {
    /// Minimum similarity to create association
    pub association_threshold: f32,
    /// Rate at which concepts consolidate
    pub consolidation_rate: f32,
}

impl Default for NeocortexConfig {
    fn default() -> Self {
        Self {
            association_threshold: 0.3,
            consolidation_rate: 0.05,
        }
    }
}

/// NeocortexActor - manages slow semantic memory
pub struct NeocortexActor {
    config: NeocortexConfig,
    /// Stored concepts/knowledge
    concepts: HashMap<String, Concept>,
    /// Index by name for quick lookup
    name_index: HashMap<String, String>, // name -> id
}

impl NeocortexActor {
    pub fn new(config: NeocortexConfig) -> Self {
        Self {
            config,
            concepts: HashMap::new(),
            name_index: HashMap::new(),
        }
    }

    /// Find associations between memories based on content similarity
    /// Returns pairs of (memory_id, memory_id, similarity_score)
    pub fn associate(&self, memories: &[Memory]) -> Vec<(MemoryId, MemoryId, f32)> {
        let mut associations = Vec::new();
        
        // Simple word overlap similarity for now
        // TODO: Use embeddings for real semantic similarity
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                let sim = self.compute_similarity(&memories[i], &memories[j]);
                if sim >= self.config.association_threshold {
                    associations.push((memories[i].id, memories[j].id, sim));
                }
            }
        }
        
        debug!("Found {} associations among {} memories", 
               associations.len(), memories.len());
        associations
    }

    /// Compute similarity between two memories
    fn compute_similarity(&self, a: &Memory, b: &Memory) -> f32 {
        // Simple Jaccard similarity on words
        let words_a: std::collections::HashSet<_> = a.content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let words_b: std::collections::HashSet<_> = b.content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Extract patterns from a set of memories
    /// Returns common themes/keywords
    pub fn extract_patterns(&self, memories: &[Memory]) -> Vec<String> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Count word frequencies across all memories
        for memory in memories {
            let words: std::collections::HashSet<_> = memory.content
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 3) // Skip short words
                .map(|s| s.to_string())
                .collect();
            
            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }
        
        // Find words that appear in multiple memories
        let threshold = (memories.len() / 2).max(2);
        let mut patterns: Vec<_> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .collect();
        
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        
        let result: Vec<String> = patterns.into_iter()
            .take(10)
            .map(|(word, _)| word)
            .collect();
        
        debug!("Extracted {} patterns from {} memories", 
               result.len(), memories.len());
        result
    }

    /// Generalize from specific memories to a concept
    pub fn generalize(&mut self, memories: &[Memory]) -> Option<String> {
        if memories.is_empty() {
            return None;
        }
        
        // Extract patterns to form concept description
        let patterns = self.extract_patterns(memories);
        if patterns.is_empty() {
            return None;
        }
        
        // Create concept name from top patterns
        let concept_name = patterns.iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("_");
        
        let description = format!(
            "Concept derived from {} memories. Key themes: {}",
            memories.len(),
            patterns.join(", ")
        );
        
        let source_ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();
        let concept = Concept::new(concept_name.clone(), description, source_ids);
        
        let id = concept.id.clone();
        self.name_index.insert(concept_name.clone(), id.clone());
        self.concepts.insert(id.clone(), concept);
        
        info!("Generalized concept: {}", concept_name);
        Some(concept_name)
    }

    /// Query for semantic knowledge
    pub fn query(&self, concept_name: &str) -> Option<&Concept> {
        self.name_index
            .get(concept_name)
            .and_then(|id| self.concepts.get(id))
    }

    /// Store consolidated knowledge
    pub fn store_knowledge(
        &mut self,
        concept_name: String,
        description: String,
        source_memories: Vec<MemoryId>,
    ) -> String {
        let concept = Concept::new(concept_name.clone(), description, source_memories);
        let id = concept.id.clone();
        
        self.name_index.insert(concept_name.clone(), id.clone());
        self.concepts.insert(id.clone(), concept);
        
        info!("Stored knowledge: {}", concept_name);
        id
    }

    /// Strengthen a concept (called during consolidation)
    pub fn strengthen(&mut self, concept_id: &str, delta: f32) {
        if let Some(concept) = self.concepts.get_mut(concept_id) {
            concept.consolidation_level = 
                (concept.consolidation_level + delta).min(1.0);
            concept.updated_at = Utc::now();
        }
    }

    /// Add relation between concepts
    pub fn add_relation(&mut self, from_id: &str, to_id: &str, strength: f32) {
        if let Some(concept) = self.concepts.get_mut(from_id) {
            // Update existing or add new
            if let Some(rel) = concept.relations.iter_mut()
                .find(|(id, _)| id == to_id) 
            {
                rel.1 = (rel.1 + strength).min(1.0);
            } else {
                concept.relations.push((to_id.to_string(), strength));
            }
        }
    }

    /// Get all concepts
    pub fn all_concepts(&self) -> Vec<&Concept> {
        self.concepts.values().collect()
    }

    /// Get concept count
    pub fn count(&self) -> usize {
        self.concepts.len()
    }

    /// Process a message and return response
    pub fn handle(&mut self, msg: NeocortexMessage, memories: &[Memory]) -> NeocortexResponse {
        match msg {
            NeocortexMessage::Associate { memory_ids: _ } => {
                // Use provided memories for association
                let links = self.associate(memories);
                NeocortexResponse::Associations { links }
            }
            NeocortexMessage::ExtractPatterns { memories: mems } => {
                let patterns = self.extract_patterns(&mems);
                NeocortexResponse::Patterns { patterns }
            }
            NeocortexMessage::Generalize { memories: mems } => {
                match self.generalize(&mems) {
                    Some(concept) => NeocortexResponse::Generalized { concept },
                    None => NeocortexResponse::Error {
                        message: "Could not generalize from memories".to_string()
                    }
                }
            }
            NeocortexMessage::Query { concept } => {
                let knowledge = self.query(&concept).map(|c| c.description.clone());
                NeocortexResponse::QueryResult { knowledge }
            }
            NeocortexMessage::StoreKnowledge { concept, description, source_memories } => {
                let id = self.store_knowledge(concept, description, source_memories);
                NeocortexResponse::KnowledgeStored { id }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(content: &str) -> Memory {
        Memory::new(content.to_string(), MemoryContext::default())
    }

    #[test]
    fn test_association() {
        // Use lower threshold for testing
        let config = NeocortexConfig {
            association_threshold: 0.1,
            ..Default::default()
        };
        let actor = NeocortexActor::new(config);
        
        let memories = vec![
            make_memory("Rust programming language systems"),
            make_memory("Rust programming memory safety"),
            make_memory("Python data science machine learning"),
        ];
        
        let associations = actor.associate(&memories);
        
        // Rust memories should be associated (share "Rust" and "programming")
        assert!(!associations.is_empty(), "Expected associations between similar memories");
    }

    #[test]
    fn test_pattern_extraction() {
        let actor = NeocortexActor::new(NeocortexConfig::default());
        
        let memories = vec![
            make_memory("Rust provides memory safety"),
            make_memory("Rust has zero-cost abstractions"),
            make_memory("Rust prevents data races"),
        ];
        
        let patterns = actor.extract_patterns(&memories);
        
        // "rust" should be a common pattern
        assert!(patterns.iter().any(|p| p == "rust"));
    }

    #[test]
    fn test_generalize() {
        let mut actor = NeocortexActor::new(NeocortexConfig::default());
        
        let memories = vec![
            make_memory("Actor model uses message passing"),
            make_memory("Actor systems provide concurrency"),
            make_memory("Actor isolation prevents shared state"),
        ];
        
        let concept = actor.generalize(&memories);
        assert!(concept.is_some());
        
        // Should be able to query the concept
        let concept_name = concept.unwrap();
        let queried = actor.query(&concept_name);
        assert!(queried.is_some());
    }

    #[test]
    fn test_store_and_query() {
        let mut actor = NeocortexActor::new(NeocortexConfig::default());
        
        actor.store_knowledge(
            "pekko".to_string(),
            "Pekko is a Rust actor framework".to_string(),
            vec![],
        );
        
        let result = actor.query("pekko");
        assert!(result.is_some());
        assert!(result.unwrap().description.contains("actor"));
    }
}
