//! Semantic Memory - Facts and Concepts
//! 
//! General knowledge and facts:
//! - "Rust uses ownership for memory safety"
//! - "User prefers async/await over callbacks"
//! - Concepts and relationships
//! - Not tied to specific events

use crate::types::{MemoryItem, MemoryType};
use crate::forgetting::ForgettingCurve;
use crate::storage::Storage;

pub struct SemanticMemory {
    storage: Storage,
}

impl SemanticMemory {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = Storage::new(db_path, "semantic")?;
        Ok(Self { storage })
    }

    /// Store a semantic fact/concept
    pub fn store(&mut self, mut item: MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        item.memory_type = MemoryType::Semantic;
        
        // Check for duplicate/similar facts and merge
        if let Some(existing) = self.find_similar(&item.content)? {
            // Strengthen existing memory instead of adding duplicate
            let mut updated = existing;
            updated.access();
            self.storage.update(&updated)?;
        } else {
            self.storage.save(&item)?;
        }
        Ok(())
    }

    /// Search semantic memories
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.search(query, limit)
    }

    /// Find similar facts (to avoid duplicates)
    fn find_similar(&self, content: &str) -> Result<Option<MemoryItem>, Box<dyn std::error::Error>> {
        let results = self.storage.search(content, 1)?;
        // If high similarity match exists, return it
        if let Some(item) = results.into_iter().next() {
            if content.to_lowercase().contains(&item.content.to_lowercase()) 
                || item.content.to_lowercase().contains(&content.to_lowercase()) {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    /// Get facts by tag
    pub fn get_by_tag(&self, tag: &str) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.get_by_tag(tag)
    }

    /// Execute arbitrary CQL query and return HTML results
    pub fn execute_cql_html(&self, query: &str) -> Result<String, String> {
        self.storage.execute_cql_html(query)
    }

    /// Apply forgetting (semantic memories decay slower)
    pub fn apply_forgetting(&mut self, curve: &ForgettingCurve) -> Result<(), Box<dyn std::error::Error>> {
        let all = self.storage.get_all()?;
        for mut item in all {
            // Semantic memories decay at half the rate of episodic
            let decay = curve.calculate_decay(&item) * 0.5 + 0.5;
            item.decay(decay);
            
            if item.is_forgotten() {
                self.storage.delete(&item.id)?;
            } else {
                self.storage.update(&item)?;
            }
        }
        Ok(())
    }
}
