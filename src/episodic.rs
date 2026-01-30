//! Episodic Memory - "When did what happen"
//! 
//! Autobiographical memory for events and experiences:
//! - Stores specific events with temporal context
//! - "Yesterday I fixed that async bug"
//! - "Last week we discussed the API design"
//! - Includes emotional context

use crate::types::{MemoryItem, MemoryType};
use crate::forgetting::ForgettingCurve;
use crate::storage::Storage;

pub struct EpisodicMemory {
    storage: Storage,
}

impl EpisodicMemory {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = Storage::new(db_path, "episodic")?;
        Ok(Self { storage })
    }

    /// Store an episodic memory
    pub fn store(&mut self, mut item: MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        item.memory_type = MemoryType::Episodic;
        self.storage.save(&item)?;
        Ok(())
    }

    /// Search episodic memories
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.search(query, limit)
    }

    /// Get memories from a specific time range
    pub fn get_by_time_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.get_by_time_range(start, end)
    }

    /// Get recent memories
    pub fn get_recent(&self, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.get_recent(limit)
    }

    /// Apply forgetting curve to old memories
    pub fn apply_forgetting(&mut self, curve: &ForgettingCurve) -> Result<(), Box<dyn std::error::Error>> {
        let all = self.storage.get_all()?;
        for mut item in all {
            let decay = curve.calculate_decay(&item);
            item.decay(decay);
            
            if item.is_forgotten() {
                self.storage.delete(&item.id)?;
            } else {
                self.storage.update(&item)?;
            }
        }
        Ok(())
    }

    /// Associate two episodic memories
    pub fn associate(&mut self, id1: uuid::Uuid, id2: uuid::Uuid) -> Result<(), Box<dyn std::error::Error>> {
        self.storage.add_association(id1, id2)?;
        self.storage.add_association(id2, id1)?;
        Ok(())
    }

    /// Get memories associated with a given memory
    pub fn get_associated(&self, id: uuid::Uuid) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.storage.get_associated(id)
    }
}
