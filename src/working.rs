//! Working Memory - Short-term, limited capacity
//! 
//! Like the human brain, working memory:
//! - Has limited capacity (~7 items, Miller's Law)
//! - Fast access but volatile
//! - Items pushed out when capacity exceeded
//! - Important items get consolidated to long-term

use crate::types::{MemoryItem, MemoryType};
use std::collections::VecDeque;

/// Working memory with limited capacity
pub struct WorkingMemory {
    items: VecDeque<MemoryItem>,
    capacity: usize,
    importance_threshold: f32,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            importance_threshold: 0.7,
        }
    }

    /// Push a new item to working memory
    /// Returns evicted item if capacity exceeded
    pub fn push(&mut self, mut item: MemoryItem) -> Option<MemoryItem> {
        item.memory_type = MemoryType::Working;
        
        let evicted = if self.items.len() >= self.capacity {
            self.items.pop_front()
        } else {
            None
        };

        self.items.push_back(item);
        evicted
    }

    /// Get all items in working memory
    pub fn get_all(&self) -> Vec<MemoryItem> {
        self.items.iter().cloned().collect()
    }

    /// Search working memory for relevant items
    pub fn search(&self, query: &str) -> Vec<MemoryItem> {
        let query_lower = query.to_lowercase();
        self.items
            .iter()
            .filter(|item| {
                item.content.to_lowercase().contains(&query_lower)
                    || item.context.as_ref().map_or(false, |c| c.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect()
    }

    /// Get items important enough for long-term storage
    pub fn get_important(&self) -> Vec<MemoryItem> {
        self.items
            .iter()
            .filter(|item| item.strength >= self.importance_threshold)
            .cloned()
            .collect()
    }

    /// Clear working memory (like after sleep)
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Current number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the most recent item
    pub fn last(&self) -> Option<&MemoryItem> {
        self.items.back()
    }

    /// Rehearse (strengthen) an item by content match
    pub fn rehearse(&mut self, content: &str) {
        for item in self.items.iter_mut() {
            if item.content.contains(content) {
                item.access();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_limit() {
        let mut wm = WorkingMemory::new(3);
        
        wm.push(MemoryItem::new("first", None));
        wm.push(MemoryItem::new("second", None));
        wm.push(MemoryItem::new("third", None));
        assert_eq!(wm.len(), 3);
        
        // Fourth item should evict first
        let evicted = wm.push(MemoryItem::new("fourth", None));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().content, "first");
        assert_eq!(wm.len(), 3);
    }

    #[test]
    fn test_search() {
        let mut wm = WorkingMemory::new(5);
        wm.push(MemoryItem::new("rust programming", None));
        wm.push(MemoryItem::new("python scripting", None));
        wm.push(MemoryItem::new("rust async", None));
        
        let results = wm.search("rust");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_millers_law() {
        // Miller's Law: 7 ± 2 items
        let mut wm = WorkingMemory::new(7);
        
        for i in 0..7 {
            wm.push(MemoryItem::new(&format!("item {}", i), None));
        }
        assert_eq!(wm.len(), 7);
        
        // Adding 8th should evict first
        wm.push(MemoryItem::new("item 7", None));
        assert_eq!(wm.len(), 7);
        
        // First item should be gone
        let results = wm.search("item 0");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut wm = WorkingMemory::new(5);
        wm.push(MemoryItem::new("test", None));
        wm.push(MemoryItem::new("test2", None));
        
        wm.clear();
        assert!(wm.is_empty());
    }

    #[test]
    fn test_rehearse() {
        let mut wm = WorkingMemory::new(5);
        wm.push(MemoryItem::new("remember this", None));
        
        let initial_count = wm.get_all()[0].access_count;
        wm.rehearse("remember");
        let after_count = wm.get_all()[0].access_count;
        
        assert!(after_count > initial_count);
    }
}
