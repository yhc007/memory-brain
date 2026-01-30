//! Memory Consolidation
//! 
//! Decides which working memories should move to long-term storage.
//! Like the brain during sleep, consolidates important memories.

use crate::types::{Emotion, MemoryItem, MemoryType};

pub struct Consolidator {
    /// Minimum strength for auto-consolidation
    strength_threshold: f32,
    /// Repetition threshold (access count)
    repetition_threshold: u32,
}

impl Consolidator {
    pub fn new() -> Self {
        Self {
            strength_threshold: 0.6,
            repetition_threshold: 3,
        }
    }

    /// Decide if a memory should be consolidated to long-term
    pub fn should_consolidate(&self, item: &MemoryItem) -> bool {
        // Emotional memories are always consolidated
        if !matches!(item.emotion, Emotion::Neutral) {
            return true;
        }

        // Strong memories are consolidated
        if item.strength >= self.strength_threshold {
            return true;
        }

        // Frequently accessed memories are consolidated
        if item.access_count >= self.repetition_threshold {
            return true;
        }

        false
    }

    /// Classify what type of long-term memory this should be
    pub fn classify(&self, item: &MemoryItem) -> MemoryType {
        let content_lower = item.content.to_lowercase();

        // Check for procedural patterns (if/when/how patterns)
        if content_lower.contains("when ") && content_lower.contains(" then ")
            || content_lower.contains("pattern:")
            || content_lower.contains("how to ")
            || content_lower.contains("always ")
            || content_lower.contains("never ")
        {
            return MemoryType::Procedural;
        }

        // Check for facts/concepts (semantic)
        if content_lower.contains(" is ")
            || content_lower.contains(" are ")
            || content_lower.contains(" means ")
            || content_lower.contains("definition:")
            || content_lower.contains("fact:")
        {
            return MemoryType::Semantic;
        }

        // Check for events (episodic) - time references
        if content_lower.contains("yesterday")
            || content_lower.contains("today")
            || content_lower.contains("last ")
            || content_lower.contains("just now")
            || content_lower.contains("earlier")
            || item.context.is_some()
        {
            return MemoryType::Episodic;
        }

        // Default to episodic (most memories are experiences)
        MemoryType::Episodic
    }

    /// Extract key information for summarization
    pub fn extract_key_info(&self, item: &MemoryItem) -> Vec<String> {
        let mut keys = Vec::new();

        // Simple keyword extraction (could use NLP later)
        for word in item.content.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() > 4 && !is_stop_word(clean) {
                keys.push(clean.to_lowercase());
            }
        }

        keys.truncate(5);
        keys
    }
}

impl Default for Consolidator {
    fn default() -> Self {
        Self::new()
    }
}

fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "this", "that", "with", "from", "have", "been", "were",
        "will", "would", "could", "should", "about", "which", "their",
        "there", "where", "when", "what", "some", "other", "just",
    ];
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryItem, Emotion};

    #[test]
    fn test_should_consolidate_strong_memory() {
        let consolidator = Consolidator::new();
        let mut item = MemoryItem::new("important fact", None);
        item.strength = 0.8;
        
        assert!(consolidator.should_consolidate(&item));
    }

    #[test]
    fn test_should_consolidate_emotional_memory() {
        let consolidator = Consolidator::new();
        let item = MemoryItem::new("exciting news", None)
            .with_emotion(Emotion::Positive);
        
        assert!(consolidator.should_consolidate(&item));
    }

    #[test]
    fn test_should_not_consolidate_weak_memory() {
        let consolidator = Consolidator::new();
        let mut item = MemoryItem::new("trivial info", None);
        item.strength = 0.3;
        item.access_count = 1;
        
        assert!(!consolidator.should_consolidate(&item));
    }

    #[test]
    fn test_classify_episodic() {
        let consolidator = Consolidator::new();
        let item = MemoryItem::new("yesterday I fixed the bug", None);
        
        let mem_type = consolidator.classify(&item);
        assert_eq!(mem_type, MemoryType::Episodic);
    }

    #[test]
    fn test_classify_semantic() {
        let consolidator = Consolidator::new();
        let item = MemoryItem::new("Rust is a systems programming language", None);
        
        let mem_type = consolidator.classify(&item);
        assert_eq!(mem_type, MemoryType::Semantic);
    }

    #[test]
    fn test_classify_procedural() {
        let consolidator = Consolidator::new();
        let item = MemoryItem::new("Pattern: when error occurs, use Result type", None);
        
        let mem_type = consolidator.classify(&item);
        assert_eq!(mem_type, MemoryType::Procedural);
    }

    #[test]
    fn test_extract_key_info() {
        let consolidator = Consolidator::new();
        let item = MemoryItem::new("Rust provides memory safety through ownership", None);
        
        let keys = consolidator.extract_key_info(&item);
        assert!(!keys.is_empty());
        assert!(keys.len() <= 5);
    }
}
