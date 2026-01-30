use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of memory storage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Working,    // Short-term, volatile
    Episodic,   // "When did what" - events
    Semantic,   // Facts and concepts
    Procedural, // Patterns and habits
}

/// Emotional valence affects memory strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Emotion {
    Neutral,
    Positive,
    Negative,
    Surprise,
}

/// A single memory item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    pub content: String,
    pub context: Option<String>,
    pub memory_type: MemoryType,
    pub emotion: Emotion,
    
    // Timestamps
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    
    // Memory strength (0.0 - 1.0)
    pub strength: f32,
    
    // Embedding vector for similarity search
    pub embedding: Option<Vec<f32>>,
    
    // Associations to other memories
    pub associations: Vec<Uuid>,
    
    // Tags for categorization
    pub tags: Vec<String>,
}

impl MemoryItem {
    pub fn new(content: &str, context: Option<&str>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content: content.to_string(),
            context: context.map(|s| s.to_string()),
            memory_type: MemoryType::Working, // Default to working memory
            emotion: Emotion::Neutral,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            strength: 1.0,
            embedding: None,
            associations: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Calculate relevance score based on strength, recency, and access frequency
    pub fn relevance_score(&self) -> f32 {
        let recency = self.recency_factor();
        let frequency = (self.access_count as f32).ln() / 10.0;
        
        self.strength * 0.5 + recency * 0.3 + frequency * 0.2
    }

    /// Recency factor (1.0 for just accessed, decays over time)
    fn recency_factor(&self) -> f32 {
        let hours_since = (Utc::now() - self.last_accessed).num_hours() as f32;
        (-hours_since / 168.0).exp() // Half-life of ~1 week
    }

    /// Mark as accessed (strengthens memory)
    pub fn access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
        // Strengthen memory on access (up to 1.0)
        self.strength = (self.strength + 0.1).min(1.0);
    }

    /// Apply decay (forgetting)
    pub fn decay(&mut self, factor: f32) {
        self.strength *= factor;
    }

    /// Check if memory is too weak to keep
    pub fn is_forgotten(&self) -> bool {
        self.strength < 0.1
    }

    /// Set memory type
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type;
        self
    }

    /// Set emotion
    pub fn with_emotion(mut self, emotion: Emotion) -> Self {
        // Emotional memories are stronger
        let is_emotional = !matches!(emotion, Emotion::Neutral);
        self.emotion = emotion;
        if is_emotional {
            self.strength = (self.strength * 1.5).min(1.0);
        }
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add association to another memory
    pub fn associate(&mut self, other_id: Uuid) {
        if !self.associations.contains(&other_id) {
            self.associations.push(other_id);
        }
    }
}

/// Query for memory search
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub text: String,
    pub memory_types: Vec<MemoryType>,
    pub min_strength: f32,
    pub limit: usize,
    pub tags: Vec<String>,
}

impl MemoryQuery {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            memory_types: vec![MemoryType::Episodic, MemoryType::Semantic, MemoryType::Procedural],
            min_strength: 0.1,
            limit: 10,
            tags: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_item_creation() {
        let item = MemoryItem::new("test content", Some("context"));
        
        assert_eq!(item.content, "test content");
        assert_eq!(item.context, Some("context".to_string()));
        assert_eq!(item.access_count, 1);
        assert_eq!(item.strength, 1.0);
    }

    #[test]
    fn test_memory_item_access() {
        let mut item = MemoryItem::new("test", None);
        let initial_strength = item.strength;
        
        item.access();
        
        assert_eq!(item.access_count, 2);
        assert!(item.strength >= initial_strength);
    }

    #[test]
    fn test_memory_item_decay() {
        let mut item = MemoryItem::new("test", None);
        
        item.decay(0.5);
        
        assert_eq!(item.strength, 0.5);
    }

    #[test]
    fn test_memory_forgotten() {
        let mut item = MemoryItem::new("test", None);
        
        item.decay(0.05);
        
        assert!(item.is_forgotten());
    }

    #[test]
    fn test_emotional_memory_stronger() {
        // Start with lower strength to see the boost
        let mut neutral = MemoryItem::new("neutral", None);
        neutral.strength = 0.5;
        
        let mut emotional = MemoryItem::new("emotional", None);
        emotional.strength = 0.5;
        let emotional = emotional.with_emotion(Emotion::Positive);
        
        // Emotional memory should be boosted (0.5 * 1.5 = 0.75)
        assert!(emotional.strength > neutral.strength);
    }

    #[test]
    fn test_memory_with_tags() {
        let item = MemoryItem::new("test", None)
            .with_tags(vec!["rust".to_string(), "programming".to_string()]);
        
        assert_eq!(item.tags.len(), 2);
        assert!(item.tags.contains(&"rust".to_string()));
    }

    #[test]
    fn test_memory_association() {
        let mut item1 = MemoryItem::new("item 1", None);
        let item2 = MemoryItem::new("item 2", None);
        
        item1.associate(item2.id);
        
        assert_eq!(item1.associations.len(), 1);
        assert!(item1.associations.contains(&item2.id));
        
        // Adding same association again should not duplicate
        item1.associate(item2.id);
        assert_eq!(item1.associations.len(), 1);
    }

    #[test]
    fn test_relevance_score() {
        let item = MemoryItem::new("test", None);
        let score = item.relevance_score();
        
        // Score should be positive
        assert!(score > 0.0);
    }
}
