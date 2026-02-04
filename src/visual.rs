//! Visual Memory Module - Human brain-inspired image memory
//! 
//! Implements image storage and retrieval using CLIP embeddings,
//! mimicking how the human visual cortex processes and stores images.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Visual memory entry - represents a stored image memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualMemory {
    /// Unique identifier
    pub id: Uuid,
    
    /// Image file path (original location)
    pub image_path: PathBuf,
    
    /// CLIP image embedding (512 or 768 dimensions)
    pub embedding: Vec<f32>,
    
    /// Text description of the image
    pub description: String,
    
    /// Contextual information (who, where, what was happening)
    pub context: VisualContext,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Emotional valence (-1.0 to 1.0, negative to positive)
    pub emotional_valence: f32,
    
    /// Memory strength (0.0 to 1.0, decays over time)
    pub strength: f32,
    
    /// Number of times this memory was recalled
    pub recall_count: u32,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    
    /// Links to related text memories (by UUID)
    pub linked_memories: Vec<Uuid>,
    
    /// Links to related visual memories (by UUID)
    pub linked_visuals: Vec<Uuid>,
}

/// Contextual information for visual memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualContext {
    /// Who was present
    pub people: Vec<String>,
    
    /// Location description
    pub location: Option<String>,
    
    /// Activity or event
    pub activity: Option<String>,
    
    /// Source (camera, screenshot, downloaded, etc.)
    pub source: ImageSource,
}

/// Source of the image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSource {
    Camera,
    Screenshot,
    Downloaded,
    Received,
    Generated,
    Unknown,
}

impl Default for VisualContext {
    fn default() -> Self {
        Self {
            people: Vec::new(),
            location: None,
            activity: None,
            source: ImageSource::Unknown,
        }
    }
}

impl VisualMemory {
    /// Create a new visual memory
    pub fn new(
        image_path: PathBuf,
        embedding: Vec<f32>,
        description: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            image_path,
            embedding,
            description,
            context: VisualContext::default(),
            tags: Vec::new(),
            emotional_valence: 0.0,
            strength: 1.0,
            recall_count: 0,
            created_at: now,
            last_accessed: now,
            linked_memories: Vec::new(),
            linked_visuals: Vec::new(),
        }
    }
    
    /// Add context to the memory
    pub fn with_context(mut self, context: VisualContext) -> Self {
        self.context = context;
        self
    }
    
    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Set emotional valence
    pub fn with_emotion(mut self, valence: f32) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }
    
    /// Record a recall event (strengthens the memory)
    pub fn recall(&mut self) {
        self.recall_count += 1;
        self.last_accessed = Utc::now();
        // Strengthen memory on recall (like human memory)
        self.strength = (self.strength + 0.1).min(1.0);
    }
    
    /// Apply forgetting curve decay
    pub fn apply_decay(&mut self, decay_rate: f32) {
        // Emotional memories decay slower (amygdala effect)
        let emotion_factor = 1.0 - (self.emotional_valence.abs() * 0.3);
        // Frequently recalled memories decay slower
        let recall_factor = 1.0 / (1.0 + self.recall_count as f32 * 0.1);
        
        let effective_decay = decay_rate * emotion_factor * recall_factor;
        self.strength = (self.strength - effective_decay).max(0.0);
    }
    
    /// Link to a text memory
    pub fn link_memory(&mut self, memory_id: Uuid) {
        if !self.linked_memories.contains(&memory_id) {
            self.linked_memories.push(memory_id);
        }
    }
    
    /// Link to another visual memory
    pub fn link_visual(&mut self, visual_id: Uuid) {
        if !self.linked_visuals.contains(&visual_id) {
            self.linked_visuals.push(visual_id);
        }
    }
}

/// CLIP embedding provider trait
pub trait ClipProvider: Send + Sync {
    /// Generate embedding for an image
    fn embed_image(&self, image_path: &std::path::Path) -> Result<Vec<f32>, ClipError>;
    
    /// Generate embedding for text (for cross-modal search)
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, ClipError>;
    
    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;
}

/// CLIP-related errors
#[derive(Debug)]
pub enum ClipError {
    IoError(std::io::Error),
    ModelError(String),
    ImageError(String),
    EncodingError(String),
}

impl std::fmt::Display for ClipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClipError::IoError(e) => write!(f, "IO error: {}", e),
            ClipError::ModelError(s) => write!(f, "Model error: {}", s),
            ClipError::ImageError(s) => write!(f, "Image error: {}", s),
            ClipError::EncodingError(s) => write!(f, "Encoding error: {}", s),
        }
    }
}

impl std::error::Error for ClipError {}

impl From<std::io::Error> for ClipError {
    fn from(err: std::io::Error) -> Self {
        ClipError::IoError(err)
    }
}

/// Calculate cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visual_memory_creation() {
        let mem = VisualMemory::new(
            PathBuf::from("/test/image.jpg"),
            vec![0.1; 512],
            "Test image".to_string(),
        );
        
        assert_eq!(mem.description, "Test image");
        assert_eq!(mem.embedding.len(), 512);
        assert_eq!(mem.strength, 1.0);
    }
    
    #[test]
    fn test_recall_strengthens_memory() {
        let mut mem = VisualMemory::new(
            PathBuf::from("/test/image.jpg"),
            vec![0.1; 512],
            "Test".to_string(),
        );
        mem.strength = 0.5;
        
        mem.recall();
        
        assert_eq!(mem.recall_count, 1);
        assert!(mem.strength > 0.5);
    }
    
    #[test]
    fn test_emotional_memories_decay_slower() {
        let mut neutral = VisualMemory::new(
            PathBuf::from("/test/neutral.jpg"),
            vec![0.1; 512],
            "Neutral".to_string(),
        );
        
        let mut emotional = VisualMemory::new(
            PathBuf::from("/test/emotional.jpg"),
            vec![0.1; 512],
            "Emotional".to_string(),
        ).with_emotion(0.9);
        
        neutral.apply_decay(0.1);
        emotional.apply_decay(0.1);
        
        // Emotional memory should have higher remaining strength
        assert!(emotional.strength > neutral.strength);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
