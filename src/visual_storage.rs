//! Visual Memory Storage - CoreDB integration for image memories
//! 
//! Stores and retrieves visual memories using CoreDB, with similarity search
//! for finding related images.
//! 
//! ## VLM Integration
//! Supports automatic image description generation using Vision Language Models
//! (VLM) via Ollama. When a VLM provider is configured, descriptions can be
//! auto-generated when storing images.

use crate::visual::{ClipProvider, VisualContext, VisualMemory, cosine_similarity};
use crate::vlm::{VlmProvider, OllamaVlm};
use chrono::Utc;
use coredb::CoreDB;
use serde_json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Visual memory storage backed by CoreDB
pub struct VisualStorage {
    db: Arc<RwLock<CoreDB>>,
    clip: Arc<dyn ClipProvider>,
    keyspace: String,
    /// In-memory cache of visual memories for fast search
    cache: RwLock<HashMap<Uuid, VisualMemory>>,
    /// Optional VLM provider for auto-generating descriptions
    vlm: Option<Arc<dyn VlmProvider>>,
}

impl VisualStorage {
    /// Create a new visual storage
    pub async fn new(
        db: Arc<RwLock<CoreDB>>,
        clip: Arc<dyn ClipProvider>,
        keyspace: &str,
    ) -> Result<Self, VisualStorageError> {
        let storage = Self {
            db,
            clip,
            keyspace: keyspace.to_string(),
            cache: RwLock::new(HashMap::new()),
            vlm: None,
        };
        
        storage.init_schema().await?;
        
        Ok(storage)
    }
    
    /// Create visual storage with VLM support for auto-description
    pub async fn with_vlm(
        db: Arc<RwLock<CoreDB>>,
        clip: Arc<dyn ClipProvider>,
        keyspace: &str,
        vlm_model: &str,
    ) -> Result<Self, VisualStorageError> {
        // Check if Ollama has the model
        let vlm: Arc<dyn VlmProvider> = Arc::new(OllamaVlm::new(vlm_model));
        
        let storage = Self {
            db,
            clip,
            keyspace: keyspace.to_string(),
            cache: RwLock::new(HashMap::new()),
            vlm: Some(vlm),
        };
        
        storage.init_schema().await?;
        
        Ok(storage)
    }
    
    /// Set VLM provider
    pub fn set_vlm(&mut self, vlm: Arc<dyn VlmProvider>) {
        self.vlm = Some(vlm);
    }
    
    /// Check if VLM is available
    pub fn has_vlm(&self) -> bool {
        self.vlm.is_some()
    }
    
    /// Store image with auto-generated description using VLM
    pub async fn store_image_auto(
        &self,
        image_path: &Path,
        context: Option<VisualContext>,
        tags: Vec<String>,
        emotional_valence: f32,
        custom_prompt: Option<&str>,
    ) -> Result<VisualMemory, VisualStorageError> {
        // Generate description using VLM
        let description = match &self.vlm {
            Some(vlm) => {
                vlm.describe_image(image_path, custom_prompt)
                    .map_err(|e| VisualStorageError::VlmError(e.to_string()))?
            }
            None => {
                return Err(VisualStorageError::VlmError(
                    "VLM not configured. Use store_image() with manual description or configure VLM.".to_string()
                ));
            }
        };
        
        // Store with generated description
        self.store_image(image_path, &description, context, tags, emotional_valence).await
    }
    
    /// Initialize CoreDB schema for visual memories
    async fn init_schema(&self) -> Result<(), VisualStorageError> {
        let db = self.db.read().await;
        
        // Create keyspace if not exists
        let create_ks = format!(
            "CREATE KEYSPACE {} WITH REPLICATION = {{'class': 'SimpleStrategy', 'replication_factor': 1}}",
            self.keyspace
        );
        let _ = db.execute_cql(&create_ks).await; // Ignore if exists
        
        // Create visual_memories table
        let create_table = format!(
            "CREATE TABLE {}.visual_memories (
                id TEXT PRIMARY KEY,
                image_path TEXT,
                embedding TEXT,
                description TEXT,
                context TEXT,
                tags TEXT,
                emotional_valence TEXT,
                strength TEXT,
                recall_count TEXT,
                created_at TEXT,
                last_accessed TEXT,
                linked_memories TEXT,
                linked_visuals TEXT
            )",
            self.keyspace
        );
        let _ = db.execute_cql(&create_table).await; // Ignore if exists
        
        Ok(())
    }
    
    /// Store a new image as visual memory
    pub async fn store_image(
        &self,
        image_path: &Path,
        description: &str,
        context: Option<VisualContext>,
        tags: Vec<String>,
        emotional_valence: f32,
    ) -> Result<VisualMemory, VisualStorageError> {
        // Generate CLIP embedding
        let embedding = self.clip.embed_image(image_path)
            .map_err(|e| VisualStorageError::EmbeddingError(e.to_string()))?;
        
        // Create visual memory
        let mut memory = VisualMemory::new(
            image_path.to_path_buf(),
            embedding,
            description.to_string(),
        )
        .with_tags(tags)
        .with_emotion(emotional_valence);
        
        if let Some(ctx) = context {
            memory = memory.with_context(ctx);
        }
        
        // Store in CoreDB
        self.store_memory(&memory).await?;
        
        // Add to cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(memory.id, memory.clone());
        }
        
        // Find and link related memories
        self.auto_link_memories(&mut memory).await?;
        
        Ok(memory)
    }
    
    /// Store a visual memory in CoreDB
    async fn store_memory(&self, memory: &VisualMemory) -> Result<(), VisualStorageError> {
        let db = self.db.read().await;
        
        let embedding_json = serde_json::to_string(&memory.embedding)
            .map_err(|e| VisualStorageError::SerializationError(e.to_string()))?;
        let context_json = serde_json::to_string(&memory.context)
            .map_err(|e| VisualStorageError::SerializationError(e.to_string()))?;
        let tags_json = serde_json::to_string(&memory.tags)
            .map_err(|e| VisualStorageError::SerializationError(e.to_string()))?;
        let linked_memories_json = serde_json::to_string(&memory.linked_memories)
            .map_err(|e| VisualStorageError::SerializationError(e.to_string()))?;
        let linked_visuals_json = serde_json::to_string(&memory.linked_visuals)
            .map_err(|e| VisualStorageError::SerializationError(e.to_string()))?;
        
        let insert = format!(
            "INSERT INTO {}.visual_memories (
                id, image_path, embedding, description, context, tags,
                emotional_valence, strength, recall_count,
                created_at, last_accessed, linked_memories, linked_visuals
            ) VALUES (
                '{}', '{}', '{}', '{}', '{}', '{}',
                '{}', '{}', '{}',
                '{}', '{}', '{}', '{}'
            )",
            self.keyspace,
            memory.id,
            memory.image_path.display().to_string().replace("'", "''"),
            embedding_json.replace("'", "''"),
            memory.description.replace("'", "''"),
            context_json.replace("'", "''"),
            tags_json.replace("'", "''"),
            memory.emotional_valence,
            memory.strength,
            memory.recall_count,
            memory.created_at.to_rfc3339(),
            memory.last_accessed.to_rfc3339(),
            linked_memories_json.replace("'", "''"),
            linked_visuals_json.replace("'", "''"),
        );
        
        db.execute_cql(&insert).await
            .map_err(|e| VisualStorageError::DatabaseError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Search for similar images by text query
    pub async fn search_by_text(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(VisualMemory, f32)>, VisualStorageError> {
        // Generate text embedding
        let query_embedding = self.clip.embed_text(query)
            .map_err(|e| VisualStorageError::EmbeddingError(e.to_string()))?;
        
        self.search_by_embedding(&query_embedding, limit).await
    }
    
    /// Search for similar images by image
    pub async fn search_by_image(
        &self,
        image_path: &Path,
        limit: usize,
    ) -> Result<Vec<(VisualMemory, f32)>, VisualStorageError> {
        // Generate image embedding
        let query_embedding = self.clip.embed_image(image_path)
            .map_err(|e| VisualStorageError::EmbeddingError(e.to_string()))?;
        
        self.search_by_embedding(&query_embedding, limit).await
    }
    
    /// Search by embedding vector
    async fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(VisualMemory, f32)>, VisualStorageError> {
        // Use cache for fast search
        let cache = self.cache.read().await;
        
        let mut scored: Vec<(VisualMemory, f32)> = cache.values()
            .map(|memory| {
                let similarity = cosine_similarity(query_embedding, &memory.embedding);
                (memory.clone(), similarity)
            })
            .collect();
        
        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        
        Ok(scored)
    }
    
    /// Get a specific visual memory by ID
    pub async fn get(&self, id: Uuid) -> Result<Option<VisualMemory>, VisualStorageError> {
        let cache = self.cache.read().await;
        Ok(cache.get(&id).cloned())
    }
    
    /// Record a recall event
    pub async fn record_recall(&self, id: Uuid) -> Result<(), VisualStorageError> {
        let mut cache = self.cache.write().await;
        if let Some(memory) = cache.get_mut(&id) {
            memory.recall();
            // Also update in database
            drop(cache);
            if let Some(mem) = self.get(id).await? {
                self.store_memory(&mem).await?;
            }
        }
        Ok(())
    }
    
    /// Automatically link to similar visual memories
    async fn auto_link_memories(&self, memory: &mut VisualMemory) -> Result<(), VisualStorageError> {
        // Find similar visual memories
        let similar = self.search_by_embedding(&memory.embedding, 5).await?;
        
        for (other, similarity) in similar {
            if other.id != memory.id && similarity > 0.7 {
                memory.link_visual(other.id);
            }
        }
        
        // Update links in storage
        if !memory.linked_visuals.is_empty() {
            self.store_memory(memory).await?;
            
            // Update cache
            let mut cache = self.cache.write().await;
            cache.insert(memory.id, memory.clone());
        }
        
        Ok(())
    }
    
    /// Load all visual memories into cache
    pub async fn load_cache(&self) -> Result<usize, VisualStorageError> {
        let db = self.db.read().await;
        
        let select = format!(
            "SELECT * FROM {}.visual_memories",
            self.keyspace
        );
        
        let result = db.execute_cql(&select).await
            .map_err(|e| VisualStorageError::DatabaseError(e.to_string()))?;
        
        let mut cache = self.cache.write().await;
        let mut count = 0;
        
        if let coredb::QueryResult::Rows(rows) = result {
            for row in rows {
                if let Some(memory) = parse_visual_memory_row(&row) {
                    cache.insert(memory.id, memory);
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
    
    /// Apply forgetting curve to all visual memories
    pub async fn apply_forgetting(&self, decay_rate: f32) -> Result<usize, VisualStorageError> {
        let mut cache = self.cache.write().await;
        let mut updated = 0;
        
        for memory in cache.values_mut() {
            let old_strength = memory.strength;
            memory.apply_decay(decay_rate);
            
            if (old_strength - memory.strength).abs() > 0.001 {
                updated += 1;
            }
        }
        
        Ok(updated)
    }
    
    /// Get statistics about visual memories
    pub async fn stats(&self) -> Result<VisualStats, VisualStorageError> {
        let cache = self.cache.read().await;
        
        Ok(VisualStats {
            total_memories: cache.len(),
            embedding_dim: self.clip.embedding_dim(),
        })
    }
}

/// Visual memory statistics
#[derive(Debug)]
pub struct VisualStats {
    pub total_memories: usize,
    pub embedding_dim: usize,
}

/// Visual storage errors
#[derive(Debug)]
pub enum VisualStorageError {
    DatabaseError(String),
    EmbeddingError(String),
    SerializationError(String),
    NotFound(String),
    VlmError(String),
}

impl std::fmt::Display for VisualStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DatabaseError(s) => write!(f, "Database error: {}", s),
            Self::EmbeddingError(s) => write!(f, "Embedding error: {}", s),
            Self::SerializationError(s) => write!(f, "Serialization error: {}", s),
            Self::NotFound(s) => write!(f, "Not found: {}", s),
            Self::VlmError(s) => write!(f, "VLM error: {}", s),
        }
    }
}

impl std::error::Error for VisualStorageError {}

// Helper function to parse a row into VisualMemory
fn parse_visual_memory_row(row: &coredb::query::Row) -> Option<VisualMemory> {
    use coredb::CassandraValue;
    
    let get_text = |row: &coredb::query::Row, key: &str| -> Option<String> {
        row.columns.get(key).and_then(|v| {
            if let CassandraValue::Text(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
    };
    
    let id: Uuid = get_text(row, "id")?.parse().ok()?;
    let image_path = get_text(row, "image_path")?.into();
    let embedding: Vec<f32> = serde_json::from_str(&get_text(row, "embedding")?).ok()?;
    let description = get_text(row, "description")?;
    let context: VisualContext = serde_json::from_str(&get_text(row, "context")?).ok()?;
    let tags: Vec<String> = serde_json::from_str(&get_text(row, "tags")?).ok()?;
    let emotional_valence: f32 = get_text(row, "emotional_valence")?.parse().ok()?;
    let strength: f32 = get_text(row, "strength")?.parse().ok()?;
    let recall_count: u32 = get_text(row, "recall_count")?.parse().ok()?;
    let created_at = chrono::DateTime::parse_from_rfc3339(&get_text(row, "created_at")?)
        .ok()?.with_timezone(&Utc);
    let last_accessed = chrono::DateTime::parse_from_rfc3339(&get_text(row, "last_accessed")?)
        .ok()?.with_timezone(&Utc);
    let linked_memories: Vec<Uuid> = serde_json::from_str(&get_text(row, "linked_memories")?).ok()?;
    let linked_visuals: Vec<Uuid> = serde_json::from_str(&get_text(row, "linked_visuals")?).ok()?;
    
    Some(VisualMemory {
        id,
        image_path,
        embedding,
        description,
        context,
        tags,
        emotional_valence,
        strength,
        recall_count,
        created_at,
        last_accessed,
        linked_memories,
        linked_visuals,
    })
}
