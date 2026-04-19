//! HippocampusActor - Fast Episodic Memory
//!
//! Implements the hippocampus role in CLS:
//! - Fast storage of new experiences
//! - High plasticity
//! - Episodic memory with temporal context
//!
//! ## Phase 2: External Backends
//! 
//! Now supports:
//! - CoreVecDB for vector storage (localhost:3100)
//! - BGE-M3 for embeddings (localhost:3201)

use std::collections::{HashMap, VecDeque};
use chrono::Utc;
use tracing::{info, debug, warn};

use crate::messages::*;
use crate::embedding::{EmbeddingClient, HashEmbedder, EMBEDDING_DIM};
use crate::storage::VecDbStorage;

/// Configuration for HippocampusActor
#[derive(Debug, Clone)]
pub struct HippocampusConfig {
    /// Maximum size of working memory (recent memories)
    pub working_memory_size: usize,
    /// Default decay rate for forgetting
    pub default_decay_rate: f32,
    /// Minimum strength before memory is forgotten
    pub min_strength: f32,
    /// CoreVecDB URL (None = in-memory only)
    pub vecdb_url: Option<String>,
    /// Embedding server URL (None = use hash embedder)
    pub embedding_url: Option<String>,
    /// Collection name for CoreVecDB
    pub collection: String,
}

impl Default for HippocampusConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 100,
            default_decay_rate: 0.1,
            min_strength: 0.01,
            vecdb_url: None,
            embedding_url: None,
            collection: "memory_actor".to_string(),
        }
    }
}

impl HippocampusConfig {
    /// Create config with external backends enabled
    pub fn with_backends() -> Self {
        Self {
            vecdb_url: Some("http://localhost:3100".to_string()),
            embedding_url: Some("http://localhost:3201".to_string()),
            ..Default::default()
        }
    }
}

/// Embedding backend (HTTP or Hash fallback)
enum EmbedderBackend {
    Http(EmbeddingClient),
    Hash(HashEmbedder),
}

impl EmbedderBackend {
    fn embed(&self, text: &str) -> Vec<f32> {
        match self {
            EmbedderBackend::Http(client) => {
                client.embed(text).unwrap_or_else(|e| {
                    warn!("HTTP embedding failed: {}, using hash fallback", e);
                    HashEmbedder::new(EMBEDDING_DIM).embed(text)
                })
            }
            EmbedderBackend::Hash(hasher) => hasher.embed(text),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            EmbedderBackend::Http(_) => EMBEDDING_DIM,
            EmbedderBackend::Hash(h) => h.dimension,
        }
    }
}

/// HippocampusActor - manages fast episodic memory
pub struct HippocampusActor {
    config: HippocampusConfig,
    /// In-memory storage (always available)
    memories: HashMap<MemoryId, Memory>,
    /// Recent memories (working memory) - FIFO queue
    working_memory: VecDeque<MemoryId>,
    /// Vector storage backend (optional)
    vecdb: Option<VecDbStorage>,
    /// Embedding backend
    embedder: EmbedderBackend,
}

impl HippocampusActor {
    /// Create a new HippocampusActor
    pub fn new(config: HippocampusConfig) -> Self {
        // Initialize embedding backend
        let embedder = if let Some(ref url) = config.embedding_url {
            let client = EmbeddingClient::new(url);
            if client.health_check() {
                info!("✅ Connected to embedding server: {}", url);
                EmbedderBackend::Http(client)
            } else {
                warn!("⚠️ Embedding server not available, using hash fallback");
                EmbedderBackend::Hash(HashEmbedder::new(EMBEDDING_DIM))
            }
        } else {
            EmbedderBackend::Hash(HashEmbedder::new(128))
        };

        // Initialize vector storage backend
        let vecdb = if let Some(ref url) = config.vecdb_url {
            match VecDbStorage::new(url, &config.collection) {
                Ok(storage) => {
                    info!("✅ Connected to CoreVecDB: {}", url);
                    Some(storage)
                }
                Err(e) => {
                    warn!("⚠️ CoreVecDB not available: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            memories: HashMap::new(),
            working_memory: VecDeque::new(),
            vecdb,
            embedder,
        }
    }

    /// Create with external backends (convenience)
    pub fn with_backends() -> Self {
        Self::new(HippocampusConfig::with_backends())
    }

    /// Store a new memory
    pub fn store(&mut self, content: String, context: MemoryContext) -> MemoryId {
        // Generate embedding
        let embedding = self.embedder.embed(&content);
        
        let mut memory = Memory::new(content, context);
        memory.embedding = Some(embedding.clone());
        let id = memory.id;
        
        // Store in CoreVecDB if available
        if let Some(ref vecdb) = self.vecdb {
            match vecdb.store(&memory, &embedding) {
                Ok(vec_id) => debug!("Stored in CoreVecDB: vec_id={}", vec_id),
                Err(e) => warn!("Failed to store in CoreVecDB: {}", e),
            }
        }
        
        // Always store in-memory for fast access
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
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        // Generate query embedding
        let query_embedding = self.embedder.embed(query);
        
        // Try CoreVecDB first (proper vector search)
        if let Some(ref vecdb) = self.vecdb {
            match vecdb.search(&query_embedding, k) {
                Ok(results) => {
                    let recall_results: Vec<RecallResult> = results
                        .into_iter()
                        .map(|r| {
                            // Update access info in local cache
                            if let Some(mem) = self.memories.get_mut(&r.memory.id) {
                                mem.access_count += 1;
                                mem.last_accessed = Utc::now();
                                mem.strength = (mem.strength + 0.1).min(1.0);
                            }
                            
                            RecallResult {
                                memory: r.memory,
                                similarity: r.score,
                            }
                        })
                        .collect();
                    
                    debug!("CoreVecDB returned {} results for: {}", recall_results.len(), query);
                    return recall_results;
                }
                Err(e) => {
                    warn!("CoreVecDB search failed: {}, falling back to in-memory", e);
                }
            }
        }
        
        // Fallback: in-memory search with cosine similarity
        self.recall_inmemory(query, &query_embedding, k)
    }

    /// In-memory recall using cosine similarity
    fn recall_inmemory(&mut self, query: &str, query_embedding: &[f32], k: usize) -> Vec<RecallResult> {
        let mut results: Vec<RecallResult> = self.memories
            .values_mut()
            .filter_map(|memory| {
                let similarity = if let Some(ref mem_emb) = memory.embedding {
                    EmbeddingClient::cosine_similarity(query_embedding, mem_emb)
                } else {
                    // Keyword fallback
                    let query_lower = query.to_lowercase();
                    let content_lower = memory.content.to_lowercase();
                    if content_lower.contains(&query_lower) { 0.5 } else { 0.0 }
                };
                
                if similarity > 0.1 {
                    // Update access info
                    memory.access_count += 1;
                    memory.last_accessed = Utc::now();
                    memory.strength = (memory.strength + 0.1).min(1.0);
                    
                    Some(RecallResult {
                        memory: memory.clone(),
                        similarity,
                    })
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity * strength
        results.sort_by(|a, b| {
            let score_a = a.similarity * a.memory.strength;
            let score_b = b.similarity * b.memory.strength;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        results.truncate(k);
        debug!("In-memory recalled {} memories for query: {}", results.len(), query);
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

    /// Restore a memory from disk (skips embedding generation).
    pub fn restore(&mut self, memory: Memory) {
        let id = memory.id;
        self.memories.insert(id, memory);
        self.working_memory.push_back(id);
        if self.working_memory.len() > self.config.working_memory_size {
            self.working_memory.pop_front();
        }
    }

    /// Get memory count
    pub fn count(&self) -> usize {
        self.memories.len()
    }

    /// Check if external backends are connected
    pub fn has_backends(&self) -> bool {
        self.vecdb.is_some()
    }

    /// Get backend status
    pub fn backend_status(&self) -> BackendStatus {
        BackendStatus {
            vecdb_connected: self.vecdb.as_ref().map(|v| v.health_check()).unwrap_or(false),
            embedding_http: matches!(self.embedder, EmbedderBackend::Http(_)),
            embedding_dim: self.embedder.dimension(),
        }
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

/// Backend connection status
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub vecdb_connected: bool,
    pub embedding_http: bool,
    pub embedding_dim: usize,
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
        
        // Check embedding was generated
        let memory = actor.get(&id).unwrap();
        assert!(memory.embedding.is_some());
        
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

    #[test]
    fn test_backend_status() {
        let actor = HippocampusActor::new(HippocampusConfig::default());
        let status = actor.backend_status();
        
        // Default config = no external backends
        assert!(!status.vecdb_connected);
        assert!(!status.embedding_http);
    }
}
