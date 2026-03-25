//! NeocortexActor - Slow Semantic Memory
//!
//! Implements the neocortex role in CLS:
//! - Slow consolidation of semantic knowledge
//! - Pattern extraction and generalization
//! - Low plasticity, robust representations
//!
//! ## Phase 3: CoreVecDB Integration
//! 
//! Concepts are stored in CoreVecDB for semantic search across knowledge.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::messages::*;
use crate::embedding::{EmbeddingClient, HashEmbedder, EMBEDDING_DIM};
// CoreVecDB storage imported from storage module

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
    /// Embedding vector (optional)
    pub embedding: Option<Vec<f32>>,
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
            embedding: None,
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
    /// CoreVecDB URL (None = in-memory only)
    pub vecdb_url: Option<String>,
    /// Embedding server URL (None = use hash embedder)
    pub embedding_url: Option<String>,
    /// Collection name for concepts
    pub collection: String,
}

impl Default for NeocortexConfig {
    fn default() -> Self {
        Self {
            association_threshold: 0.3,
            consolidation_rate: 0.05,
            vecdb_url: None,
            embedding_url: None,
            collection: "concepts".to_string(),
        }
    }
}

impl NeocortexConfig {
    /// Create config with external backends enabled
    pub fn with_backends() -> Self {
        Self {
            vecdb_url: Some("http://localhost:3100".to_string()),
            embedding_url: Some("http://localhost:3201".to_string()),
            ..Default::default()
        }
    }
}

/// CoreVecDB storage for concepts
struct ConceptStorage {
    base_url: String,
    collection: String,
}

impl ConceptStorage {
    fn new(base_url: &str, collection: &str) -> Result<Self, String> {
        let storage = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            collection: collection.to_string(),
        };
        storage.ensure_collection()?;
        Ok(storage)
    }

    fn ensure_collection(&self) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        match ureq::get(&url).call() {
            Ok(resp) if resp.status() == 200 => Ok(()),
            _ => {
                let req = serde_json::json!({
                    "name": self.collection,
                    "dim": EMBEDDING_DIM,
                    "distance": "cosine",
                    "indexed_fields": ["name"],
                    "numeric_fields": ["consolidation_level", "created_at"]
                });

                match ureq::post(&format!("{}/collections", self.base_url)).send_json(&req) {
                    Ok(_) => {
                        info!("Created concepts collection: {}", self.collection);
                        Ok(())
                    }
                    Err(ureq::Error::Status(409, _)) => Ok(()),
                    Err(e) => Err(format!("Failed to create collection: {}", e)),
                }
            }
        }
    }

    fn store(&self, concept: &Concept, embedding: &[f32]) -> Result<u64, String> {
        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), concept.id.clone());
        metadata.insert("name".to_string(), concept.name.clone());
        metadata.insert("description".to_string(), concept.description.clone());
        metadata.insert("consolidation_level".to_string(), concept.consolidation_level.to_string());
        metadata.insert("created_at".to_string(), concept.created_at.timestamp_millis().to_string());

        // Store source memories as comma-separated
        if !concept.source_memories.is_empty() {
            let sources: Vec<String> = concept.source_memories.iter().map(|id| id.to_string()).collect();
            metadata.insert("source_memories".to_string(), sources.join(","));
        }

        let req = serde_json::json!({
            "vectors": [{
                "vector": embedding,
                "metadata": metadata
            }]
        });

        let url = format!("{}/collections/{}/upsert_batch", self.base_url, self.collection);
        let resp = ureq::post(&url)
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() == 200 {
            #[derive(serde::Deserialize)]
            struct UpsertResp { start_id: u64 }
            let result: UpsertResp = resp.into_json().map_err(|e| format!("JSON error: {}", e))?;
            debug!("Stored concept {} -> vec_id {}", concept.name, result.start_id);
            Ok(result.start_id)
        } else {
            Err(format!("Store failed: status {}", resp.status()))
        }
    }

    fn search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        let req = serde_json::json!({
            "vector": query_embedding,
            "k": k,
            "include_metadata": true
        });

        let url = format!("{}/collections/{}/search", self.base_url, self.collection);
        let resp = ureq::post(&url)
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() != 200 {
            return Err(format!("Search failed: status {}", resp.status()));
        }

        #[derive(serde::Deserialize)]
        struct SearchResult {
            score: f32,
            metadata: Option<HashMap<String, String>>,
        }
        #[derive(serde::Deserialize)]
        struct SearchResp {
            results: Vec<SearchResult>,
        }

        let search_resp: SearchResp = resp.into_json().map_err(|e| format!("JSON error: {}", e))?;

        let results = search_resp.results
            .into_iter()
            .filter_map(|r| {
                r.metadata.and_then(|m| {
                    m.get("name").map(|name| (name.clone(), r.score))
                })
            })
            .collect();

        Ok(results)
    }
}

/// Embedding backend
enum EmbedderBackend {
    Http(EmbeddingClient),
    Hash(HashEmbedder),
}

impl EmbedderBackend {
    fn embed(&self, text: &str) -> Vec<f32> {
        match self {
            EmbedderBackend::Http(client) => {
                client.embed(text).unwrap_or_else(|e| {
                    warn!("HTTP embedding failed: {}", e);
                    HashEmbedder::new(EMBEDDING_DIM).embed(text)
                })
            }
            EmbedderBackend::Hash(hasher) => hasher.embed(text),
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
    /// Vector storage backend (optional)
    storage: Option<ConceptStorage>,
    /// Embedding backend
    embedder: EmbedderBackend,
}

impl NeocortexActor {
    pub fn new(config: NeocortexConfig) -> Self {
        // Initialize embedding backend
        let embedder = if let Some(ref url) = config.embedding_url {
            let client = EmbeddingClient::new(url);
            if client.health_check() {
                info!("✅ Neocortex connected to embedding server: {}", url);
                EmbedderBackend::Http(client)
            } else {
                warn!("⚠️ Embedding server not available for neocortex");
                EmbedderBackend::Hash(HashEmbedder::new(EMBEDDING_DIM))
            }
        } else {
            EmbedderBackend::Hash(HashEmbedder::new(128))
        };

        // Initialize concept storage
        let storage = if let Some(ref url) = config.vecdb_url {
            match ConceptStorage::new(url, &config.collection) {
                Ok(s) => {
                    info!("✅ Neocortex connected to CoreVecDB: {}", url);
                    Some(s)
                }
                Err(e) => {
                    warn!("⚠️ CoreVecDB not available for neocortex: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            concepts: HashMap::new(),
            name_index: HashMap::new(),
            storage,
            embedder,
        }
    }

    /// Create with external backends (convenience)
    pub fn with_backends() -> Self {
        Self::new(NeocortexConfig::with_backends())
    }

    /// Find associations between memories based on content similarity
    pub fn associate(&self, memories: &[Memory]) -> Vec<(MemoryId, MemoryId, f32)> {
        let mut associations = Vec::new();
        
        // Use embeddings if available
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                let sim = if let (Some(emb_a), Some(emb_b)) = 
                    (&memories[i].embedding, &memories[j].embedding) 
                {
                    EmbeddingClient::cosine_similarity(emb_a, emb_b)
                } else {
                    self.compute_word_similarity(&memories[i], &memories[j])
                };
                
                if sim >= self.config.association_threshold {
                    associations.push((memories[i].id, memories[j].id, sim));
                }
            }
        }
        
        debug!("Found {} associations among {} memories", 
               associations.len(), memories.len());
        associations
    }

    /// Compute word-based similarity (fallback)
    fn compute_word_similarity(&self, a: &Memory, b: &Memory) -> f32 {
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
        
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    /// Extract patterns from a set of memories
    pub fn extract_patterns(&self, memories: &[Memory]) -> Vec<String> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        for memory in memories {
            let words: std::collections::HashSet<_> = memory.content
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|s| s.to_string())
                .collect();
            
            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }
        
        let threshold = (memories.len() / 2).max(2);
        let mut patterns: Vec<_> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .collect();
        
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        
        patterns.into_iter().take(10).map(|(word, _)| word).collect()
    }

    /// Generalize from specific memories to a concept
    pub fn generalize(&mut self, memories: &[Memory]) -> Option<String> {
        if memories.is_empty() {
            return None;
        }
        
        let patterns = self.extract_patterns(memories);
        if patterns.is_empty() {
            return None;
        }
        
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
        let mut concept = Concept::new(concept_name.clone(), description.clone(), source_ids);
        
        // Generate embedding for the concept
        let embedding = self.embedder.embed(&description);
        concept.embedding = Some(embedding.clone());
        
        // Store in CoreVecDB if available
        if let Some(ref storage) = self.storage {
            if let Err(e) = storage.store(&concept, &embedding) {
                warn!("Failed to store concept in CoreVecDB: {}", e);
            }
        }
        
        let id = concept.id.clone();
        self.name_index.insert(concept_name.clone(), id.clone());
        self.concepts.insert(id, concept);
        
        info!("Generalized concept: {}", concept_name);
        Some(concept_name)
    }

    /// Query for semantic knowledge
    pub fn query(&self, concept_name: &str) -> Option<&Concept> {
        self.name_index
            .get(concept_name)
            .and_then(|id| self.concepts.get(id))
    }

    /// Semantic search for related concepts
    pub fn search_concepts(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if let Some(ref storage) = self.storage {
            let query_embedding = self.embedder.embed(query);
            match storage.search(&query_embedding, k) {
                Ok(results) => return results,
                Err(e) => warn!("Concept search failed: {}", e),
            }
        }
        
        // Fallback: simple name matching
        self.concepts.values()
            .filter(|c| c.name.contains(query) || c.description.contains(query))
            .take(k)
            .map(|c| (c.name.clone(), 0.5))
            .collect()
    }

    /// Store consolidated knowledge
    pub fn store_knowledge(
        &mut self,
        concept_name: String,
        description: String,
        source_memories: Vec<MemoryId>,
    ) -> String {
        let mut concept = Concept::new(concept_name.clone(), description.clone(), source_memories);
        
        // Generate embedding
        let embedding = self.embedder.embed(&description);
        concept.embedding = Some(embedding.clone());
        
        // Store in CoreVecDB
        if let Some(ref storage) = self.storage {
            if let Err(e) = storage.store(&concept, &embedding) {
                warn!("Failed to store knowledge in CoreVecDB: {}", e);
            }
        }
        
        let id = concept.id.clone();
        self.name_index.insert(concept_name.clone(), id.clone());
        self.concepts.insert(id.clone(), concept);
        
        info!("Stored knowledge: {}", concept_name);
        id
    }

    /// Strengthen a concept (called during consolidation)
    pub fn strengthen(&mut self, concept_id: &str, delta: f32) {
        if let Some(concept) = self.concepts.get_mut(concept_id) {
            concept.consolidation_level = (concept.consolidation_level + delta).min(1.0);
            concept.updated_at = Utc::now();
        }
    }

    /// Add relation between concepts
    pub fn add_relation(&mut self, from_id: &str, to_id: &str, strength: f32) {
        if let Some(concept) = self.concepts.get_mut(from_id) {
            if let Some(rel) = concept.relations.iter_mut().find(|(id, _)| id == to_id) {
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

    /// Check if external backends are connected
    pub fn has_backends(&self) -> bool {
        self.storage.is_some()
    }

    /// Process a message and return response
    pub fn handle(&mut self, msg: NeocortexMessage, memories: &[Memory]) -> NeocortexResponse {
        match msg {
            NeocortexMessage::Associate { memory_ids: _ } => {
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
        assert!(!associations.is_empty());
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
        
        let concept_name = concept.unwrap();
        let queried = actor.query(&concept_name);
        assert!(queried.is_some());
        
        // Check embedding was generated
        assert!(queried.unwrap().embedding.is_some());
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
