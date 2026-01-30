//! # Memory Brain
//! 
//! Human brain-inspired memory system for LLMs.
//! 
//! ## Architecture
//! - **Working Memory**: Short-term, limited capacity (~7 items)
//! - **Episodic Memory**: "When did what happen" - autobiographical
//! - **Semantic Memory**: Facts and concepts
//! - **Procedural Memory**: Patterns and habits (code patterns)
//! 
//! ## Brain-like Features
//! - Memory consolidation (short-term → long-term)
//! - Forgetting curve (unused memories fade)
//! - Associative recall (related memories activate together)

pub mod working;
pub mod episodic;
pub mod semantic;
pub mod procedural;
pub mod consolidate;
pub mod forgetting;
pub mod types;
pub mod storage;
pub mod embedding;
pub mod glove;
pub mod llm;
pub mod audit;
pub mod cache;
pub mod hnsw_index;
pub mod server;
pub mod sam;
pub mod dream;
pub mod mindmap;

#[cfg(feature = "coredb-backend")]
pub mod coredb_storage;

pub use types::*;
pub use working::WorkingMemory;
pub use episodic::EpisodicMemory;
pub use semantic::SemanticMemory;
pub use procedural::ProceduralMemory;
pub use consolidate::Consolidator;
pub use forgetting::ForgettingCurve;
pub use embedding::{Embedder, HashEmbedder, TfIdfEmbedder, cosine_similarity};
pub use glove::GloVeEmbedder;
pub use llm::{LlmProvider, OllamaProvider, OpenAIProvider, MlxLmProvider, EchoProvider, MemoryChat, auto_detect_provider};
pub use cache::{CachedEmbedder, CacheStats, BatchProcessor};
pub use hnsw_index::{HnswIndex, IndexStats};
pub use sam::{SamBrain, SamMemory, SamMemoryType, SamBrainStats};
pub use dream::{DreamEngine, DreamState, DreamPhase};
pub use mindmap::MindMap;
#[cfg(feature = "mlx")]
pub use embedding::{MlxEmbedder, create_mlx_embedder};
#[cfg(feature = "coredb-backend")]
pub use coredb_storage::CoreDBStorage;

use std::sync::Arc;

/// Check if a word is a stop word (common words to skip in search)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "what",
        "when", "where", "which", "who", "whom", "this", "that", "these",
        "those", "with", "from", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "like", "know", "think", "want", "tell", "your", "you", "for",
    ];
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

/// The unified brain - coordinates all memory systems
pub struct Brain {
    pub working: WorkingMemory,
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
    pub procedural: ProceduralMemory,
    consolidator: Consolidator,
    forgetting: ForgettingCurve,
    embedder: Arc<dyn Embedder>,
}

impl Brain {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Use CachedEmbedder wrapping HashEmbedder for better performance
        let inner = HashEmbedder::new(256);
        let embedder = Arc::new(CachedEmbedder::with_default_cache(inner));
        Self::with_embedder(db_path, embedder)
    }

    pub fn with_embedder(db_path: &str, embedder: Arc<dyn Embedder>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            working: WorkingMemory::new(7), // Miller's magic number
            episodic: EpisodicMemory::new(db_path)?,
            semantic: SemanticMemory::new(db_path)?,
            procedural: ProceduralMemory::new(db_path)?,
            consolidator: Consolidator::new(),
            forgetting: ForgettingCurve::new(),
            embedder,
        })
    }

    /// Get the current embedder
    pub fn embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }

    /// Process new input and update memories
    pub fn process(&mut self, input: &str, context: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Generate embedding for the input
        let embedding = self.embedder.embed(input);
        
        // 2. Create memory item with embedding
        let mut memory_item = MemoryItem::new(input, context);
        memory_item.embedding = Some(embedding);

        // 3. Classify memory type before consolidation
        memory_item.memory_type = self.consolidator.classify(&memory_item);

        // 4. Add to working memory
        self.working.push(memory_item.clone());

        // 5. Also store to long-term immediately (for CLI usage where brain is recreated each time)
        self.consolidate_memory(memory_item)?;

        Ok(())
    }

    /// Recall relevant memories for a query
    pub fn recall(&mut self, query: &str, limit: usize) -> Vec<MemoryItem> {
        let mut results = Vec::new();

        // Generate query embedding for semantic search
        let query_embedding = self.embedder.embed(query);

        // 1. Check working memory first (fastest)
        results.extend(self.working.search(query));

        // 2. Extract keywords from query for text search
        let keywords: Vec<String> = query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 2)
            .filter(|w| !is_stop_word(w))
            .map(|w| w.to_string())
            .collect();

        // 3. Search each keyword in memories
        for keyword in &keywords {
            if let Ok(episodic) = self.episodic.search(&keyword, limit) {
                results.extend(episodic);
            }
            if let Ok(semantic) = self.semantic.search(&keyword, limit) {
                results.extend(semantic);
            }
        }

        // 4. Also try the full query (for exact matches)
        if let Ok(semantic) = self.semantic.search(query, limit) {
            results.extend(semantic);
        }

        // 4. Re-rank by embedding similarity
        for item in results.iter_mut() {
            if let Some(ref emb) = item.embedding {
                let sim = cosine_similarity(&query_embedding, emb);
                // Boost strength by similarity (temporary for sorting)
                item.strength = item.strength * 0.5 + sim * 0.5;
            }
        }

        // 5. Apply forgetting curve (boost recently accessed)
        self.forgetting.apply_decay(&mut results);

        // 6. Sort by relevance and recency
        results.sort_by(|a, b| b.relevance_score().partial_cmp(&a.relevance_score()).unwrap());
        
        // 7. Deduplicate by content
        let mut seen = std::collections::HashSet::new();
        results.retain(|item| seen.insert(item.content.clone()));
        
        results.truncate(limit);

        results
    }

    /// Semantic search using embeddings only
    pub fn semantic_search(&self, query: &str, limit: usize) -> Vec<(MemoryItem, f32)> {
        let query_embedding = self.embedder.embed(query);
        let mut results: Vec<(MemoryItem, f32)> = Vec::new();

        // Search all memory stores
        if let Ok(items) = self.semantic.search("", 1000) {
            for item in items {
                if let Some(ref emb) = item.embedding {
                    let similarity = cosine_similarity(&query_embedding, emb);
                    if similarity > 0.05 {
                        results.push((item, similarity));
                    }
                }
            }
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        results
    }

    /// Consolidate memory from working to long-term
    fn consolidate_memory(&mut self, item: MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        match item.memory_type {
            MemoryType::Episodic => self.episodic.store(item)?,
            MemoryType::Semantic => self.semantic.store(item)?,
            MemoryType::Procedural => self.procedural.store(item)?,
            MemoryType::Working => {} // Stay in working memory
        }
        Ok(())
    }

    /// Sleep phase - consolidate and clean up memories
    pub fn sleep(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Move important working memories to long-term
        let important = self.working.get_important();
        for item in important {
            self.consolidate_memory(item)?;
        }

        // 2. Apply forgetting to old memories
        self.episodic.apply_forgetting(&self.forgetting)?;
        self.semantic.apply_forgetting(&self.forgetting)?;

        // 3. Clear working memory
        self.working.clear();

        Ok(())
    }
}
