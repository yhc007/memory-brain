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
pub mod inverted_index;
pub mod bloom_filter;
pub mod simd_ops;
pub mod compression;
pub mod merge;
pub mod bench;
pub mod watch;
pub mod server;
pub mod sam;
pub mod dream;
pub mod mindmap;
pub mod predict;
pub mod tui;
pub mod web_ui;

// Visual memory (image storage like human visual cortex)
pub mod visual;
pub mod clip_onnx;
pub mod visual_storage;
pub mod vlm;

// Hippocampus - memory formation, replay, episode chains, auto-importance
pub mod hippocampus;

// coredb_storage merged into storage.rs

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
pub use inverted_index::InvertedIndex;
pub use bloom_filter::{BloomFilter, CountingBloomFilter, BloomStats};
pub use simd_ops::{cosine_similarity_simd, dot_product_simd, l2_norm_simd, batch_cosine_similarity, top_k_similar};
pub use compression::{QuantizedEmbedding, CompressedF32, CompressionStats, compress_embeddings, decompress_embeddings};
pub use merge::{MemoryMerger, MergeConfig, MergeResult, analyze_duplicates, merge_duplicates};
pub use sam::{SamBrain, SamMemory, SamMemoryType, SamBrainStats};
pub use dream::{DreamEngine, DreamState, DreamPhase};
pub use mindmap::MindMap;
pub use predict::{Predictor, Prediction, ForgettingAlert, Pattern};
#[cfg(feature = "mlx")]
pub use embedding::{MlxEmbedder, create_mlx_embedder};
// CoreDBStorage is now the default Storage

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
    /// Inverted index for fast keyword search
    pub keyword_index: InvertedIndex,
    /// Bloom filter for fast "exists?" checks
    pub keyword_bloom: BloomFilter,
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
            keyword_index: InvertedIndex::new(),
            keyword_bloom: BloomFilter::new(10000, 0.01), // 10K items, 1% FPR
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

        // 5. Add to keyword index for fast search
        self.keyword_index.add(memory_item.id, input);
        
        // 6. Add keywords to bloom filter for instant "exists?" check
        for word in input.split_whitespace() {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if word.len() >= 2 {
                self.keyword_bloom.add_str(word);
            }
        }

        // 7. 🔗 Auto-link related memories!
        if let Some(ref emb) = memory_item.embedding {
            let related = self.find_related_memories(emb, 0.4, 5);
            for (related_id, similarity) in related {
                // Only link if similarity is meaningful
                if similarity > 0.4 {
                    memory_item.associate(related_id);
                }
            }
        }

        // 8. Also store to long-term immediately (for CLI usage where brain is recreated each time)
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

        // 2. Try inverted index first (O(1) lookup!) 🚀
        let indexed_ids = self.keyword_index.search_ranked(query, limit * 2);
        if !indexed_ids.is_empty() {
            // Fetch memories by IDs from semantic store
            for (id, _score) in &indexed_ids {
                if let Ok(items) = self.semantic.search("", 1000) {
                    if let Some(item) = items.into_iter().find(|i| i.id == *id) {
                        results.push(item);
                    }
                }
            }
        }

        // 3. Fallback: Extract keywords for text search (if index is empty/sparse)
        if results.len() < limit {
            let keywords: Vec<String> = query
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| w.len() > 2)
                .filter(|w| !is_stop_word(w))
                .map(|w| w.to_string())
                .collect();

            // 4. Bloom filter pre-check: skip keywords that definitely don't exist 🌸
            let keywords: Vec<String> = keywords
                .into_iter()
                .filter(|k| self.keyword_bloom.contains_str(k))
                .collect();

            // 5. Search each keyword in memories (LIKE fallback)
            for keyword in &keywords {
                if let Ok(episodic) = self.episodic.search(&keyword, limit) {
                    results.extend(episodic);
                }
                if let Ok(semantic) = self.semantic.search(&keyword, limit) {
                    results.extend(semantic);
                }
            }
        }

        // 5. Also try the full query (for exact matches)
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

    /// 🔗 Find related memories by embedding similarity
    fn find_related_memories(&self, embedding: &[f32], threshold: f32, limit: usize) -> Vec<(uuid::Uuid, f32)> {
        let mut related = Vec::new();
        
        // Search in semantic memory (main knowledge store)
        if let Ok(items) = self.semantic.search("", 100) {
            for item in items {
                if let Some(ref item_emb) = item.embedding {
                    let similarity = cosine_similarity(embedding, item_emb);
                    if similarity > threshold {
                        related.push((item.id, similarity));
                    }
                }
            }
        }
        
        // Search in episodic memory (experiences)
        if let Ok(items) = self.episodic.search("", 50) {
            for item in items {
                if let Some(ref item_emb) = item.embedding {
                    let similarity = cosine_similarity(embedding, item_emb);
                    if similarity > threshold && !related.iter().any(|(id, _)| *id == item.id) {
                        related.push((item.id, similarity));
                    }
                }
            }
        }
        
        // Sort by similarity (highest first)
        related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        related.truncate(limit);
        related
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

    /// Rebuild keyword index and bloom filter from existing memories
    /// 
    /// Call this after loading a database to populate the in-memory indexes.
    /// Update the strength of a memory by its ID (partial match)
    pub fn update_strength(&mut self, id_prefix: &str, new_strength: f32) -> Result<(), Box<dyn std::error::Error>> {
        let strength = new_strength.clamp(0.0, 1.0);
        
        // Search through all memory stores
        // Check episodic
        if let Ok(items) = self.episodic.search("", 100000) {
            for mut item in items {
                if item.id.to_string().starts_with(id_prefix) {
                    item.strength = strength;
                    let _ = self.episodic.store(item);
                    return Ok(());
                }
            }
        }
        // Check semantic
        if let Ok(items) = self.semantic.search("", 100000) {
            for mut item in items {
                if item.id.to_string().starts_with(id_prefix) {
                    item.strength = strength;
                    let _ = self.semantic.store(item);
                    return Ok(());
                }
            }
        }
        // Check procedural
        if let Ok(items) = self.procedural.search("", 100000) {
            for mut item in items {
                if item.id.to_string().starts_with(id_prefix) {
                    item.strength = strength;
                    let _ = self.procedural.store(item);
                    return Ok(());
                }
            }
        }
        
        Err(format!("Memory not found: {}", id_prefix).into())
    }

    /// Execute CQL query through the underlying CoreDB (via semantic store's storage)
    pub fn storage_execute_cql(&self, query: &str) -> Result<String, String> {
        self.semantic.execute_cql_html(query)
    }

    pub fn rebuild_indexes(&mut self) -> Result<RebuildStats, Box<dyn std::error::Error>> {
        let mut stats = RebuildStats::default();

        // Clear existing indexes
        self.keyword_index.clear();
        self.keyword_bloom.clear();

        // Load all episodic memories
        if let Ok(items) = self.episodic.search("", 100000) {
            for item in &items {
                self.keyword_index.add(item.id, &item.content);
                for word in item.content.split_whitespace() {
                    let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if word.len() >= 2 {
                        self.keyword_bloom.add_str(word);
                    }
                }
            }
            stats.episodic_count = items.len();
        }

        // Load all semantic memories
        if let Ok(items) = self.semantic.search("", 100000) {
            for item in &items {
                self.keyword_index.add(item.id, &item.content);
                for word in item.content.split_whitespace() {
                    let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if word.len() >= 2 {
                        self.keyword_bloom.add_str(word);
                    }
                }
            }
            stats.semantic_count = items.len();
        }

        // Load all procedural memories
        if let Ok(items) = self.procedural.search("", 100000) {
            for item in &items {
                self.keyword_index.add(item.id, &item.content);
                for word in item.content.split_whitespace() {
                    let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if word.len() >= 2 {
                        self.keyword_bloom.add_str(word);
                    }
                }
            }
            stats.procedural_count = items.len();
        }

        stats.index_stats = self.keyword_index.stats();
        stats.bloom_stats = self.keyword_bloom.stats();

        Ok(stats)
    }
}

/// Statistics from rebuild_indexes
#[derive(Debug, Default)]
pub struct RebuildStats {
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub procedural_count: usize,
    pub index_stats: inverted_index::IndexStats,
    pub bloom_stats: BloomStats,
}

impl std::fmt::Display for RebuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "📊 Rebuild Complete!")?;
        writeln!(f, "  Episodic:   {} memories", self.episodic_count)?;
        writeln!(f, "  Semantic:   {} memories", self.semantic_count)?;
        writeln!(f, "  Procedural: {} memories", self.procedural_count)?;
        writeln!(f, "  Total:      {} memories", self.episodic_count + self.semantic_count + self.procedural_count)?;
        writeln!(f, "")?;
        writeln!(f, "  {}", self.index_stats)?;
        write!(f, "  {}", self.bloom_stats)
    }
}
