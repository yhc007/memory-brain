//! DreamActor - Background Consolidation
//!
//! Implements the sleep/dream role in CLS:
//! - Offline memory consolidation
//! - Replay of important memories
//! - Transfer from hippocampus to neocortex
//! - Pruning of weak memories
//!
//! ## Phase 4: Dream Journal
//! 
//! Consolidation insights are optionally stored in CoreVecDB for analysis.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::messages::*;
use crate::hippocampus::HippocampusActor;
use crate::neocortex::NeocortexActor;
use crate::embedding::{EmbeddingClient, HashEmbedder, EMBEDDING_DIM};

/// Configuration for DreamActor
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Number of memories to process per consolidation cycle
    pub batch_size: usize,
    /// Strength threshold for replay (memories above this get replayed)
    pub replay_threshold: f32,
    /// Strength threshold for pruning (memories below this get removed)
    pub prune_threshold: f32,
    /// How much to reinforce during replay
    pub replay_reinforcement: f32,
    /// Decay rate during dream
    pub dream_decay_rate: f32,
    /// CoreVecDB URL for dream journal (optional)
    pub vecdb_url: Option<String>,
    /// Embedding server URL (optional)
    pub embedding_url: Option<String>,
    /// Collection name for dream journal
    pub collection: String,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            replay_threshold: 0.5,
            prune_threshold: 0.05,
            replay_reinforcement: 0.1,
            dream_decay_rate: 0.02,
            vecdb_url: None,
            embedding_url: None,
            collection: "dream_journal".to_string(),
        }
    }
}

impl DreamConfig {
    /// Create config with external backends enabled
    pub fn with_backends() -> Self {
        Self {
            vecdb_url: Some("http://localhost:3100".to_string()),
            embedding_url: Some("http://localhost:3201".to_string()),
            ..Default::default()
        }
    }
}

/// Statistics from a consolidation run
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    pub memories_processed: usize,
    pub memories_replayed: usize,
    pub memories_pruned: usize,
    pub concepts_created: usize,
    pub associations_found: usize,
    pub timestamp: Option<DateTime<Utc>>,
    pub insights: Vec<String>,
}

/// Dream Journal storage
struct DreamJournal {
    base_url: String,
    collection: String,
}

impl DreamJournal {
    fn new(base_url: &str, collection: &str) -> Result<Self, String> {
        let journal = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            collection: collection.to_string(),
        };
        journal.ensure_collection()?;
        Ok(journal)
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
                    "indexed_fields": [],
                    "numeric_fields": ["timestamp", "memories_processed", "concepts_created"]
                });

                match ureq::post(&format!("{}/collections", self.base_url)).send_json(&req) {
                    Ok(_) => {
                        info!("Created dream journal collection: {}", self.collection);
                        Ok(())
                    }
                    Err(ureq::Error::Status(409, _)) => Ok(()),
                    Err(e) => Err(format!("Failed to create collection: {}", e)),
                }
            }
        }
    }

    fn record(&self, stats: &ConsolidationStats, embedding: &[f32]) -> Result<(), String> {
        let mut metadata = HashMap::new();
        
        let timestamp = stats.timestamp.unwrap_or_else(Utc::now);
        metadata.insert("timestamp".to_string(), timestamp.timestamp_millis().to_string());
        metadata.insert("memories_processed".to_string(), stats.memories_processed.to_string());
        metadata.insert("memories_replayed".to_string(), stats.memories_replayed.to_string());
        metadata.insert("memories_pruned".to_string(), stats.memories_pruned.to_string());
        metadata.insert("concepts_created".to_string(), stats.concepts_created.to_string());
        metadata.insert("associations_found".to_string(), stats.associations_found.to_string());
        
        if !stats.insights.is_empty() {
            metadata.insert("insights".to_string(), stats.insights.join("; "));
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
            debug!("Recorded dream journal entry");
            Ok(())
        } else {
            Err(format!("Record failed: status {}", resp.status()))
        }
    }
}

/// Embedding backend for dream summaries
enum EmbedderBackend {
    Http(EmbeddingClient),
    Hash(HashEmbedder),
}

impl EmbedderBackend {
    fn embed(&self, text: &str) -> Vec<f32> {
        match self {
            EmbedderBackend::Http(client) => {
                client.embed(text).unwrap_or_else(|_| {
                    HashEmbedder::new(EMBEDDING_DIM).embed(text)
                })
            }
            EmbedderBackend::Hash(hasher) => hasher.embed(text),
        }
    }
}

/// DreamActor - manages background consolidation
pub struct DreamActor {
    config: DreamConfig,
    is_running: bool,
    last_run: Option<DateTime<Utc>>,
    total_processed: usize,
    last_stats: ConsolidationStats,
    /// Dream journal for storing consolidation history
    journal: Option<DreamJournal>,
    /// Embedding backend for dream summaries
    embedder: EmbedderBackend,
}

impl DreamActor {
    pub fn new(config: DreamConfig) -> Self {
        // Initialize embedding backend
        let embedder = if let Some(ref url) = config.embedding_url {
            let client = EmbeddingClient::new(url);
            if client.health_check() {
                EmbedderBackend::Http(client)
            } else {
                EmbedderBackend::Hash(HashEmbedder::new(EMBEDDING_DIM))
            }
        } else {
            EmbedderBackend::Hash(HashEmbedder::new(128))
        };

        // Initialize dream journal
        let journal = if let Some(ref url) = config.vecdb_url {
            match DreamJournal::new(url, &config.collection) {
                Ok(j) => {
                    info!("✅ Dream journal connected: {}", url);
                    Some(j)
                }
                Err(e) => {
                    warn!("⚠️ Dream journal not available: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            is_running: false,
            last_run: None,
            total_processed: 0,
            last_stats: ConsolidationStats::default(),
            journal,
            embedder,
        }
    }

    /// Create with external backends (convenience)
    pub fn with_backends() -> Self {
        Self::new(DreamConfig::with_backends())
    }

    /// Run a consolidation cycle
    pub fn consolidate(
        &mut self,
        hippocampus: &mut HippocampusActor,
        neocortex: &mut NeocortexActor,
    ) -> ConsolidationStats {
        if self.is_running {
            debug!("Consolidation already running");
            return self.last_stats.clone();
        }
        
        self.is_running = true;
        let mut stats = ConsolidationStats {
            timestamp: Some(Utc::now()),
            ..Default::default()
        };
        
        info!("🌙 Starting dream consolidation...");
        
        // 1. Get all memories from hippocampus
        let memories = hippocampus.all_memories();
        stats.memories_processed = memories.len();
        
        if memories.is_empty() {
            info!("No memories to consolidate");
            self.is_running = false;
            return stats;
        }
        
        // 2. Find associations between memories
        let associations = neocortex.associate(&memories);
        stats.associations_found = associations.len();
        
        // Create links in hippocampus based on associations
        for (source, target, _strength) in &associations {
            hippocampus.link(source, target);
        }
        
        // Generate insight about associations
        if stats.associations_found > 0 {
            stats.insights.push(format!(
                "Found {} associations between memories",
                stats.associations_found
            ));
        }
        
        // 3. Group similar memories and generalize to concepts
        let strong_memories: Vec<_> = memories.iter()
            .filter(|m| m.strength >= self.config.replay_threshold)
            .cloned()
            .collect();
        
        if strong_memories.len() >= 3 {
            if let Some(concept_name) = neocortex.generalize(&strong_memories) {
                stats.concepts_created += 1;
                stats.insights.push(format!("Created concept: {}", concept_name));
            }
        }
        
        // 4. Replay important memories (reinforce them)
        for memory in &strong_memories {
            hippocampus.reinforce(&memory.id, self.config.replay_reinforcement);
            stats.memories_replayed += 1;
        }
        
        // 5. Apply forgetting to weak memories
        let pruned = hippocampus.apply_forgetting(self.config.dream_decay_rate);
        stats.memories_pruned = pruned;
        
        if pruned > 0 {
            stats.insights.push(format!("Pruned {} weak memories", pruned));
        }
        
        // 6. Record to dream journal if available
        if let Some(ref journal) = self.journal {
            let summary = format!(
                "Dream consolidation: {} memories, {} replayed, {} pruned, {} concepts",
                stats.memories_processed,
                stats.memories_replayed,
                stats.memories_pruned,
                stats.concepts_created
            );
            let embedding = self.embedder.embed(&summary);
            
            if let Err(e) = journal.record(&stats, &embedding) {
                warn!("Failed to record dream journal: {}", e);
            }
        }
        
        // Update state
        self.last_run = Some(Utc::now());
        self.total_processed += stats.memories_processed;
        self.last_stats = stats.clone();
        self.is_running = false;
        
        info!(
            "🌙 Dream complete: {} processed, {} replayed, {} pruned, {} concepts",
            stats.memories_processed,
            stats.memories_replayed,
            stats.memories_pruned,
            stats.concepts_created
        );
        
        stats
    }

    /// Replay specific memories to strengthen them
    pub fn replay(
        &mut self,
        memory_ids: &[MemoryId],
        hippocampus: &mut HippocampusActor,
    ) -> usize {
        let mut count = 0;
        for id in memory_ids {
            if hippocampus.reinforce(id, self.config.replay_reinforcement).is_some() {
                count += 1;
            }
        }
        debug!("Replayed {} memories", count);
        count
    }

    /// Prune weak memories
    pub fn prune(&mut self, hippocampus: &mut HippocampusActor) -> usize {
        let removed = hippocampus.apply_forgetting(self.config.dream_decay_rate * 2.0);
        info!("Pruned {} weak memories", removed);
        removed
    }

    /// Start consolidation mode
    pub fn start(&mut self) {
        self.is_running = true;
        info!("Dream consolidation started");
    }

    /// Stop consolidation mode
    pub fn stop(&mut self) {
        self.is_running = false;
        info!("Dream consolidation stopped");
    }

    /// Check if consolidation is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get last run time
    pub fn last_run(&self) -> Option<DateTime<Utc>> {
        self.last_run
    }

    /// Get total memories processed
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Check if dream journal is connected
    pub fn has_journal(&self) -> bool {
        self.journal.is_some()
    }

    /// Get last consolidation stats
    pub fn last_stats(&self) -> &ConsolidationStats {
        &self.last_stats
    }

    /// Process a message and return response
    pub fn handle(
        &mut self,
        msg: DreamMessage,
        hippocampus: &mut HippocampusActor,
        neocortex: &mut NeocortexActor,
    ) -> DreamResponse {
        match msg {
            DreamMessage::StartConsolidation => {
                self.consolidate(hippocampus, neocortex);
                DreamResponse::ConsolidationStarted
            }
            DreamMessage::StopConsolidation => {
                self.stop();
                DreamResponse::ConsolidationStopped
            }
            DreamMessage::Replay { memory_ids } => {
                let count = self.replay(&memory_ids, hippocampus);
                DreamResponse::Replayed { count }
            }
            DreamMessage::Prune { threshold: _ } => {
                let removed = self.prune(hippocampus);
                DreamResponse::Pruned { removed }
            }
            DreamMessage::GetStatus => {
                DreamResponse::Status {
                    is_running: self.is_running,
                    last_run: self.last_run,
                    memories_processed: self.total_processed,
                }
            }
            DreamMessage::Tick => {
                if !self.is_running {
                    self.consolidate(hippocampus, neocortex);
                }
                DreamResponse::ConsolidationStarted
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hippocampus::HippocampusConfig;
    use crate::neocortex::NeocortexConfig;

    #[test]
    fn test_consolidation() {
        let mut dream = DreamActor::new(DreamConfig::default());
        let mut hippocampus = HippocampusActor::new(HippocampusConfig::default());
        let neocortex_config = NeocortexConfig {
            association_threshold: 0.1,
            ..Default::default()
        };
        let mut neocortex = NeocortexActor::new(neocortex_config);
        
        hippocampus.store(
            "Rust programming language systems".to_string(),
            MemoryContext::default(),
        );
        hippocampus.store(
            "Rust programming memory safety".to_string(),
            MemoryContext::default(),
        );
        hippocampus.store(
            "Rust programming zero-cost abstractions".to_string(),
            MemoryContext::default(),
        );
        
        let stats = dream.consolidate(&mut hippocampus, &mut neocortex);
        
        assert_eq!(stats.memories_processed, 3);
        assert!(stats.timestamp.is_some());
    }

    #[test]
    fn test_replay() {
        let mut dream = DreamActor::new(DreamConfig::default());
        let mut hippocampus = HippocampusActor::new(HippocampusConfig::default());
        
        let id = hippocampus.store(
            "Important memory".to_string(),
            MemoryContext::default(),
        );
        
        hippocampus.reinforce(&id, -0.5);
        let before = hippocampus.get(&id).unwrap().strength;
        
        dream.replay(&[id], &mut hippocampus);
        
        let after = hippocampus.get(&id).unwrap().strength;
        assert!(after > before);
    }

    #[test]
    fn test_status() {
        let dream = DreamActor::new(DreamConfig::default());
        
        assert!(!dream.is_running());
        assert!(dream.last_run().is_none());
        assert_eq!(dream.total_processed(), 0);
        assert!(!dream.has_journal());
    }

    #[test]
    fn test_insights() {
        let mut dream = DreamActor::new(DreamConfig::default());
        let mut hippocampus = HippocampusActor::new(HippocampusConfig::default());
        let mut neocortex = NeocortexActor::new(NeocortexConfig {
            association_threshold: 0.1,
            ..Default::default()
        });
        
        // Add memories
        hippocampus.store("Test memory one".to_string(), MemoryContext::default());
        hippocampus.store("Test memory two".to_string(), MemoryContext::default());
        
        let stats = dream.consolidate(&mut hippocampus, &mut neocortex);
        
        // Should have timestamp
        assert!(stats.timestamp.is_some());
        // May or may not have insights depending on associations found
    }
}
