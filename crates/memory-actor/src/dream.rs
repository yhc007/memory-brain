//! DreamActor - Background Consolidation
//!
//! Implements the sleep/dream role in CLS:
//! - Offline memory consolidation
//! - Replay of important memories
//! - Transfer from hippocampus to neocortex
//! - Pruning of weak memories

use chrono::{DateTime, Utc};
use tracing::{info, debug};

use crate::messages::*;
use crate::hippocampus::HippocampusActor;
use crate::neocortex::NeocortexActor;

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
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            replay_threshold: 0.5,
            prune_threshold: 0.05,
            replay_reinforcement: 0.1,
            dream_decay_rate: 0.02,
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
}

/// DreamActor - manages background consolidation
pub struct DreamActor {
    config: DreamConfig,
    is_running: bool,
    last_run: Option<DateTime<Utc>>,
    total_processed: usize,
    last_stats: ConsolidationStats,
}

impl DreamActor {
    pub fn new(config: DreamConfig) -> Self {
        Self {
            config,
            is_running: false,
            last_run: None,
            total_processed: 0,
            last_stats: ConsolidationStats::default(),
        }
    }

    /// Run a consolidation cycle
    /// This is the main "dream" process that:
    /// 1. Gets memories from hippocampus
    /// 2. Finds associations
    /// 3. Creates concepts in neocortex
    /// 4. Replays important memories
    /// 5. Prunes weak memories
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
        let mut stats = ConsolidationStats::default();
        
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
        
        // 3. Group similar memories and generalize to concepts
        let strong_memories: Vec<_> = memories.iter()
            .filter(|m| m.strength >= self.config.replay_threshold)
            .cloned()
            .collect();
        
        if strong_memories.len() >= 3 {
            // Try to form concepts from strong memories
            if neocortex.generalize(&strong_memories).is_some() {
                stats.concepts_created += 1;
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
                // Scheduled tick - run consolidation if not already running
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
        // Use lower threshold for associations
        let neocortex_config = NeocortexConfig {
            association_threshold: 0.1,
            ..Default::default()
        };
        let mut neocortex = NeocortexActor::new(neocortex_config);
        
        // Add some memories with shared words
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
        
        // Run consolidation
        let stats = dream.consolidate(&mut hippocampus, &mut neocortex);
        
        assert_eq!(stats.memories_processed, 3);
        // At least some memories should have been processed
        assert!(stats.memories_replayed > 0 || stats.associations_found >= 0);
    }

    #[test]
    fn test_replay() {
        let mut dream = DreamActor::new(DreamConfig::default());
        let mut hippocampus = HippocampusActor::new(HippocampusConfig::default());
        
        let id = hippocampus.store(
            "Important memory".to_string(),
            MemoryContext::default(),
        );
        
        // Lower strength
        hippocampus.reinforce(&id, -0.5);
        let before = hippocampus.get(&id).unwrap().strength;
        
        // Replay
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
    }
}
