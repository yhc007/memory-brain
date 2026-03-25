//! MemoryGuardian - Supervisor Actor
//!
//! The top-level supervisor that coordinates:
//! - HippocampusActor (fast episodic memory)
//! - NeocortexActor (slow semantic memory)
//! - DreamActor (background consolidation)

use tracing::info;

use crate::messages::*;
use crate::hippocampus::{HippocampusActor, HippocampusConfig};
use crate::neocortex::{NeocortexActor, NeocortexConfig};
use crate::dream::{DreamActor, DreamConfig};

/// Configuration for the entire memory system
#[derive(Debug, Clone)]
pub struct MemorySystemConfig {
    pub hippocampus: HippocampusConfig,
    pub neocortex: NeocortexConfig,
    pub dream: DreamConfig,
}

impl Default for MemorySystemConfig {
    fn default() -> Self {
        Self {
            hippocampus: HippocampusConfig::default(),
            neocortex: NeocortexConfig::default(),
            dream: DreamConfig::default(),
        }
    }
}

/// Statistics for the memory system
#[derive(Debug, Clone)]
pub struct MemorySystemStats {
    pub total_memories: usize,
    pub total_concepts: usize,
    pub hippocampus_active: bool,
    pub neocortex_active: bool,
    pub dream_active: bool,
    pub dream_last_run: Option<chrono::DateTime<chrono::Utc>>,
}

/// MemoryGuardian - the supervisor actor for the CLS memory system
/// 
/// This is the main entry point for interacting with the memory system.
/// It coordinates the hippocampus, neocortex, and dream actors.
pub struct MemoryGuardian {
    hippocampus: HippocampusActor,
    neocortex: NeocortexActor,
    dream: DreamActor,
    is_active: bool,
}

impl MemoryGuardian {
    /// Create a new memory system with default configuration
    pub fn new(config: MemorySystemConfig) -> Self {
        info!("🧠 Initializing CLS Memory System...");
        
        let guardian = Self {
            hippocampus: HippocampusActor::new(config.hippocampus),
            neocortex: NeocortexActor::new(config.neocortex),
            dream: DreamActor::new(config.dream),
            is_active: true,
        };
        
        info!("🧠 CLS Memory System initialized");
        guardian
    }

    /// Store a new memory
    pub fn store(&mut self, content: String, context: MemoryContext) -> MemoryId {
        let id = self.hippocampus.store(content, context);
        info!("Stored memory: {}", id);
        id
    }

    /// Recall memories by semantic similarity
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        self.hippocampus.recall(query, k)
    }

    /// Get a specific memory by ID
    pub fn get(&self, id: &MemoryId) -> Option<Memory> {
        self.hippocampus.get(id)
    }

    /// Start dream consolidation
    pub fn start_dream(&mut self) {
        info!("🌙 Starting dream consolidation...");
        self.dream.consolidate(&mut self.hippocampus, &mut self.neocortex);
    }

    /// Stop dream consolidation
    pub fn stop_dream(&mut self) {
        self.dream.stop();
    }

    /// Get system statistics
    pub fn stats(&self) -> MemorySystemStats {
        MemorySystemStats {
            total_memories: self.hippocampus.count(),
            total_concepts: self.neocortex.count(),
            hippocampus_active: true,
            neocortex_active: true,
            dream_active: self.dream.is_running(),
            dream_last_run: self.dream.last_run(),
        }
    }

    /// Query semantic knowledge
    pub fn query_knowledge(&self, concept: &str) -> Option<String> {
        self.neocortex.query(concept).map(|c| c.description.clone())
    }

    /// Get recent memories (working memory)
    pub fn recent(&self, limit: usize) -> Vec<Memory> {
        self.hippocampus.get_recent(limit)
    }

    /// Shutdown the memory system
    pub fn shutdown(&mut self) {
        info!("🧠 Shutting down CLS Memory System...");
        self.dream.stop();
        self.is_active = false;
        info!("🧠 CLS Memory System shutdown complete");
    }

    /// Check if the system is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Process a guardian message
    pub fn handle(&mut self, msg: GuardianMessage) -> GuardianResponse {
        if !self.is_active {
            return GuardianResponse::Error {
                message: "Memory system is not active".to_string(),
            };
        }

        match msg {
            GuardianMessage::Store { content, context } => {
                let id = self.store(content, context);
                GuardianResponse::Stored { id }
            }
            GuardianMessage::Recall { query, k } => {
                let results = self.recall(&query, k);
                GuardianResponse::Recalled { results }
            }
            GuardianMessage::StartDream => {
                self.start_dream();
                GuardianResponse::DreamStarted
            }
            GuardianMessage::StopDream => {
                self.stop_dream();
                GuardianResponse::DreamStopped
            }
            GuardianMessage::GetStats => {
                let stats = self.stats();
                GuardianResponse::Stats {
                    total_memories: stats.total_memories,
                    hippocampus_active: stats.hippocampus_active,
                    neocortex_active: stats.neocortex_active,
                    dream_active: stats.dream_active,
                }
            }
            GuardianMessage::Shutdown => {
                self.shutdown();
                GuardianResponse::ShuttingDown
            }
        }
    }
}

// ============================================================================
// Integration with Pekko Actor (future implementation)
// ============================================================================

/// Actor wrapper for MemoryGuardian
/// 
/// This will integrate with pekko-actor for proper actor supervision
/// and message passing in a distributed system.
/// 
/// ```rust,ignore
/// use pekko_actor::{ActorSystem, Props};
/// use memory_actor::MemoryGuardianActor;
/// 
/// let system = ActorSystem::new("memory-system");
/// let guardian = system.spawn(Props::new(|| MemoryGuardianActor::new()));
/// 
/// // Send messages
/// guardian.tell(GuardianMessage::Store { ... });
/// 
/// // Ask pattern
/// let result = guardian.ask(|reply| GuardianMessage::Recall { 
///     query: "rust".to_string(),
///     k: 5,
///     reply_to: reply,
/// }).await;
/// ```
#[cfg(feature = "pekko")]
pub mod actor {
    use super::*;
    // TODO: Implement proper pekko-actor integration
    // This would use the actual Actor trait from pekko-actor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardian_store_and_recall() {
        let mut guardian = MemoryGuardian::new(MemorySystemConfig::default());
        
        let id = guardian.store(
            "Pekko is a Rust actor framework".to_string(),
            MemoryContext {
                source: "test".to_string(),
                tags: vec!["rust".to_string(), "actor".to_string()],
                ..Default::default()
            },
        );
        
        // Should be able to recall
        let results = guardian.recall("Pekko", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, id);
    }

    #[test]
    fn test_guardian_dream() {
        let mut guardian = MemoryGuardian::new(MemorySystemConfig::default());
        
        // Add memories
        guardian.store("Memory one about Rust".to_string(), MemoryContext::default());
        guardian.store("Memory two about Rust".to_string(), MemoryContext::default());
        guardian.store("Memory three about Rust".to_string(), MemoryContext::default());
        
        // Run dream
        guardian.start_dream();
        
        let stats = guardian.stats();
        assert!(stats.dream_last_run.is_some());
    }

    #[test]
    fn test_guardian_stats() {
        let mut guardian = MemoryGuardian::new(MemorySystemConfig::default());
        
        guardian.store("Test memory".to_string(), MemoryContext::default());
        
        let stats = guardian.stats();
        assert_eq!(stats.total_memories, 1);
        assert!(stats.hippocampus_active);
        assert!(stats.neocortex_active);
    }

    #[test]
    fn test_guardian_shutdown() {
        let mut guardian = MemoryGuardian::new(MemorySystemConfig::default());
        
        assert!(guardian.is_active());
        
        guardian.shutdown();
        
        assert!(!guardian.is_active());
    }

    #[test]
    fn test_guardian_message_handling() {
        let mut guardian = MemoryGuardian::new(MemorySystemConfig::default());
        
        // Store via message
        let response = guardian.handle(GuardianMessage::Store {
            content: "Test content".to_string(),
            context: MemoryContext::default(),
        });
        
        match response {
            GuardianResponse::Stored { id } => {
                // Recall via message
                let recall_response = guardian.handle(GuardianMessage::Recall {
                    query: "Test".to_string(),
                    k: 5,
                });
                
                match recall_response {
                    GuardianResponse::Recalled { results } => {
                        assert!(!results.is_empty());
                        assert_eq!(results[0].memory.id, id);
                    }
                    _ => panic!("Expected Recalled response"),
                }
            }
            _ => panic!("Expected Stored response"),
        }
    }
}
