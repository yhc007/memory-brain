//! CLS Memory System Messages
//! 
//! Message types for communication between memory actors.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a memory
pub type MemoryId = Uuid;

/// Memory importance/strength (0.0 - 1.0)
pub type Strength = f32;

/// Embedding vector
pub type Embedding = Vec<f32>;

/// Context information for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Source of the memory (conversation, observation, etc.)
    pub source: String,
    /// Timestamp when memory was created
    pub timestamp: DateTime<Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl Default for MemoryContext {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            timestamp: Utc::now(),
            tags: vec![],
            metadata: None,
        }
    }
}

/// A memory unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub content: String,
    pub embedding: Option<Embedding>,
    pub context: MemoryContext,
    pub strength: Strength,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    /// Links to related memories
    pub links: Vec<MemoryId>,
}

impl Memory {
    pub fn new(content: String, context: MemoryContext) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding: None,
            context,
            strength: 1.0,
            access_count: 0,
            last_accessed: now,
            created_at: now,
            links: vec![],
        }
    }
}

/// Recall result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub memory: Memory,
    pub similarity: f32,
}

// ============================================================================
// Hippocampus Messages (Fast Episodic Memory)
// ============================================================================

/// Messages for HippocampusActor
#[derive(Debug, Clone)]
pub enum HippocampusMessage {
    /// Store a new memory
    Store {
        content: String,
        context: MemoryContext,
    },
    
    /// Recall memories by semantic similarity
    Recall {
        query: String,
        k: usize,
    },
    
    /// Get a specific memory by ID
    Get {
        id: MemoryId,
    },
    
    /// Update memory strength (reinforcement)
    Reinforce {
        id: MemoryId,
        delta: f32,
    },
    
    /// Apply forgetting to all memories
    ApplyForgetting {
        decay_rate: f32,
    },
    
    /// Get recent memories (working memory)
    GetRecent {
        limit: usize,
    },
    
    /// Link two memories
    Link {
        source: MemoryId,
        target: MemoryId,
    },
}

/// Response from HippocampusActor
#[derive(Debug, Clone)]
pub enum HippocampusResponse {
    Stored { id: MemoryId },
    Recalled { results: Vec<RecallResult> },
    Found { memory: Option<Memory> },
    Reinforced { new_strength: f32 },
    ForgettingApplied { affected: usize },
    RecentMemories { memories: Vec<Memory> },
    Linked,
    Error { message: String },
}

// ============================================================================
// Neocortex Messages (Slow Semantic Memory)
// ============================================================================

/// Messages for NeocortexActor
#[derive(Debug, Clone)]
pub enum NeocortexMessage {
    /// Find associations between memories
    Associate {
        memory_ids: Vec<MemoryId>,
    },
    
    /// Extract patterns from memories
    ExtractPatterns {
        memories: Vec<Memory>,
    },
    
    /// Generalize from specific memories to concepts
    Generalize {
        memories: Vec<Memory>,
    },
    
    /// Query semantic knowledge
    Query {
        concept: String,
    },
    
    /// Store consolidated knowledge
    StoreKnowledge {
        concept: String,
        description: String,
        source_memories: Vec<MemoryId>,
    },
}

/// Response from NeocortexActor
#[derive(Debug, Clone)]
pub enum NeocortexResponse {
    Associations { links: Vec<(MemoryId, MemoryId, f32)> },
    Patterns { patterns: Vec<String> },
    Generalized { concept: String },
    QueryResult { knowledge: Option<String> },
    KnowledgeStored { id: String },
    Error { message: String },
}

// ============================================================================
// Dream Messages (Background Consolidation)
// ============================================================================

/// Messages for DreamActor
#[derive(Debug, Clone)]
pub enum DreamMessage {
    /// Start consolidation process
    StartConsolidation,
    
    /// Stop consolidation
    StopConsolidation,
    
    /// Replay specific memories to strengthen them
    Replay {
        memory_ids: Vec<MemoryId>,
    },
    
    /// Prune weak memories
    Prune {
        threshold: f32,
    },
    
    /// Get consolidation status
    GetStatus,
    
    /// Scheduled tick (from scheduler)
    Tick,
}

/// Response from DreamActor
#[derive(Debug, Clone)]
pub enum DreamResponse {
    ConsolidationStarted,
    ConsolidationStopped,
    Replayed { count: usize },
    Pruned { removed: usize },
    Status { 
        is_running: bool,
        last_run: Option<DateTime<Utc>>,
        memories_processed: usize,
    },
    Error { message: String },
}

// ============================================================================
// Guardian Messages (Supervisor)
// ============================================================================

/// Messages for MemoryGuardianActor
#[derive(Debug, Clone)]
pub enum GuardianMessage {
    /// Store memory (routes to Hippocampus)
    Store {
        content: String,
        context: MemoryContext,
    },
    
    /// Recall memory (routes to Hippocampus)  
    Recall {
        query: String,
        k: usize,
    },
    
    /// Start dream/consolidation
    StartDream,
    
    /// Stop dream/consolidation
    StopDream,
    
    /// Get system stats
    GetStats,
    
    /// Shutdown all actors
    Shutdown,
}

/// Response from MemoryGuardianActor
#[derive(Debug, Clone)]
pub enum GuardianResponse {
    Stored { id: MemoryId },
    Recalled { results: Vec<RecallResult> },
    DreamStarted,
    DreamStopped,
    Stats {
        total_memories: usize,
        hippocampus_active: bool,
        neocortex_active: bool,
        dream_active: bool,
    },
    ShuttingDown,
    Error { message: String },
}
