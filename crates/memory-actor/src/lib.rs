//! # Memory Actor - CLS-based Memory System
//!
//! A Complementary Learning System (CLS) implementation using Pekko Actor.
//!
//! ## Architecture
//!
//! ```text
//!                     ┌─────────────────────┐
//!                     │   MemoryGuardian    │
//!                     │   (Supervisor)      │
//!                     └──────────┬──────────┘
//!                                │
//!          ┌─────────────────────┼─────────────────────┐
//!          │                     │                     │
//!          ▼                     ▼                     ▼
//!  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
//!  │ HippocampusActor│  │  NeocortexActor │  │   DreamActor    │
//!  │                 │  │                 │  │                 │
//!  │ - store()       │  │ - query()       │  │ - consolidate() │
//!  │ - recall()      │  │ - associate()   │  │ - replay()      │
//!  │ - forget()      │  │ - generalize()  │  │ - prune()       │
//!  └─────────────────┘  └─────────────────┘  └─────────────────┘
//! ```
//!
//! ## CLS Theory
//!
//! The Complementary Learning Systems theory proposes that intelligent agents
//! require two learning systems:
//!
//! 1. **Hippocampus** (fast learning): Quickly stores episodic memories with
//!    high plasticity. Prone to interference but captures specific experiences.
//!
//! 2. **Neocortex** (slow learning): Gradually extracts statistical regularities
//!    and semantic knowledge. Low plasticity but robust generalization.
//!
//! 3. **Sleep/Dream**: Offline consolidation transfers knowledge from hippocampus
//!    to neocortex through memory replay, without interfering with new learning.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use memory_actor::{MemoryGuardian, MemoryContext, GuardianMessage};
//!
//! // Create the memory system
//! let guardian = MemoryGuardian::new(Default::default());
//!
//! // Store a memory
//! guardian.send(GuardianMessage::Store {
//!     content: "Rust is awesome".to_string(),
//!     context: MemoryContext::default(),
//! });
//!
//! // Recall memories
//! guardian.send(GuardianMessage::Recall {
//!     query: "programming language".to_string(),
//!     k: 5,
//! });
//! ```

pub mod messages;
pub mod hippocampus;
pub mod neocortex;
pub mod dream;
pub mod guardian;

// Re-exports
pub use messages::*;
pub use hippocampus::{HippocampusActor, HippocampusConfig};
pub use neocortex::{NeocortexActor, NeocortexConfig, Concept};
pub use dream::{DreamActor, DreamConfig, ConsolidationStats};
pub use guardian::{MemoryGuardian, MemorySystemConfig, MemorySystemStats};
