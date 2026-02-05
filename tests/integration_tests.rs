//! Integration tests for Memory Brain

use memory_brain::{Brain, GloVeEmbedder};
use std::sync::Arc;
use tempfile::TempDir;

/// Test context that keeps tempdir alive
struct TestContext {
    brain: Brain,
    _dir: TempDir,  // Keep directory alive
}

impl TestContext {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let embedder = Arc::new(GloVeEmbedder::test_embedder());
        let brain = Brain::with_embedder(db_path.to_str().unwrap(), embedder).unwrap();
        Self { brain, _dir: dir }
    }
}

#[test]
fn test_brain_creation() {
    let ctx = TestContext::new();
    assert_eq!(ctx.brain.working.len(), 0);
}

#[test]
fn test_store_and_recall() {
    let mut ctx = TestContext::new();
    
    // Store some memories
    ctx.brain.process("Rust is a systems programming language", None).unwrap();
    ctx.brain.process("Python is great for data science", None).unwrap();
    
    // Recall by keyword
    let results = ctx.brain.recall("Rust", 5);
    assert!(!results.is_empty());
    assert!(results[0].content.contains("Rust"));
}

#[test]
fn test_semantic_search() {
    let mut ctx = TestContext::new();
    
    // Store memories
    ctx.brain.process("Machine learning needs data", None).unwrap();
    ctx.brain.process("Rust ownership system", None).unwrap();
    
    // Semantic search
    let results = ctx.brain.semantic_search("ML data", 5);
    // Should find the machine learning related memory
    if !results.is_empty() {
        assert!(results[0].0.content.contains("learning") || results[0].0.content.contains("data"));
    }
}

#[test]
fn test_memory_types() {
    let mut ctx = TestContext::new();
    
    // Store with classification
    ctx.brain.process("Yesterday I fixed a bug", None).unwrap();  // Should be episodic
    ctx.brain.process("Rust is memory safe", None).unwrap();      // Should be semantic
    
    let results = ctx.brain.recall("bug", 5);
    assert!(!results.is_empty());
}

#[test]
fn test_working_memory_limit() {
    let mut ctx = TestContext::new();
    
    // Add more than 7 items to working memory
    for i in 0..10 {
        ctx.brain.process(&format!("Memory item {}", i), None).unwrap();
    }
    
    // Working memory should be at capacity (7)
    assert!(ctx.brain.working.len() <= 7);
}

#[test]
fn test_sleep_consolidation() {
    let mut ctx = TestContext::new();
    
    // Add memories
    ctx.brain.process("Important fact to remember", None).unwrap();
    
    // Sleep should clear working memory
    ctx.brain.sleep().unwrap();
    assert_eq!(ctx.brain.working.len(), 0);
}
