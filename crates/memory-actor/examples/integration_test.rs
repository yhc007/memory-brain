//! Integration test for memory-actor with real backends
//! 
//! Run: cargo run --example integration_test

use memory_actor::{
    HippocampusActor, HippocampusConfig, 
    NeocortexActor, NeocortexConfig,
    MemoryContext, Memory,
};

fn main() {
    println!("🧠 Memory-Actor Integration Test (Phase 3)\n");

    // =========================================================================
    // 1. Hippocampus (Episodic Memory)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("📦 HIPPOCAMPUS (Episodic Memory)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut hippocampus = HippocampusActor::new(HippocampusConfig::with_backends());
    
    let status = hippocampus.backend_status();
    println!("\n📊 Backend Status:");
    println!("  CoreVecDB: {}", if status.vecdb_connected { "✅" } else { "❌" });
    println!("  Embedding: {} ({}D)", 
        if status.embedding_http { "HTTP ✅" } else { "Hash 🔧" },
        status.embedding_dim
    );
    
    if !status.vecdb_connected {
        println!("\n⚠️  CoreVecDB not running! Start it with:");
        println!("   cd /Volumes/T7/Work/CoreVecDB && cargo run --release");
        return;
    }

    // Store memories
    println!("\n📝 Storing episodic memories...");
    
    let id1 = hippocampus.store(
        "Pekko is a Rust actor framework inspired by Akka".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["rust".to_string(), "actor".to_string()],
            ..Default::default()
        },
    );
    println!("  ✓ {}", &id1.to_string()[..8]);

    let id2 = hippocampus.store(
        "CoreVecDB is a high-performance vector database in Rust".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["rust".to_string(), "database".to_string()],
            ..Default::default()
        },
    );
    println!("  ✓ {}", &id2.to_string()[..8]);

    let id3 = hippocampus.store(
        "Actor model enables concurrent and distributed systems".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["concurrency".to_string(), "actor".to_string()],
            ..Default::default()
        },
    );
    println!("  ✓ {}", &id3.to_string()[..8]);

    // Recall
    println!("\n🔍 Recalling 'Rust actor'...");
    let results = hippocampus.recall("Rust actor", 3);
    for (i, r) in results.iter().enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, r.similarity, truncate(&r.memory.content, 50));
    }

    println!("\n📊 Hippocampus count: {}", hippocampus.count());

    // =========================================================================
    // 2. Neocortex (Semantic Memory)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🧬 NEOCORTEX (Semantic Memory)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut neocortex = NeocortexActor::new(NeocortexConfig::with_backends());
    
    println!("\n📊 Backend: {}", if neocortex.has_backends() { "CoreVecDB ✅" } else { "In-memory 🔧" });

    // Create memories for generalization
    let memories: Vec<Memory> = vec![
        Memory::new("Pekko uses actor model for concurrency".to_string(), MemoryContext::default()),
        Memory::new("Actors communicate via message passing".to_string(), MemoryContext::default()),
        Memory::new("Actor isolation prevents shared state bugs".to_string(), MemoryContext::default()),
    ];

    // Generalize to concept
    println!("\n🔮 Generalizing from 3 memories about actors...");
    if let Some(concept_name) = neocortex.generalize(&memories) {
        println!("  ✓ Created concept: '{}'", concept_name);
        
        if let Some(concept) = neocortex.query(&concept_name) {
            println!("  Description: {}", truncate(&concept.description, 60));
            println!("  Has embedding: {}", concept.embedding.is_some());
        }
    }

    // Store explicit knowledge
    println!("\n📚 Storing explicit knowledge...");
    neocortex.store_knowledge(
        "CLS_theory".to_string(),
        "Complementary Learning Systems: hippocampus for fast episodic, neocortex for slow semantic".to_string(),
        vec![],
    );
    println!("  ✓ Stored 'CLS_theory'");

    // Search concepts
    println!("\n🔍 Searching concepts for 'actor'...");
    let concept_results = neocortex.search_concepts("actor concurrency", 3);
    for (name, score) in &concept_results {
        println!("  - {} [{:.3}]", name, score);
    }

    println!("\n📊 Neocortex count: {}", neocortex.count());

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ INTEGRATION TEST COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Hippocampus: {} episodic memories", hippocampus.count());
    println!("  Neocortex: {} semantic concepts", neocortex.count());
    println!("\n🧠 CLS Memory System ready!");
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
