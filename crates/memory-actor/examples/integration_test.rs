//! Integration test for memory-actor with real backends
//! 
//! Run: cargo run --example integration_test

use memory_actor::{
    HippocampusActor, HippocampusConfig, 
    MemoryContext,
};

fn main() {
    println!("🧠 Memory-Actor Integration Test\n");

    // Create actor with external backends
    let config = HippocampusConfig::with_backends();
    println!("Config: {:?}", config);
    
    let mut actor = HippocampusActor::new(config);
    
    // Check backend status
    let status = actor.backend_status();
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

    // Store some memories
    println!("\n📝 Storing memories...");
    
    let id1 = actor.store(
        "Pekko is a Rust actor framework inspired by Akka".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["rust".to_string(), "actor".to_string()],
            ..Default::default()
        },
    );
    println!("  Stored: {} (pekko)", id1);

    let id2 = actor.store(
        "CoreVecDB is a vector database written in Rust".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["rust".to_string(), "database".to_string()],
            ..Default::default()
        },
    );
    println!("  Stored: {} (corevecdb)", id2);

    let id3 = actor.store(
        "BGE-M3 provides multilingual embeddings for text".to_string(),
        MemoryContext {
            source: "test".to_string(),
            tags: vec!["embedding".to_string(), "nlp".to_string()],
            ..Default::default()
        },
    );
    println!("  Stored: {} (bge-m3)", id3);

    // Recall memories
    println!("\n🔍 Recalling 'Rust framework'...");
    let results = actor.recall("Rust framework", 3);
    
    for (i, r) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}] {}", 
            i + 1,
            r.similarity, 
            &r.memory.content[..r.memory.content.len().min(60)]
        );
    }

    println!("\n🔍 Recalling 'vector database'...");
    let results = actor.recall("vector database", 3);
    
    for (i, r) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}] {}", 
            i + 1,
            r.similarity, 
            &r.memory.content[..r.memory.content.len().min(60)]
        );
    }

    // Stats
    println!("\n📊 Memory count: {}", actor.count());
    println!("\n✅ Integration test complete!");
}
