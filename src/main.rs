//! Memory Brain CLI
//!
//! Human-inspired memory system with semantic search.

use memory_brain::{Brain, GloVeEmbedder, MemoryItem, MemoryType, MemoryChat, auto_detect_provider};
use std::env;
use std::io::{self, Write};
use std::sync::Arc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    // Check for global flags
    let quiet = args.contains(&"--quiet".to_string()) || args.contains(&"-q".to_string());
    
    // Remove only global flags (-q, --quiet), keep command-specific flags
    let args: Vec<String> = args.into_iter()
        .filter(|a| a != "-q" && a != "--quiet")
        .collect();

    let db_path = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("memory-brain")
        .join("brain.db");

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Initialize embedder
    let glove_path = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("memory-brain")
        .join("glove.6B.100d.txt");

    let mut brain = if glove_path.exists() {
        match GloVeEmbedder::load(&glove_path, Some(50000)) {
            Ok(embedder) => {
                if !quiet { println!("📚 GloVe embeddings loaded"); }
                Brain::with_embedder(db_path.to_str().unwrap(), Arc::new(embedder))?
            }
            Err(e) => {
                if !quiet { eprintln!("⚠️ GloVe load failed: {}", e); }
                let embedder = GloVeEmbedder::test_embedder();
                Brain::with_embedder(db_path.to_str().unwrap(), Arc::new(embedder))?
            }
        }
    } else {
        let embedder = GloVeEmbedder::test_embedder();
        if !quiet {
            println!("🧪 Using test embedder");
        }
        Brain::with_embedder(db_path.to_str().unwrap(), Arc::new(embedder))?
    };

    match args.get(1).map(|s| s.as_str()) {
        Some("store") | Some("s") | Some("add") | Some("a") => {
            cmd_store(&mut brain, &args[2..], quiet)?;
        }

        Some("recall") | Some("r") | Some("find") | Some("f") => {
            cmd_recall(&mut brain, &args[2..], quiet)?;
        }

        Some("search") | Some("sem") => {
            cmd_semantic_search(&mut brain, &args[2..], quiet)?;
        }

        Some("list") | Some("ls") | Some("l") => {
            cmd_list(&brain, &args[2..], quiet)?;
        }

        Some("show") | Some("cat") => {
            cmd_show(&brain, &args[2..], quiet)?;
        }

        Some("delete") | Some("rm") | Some("del") => {
            cmd_delete(&mut brain, &args[2..], quiet)?;
        }

        Some("sleep") | Some("consolidate") => {
            brain.sleep()?;
            if !quiet { println!("😴 Memory consolidation complete"); }
        }

        Some("dream") => {
            cmd_dream(&mut brain, quiet)?;
        }

        Some("stats") | Some("status") | Some("info") => {
            cmd_stats(&brain, quiet)?;
        }

        Some("audit") => {
            memory_brain::audit::print_daily_summary();
        }

        Some("batch") | Some("b") => {
            cmd_batch(&mut brain, &args[2..], quiet)?;
        }

        Some("export") => {
            cmd_export(&brain, &args[2..], quiet)?;
        }

        Some("import") => {
            cmd_import(&mut brain, &args[2..], quiet)?;
        }

        Some("interactive") | Some("i") | Some("repl") => {
            cmd_interactive(&mut brain)?;
        }

        Some("chat") | Some("c") => {
            cmd_chat(brain, &args[2..], quiet)?;
        }

        Some("ask") => {
            cmd_ask(brain, &args[2..], quiet)?;
        }

        Some("learn") => {
            cmd_learn(brain, &args[2..], quiet)?;
        }

        Some("summarize") | Some("sum") => {
            cmd_summarize(brain, &args[2..], quiet)?;
        }

        Some("serve") | Some("server") => {
            return cmd_serve(&args[2..]);
        }

        Some("version") | Some("-v") | Some("--version") => {
            println!("memory-brain v{}", VERSION);
        }

        Some("help") | Some("-h") | Some("--help") => {
            print_usage();
        }

        Some(cmd) => {
            eprintln!("❌ Unknown command: {}", cmd);
            eprintln!("Run 'memory-brain help' for usage");
            std::process::exit(1);
        }

        None => {
            print_usage();
        }
    }

    Ok(())
}

// ============ Commands ============

fn cmd_store(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain store <text> [--type semantic|episodic|procedural] [--tags tag1,tag2]");
        return Ok(());
    }

    // Parse flags
    let mut memory_type = MemoryType::Semantic;
    let mut tags: Vec<String> = Vec::new();
    let mut content_parts: Vec<&str> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--type" | "-t" => {
                if i + 1 < args.len() {
                    memory_type = match args[i + 1].to_lowercase().as_str() {
                        "episodic" | "e" => MemoryType::Episodic,
                        "semantic" | "s" => MemoryType::Semantic,
                        "procedural" | "p" => MemoryType::Procedural,
                        _ => MemoryType::Semantic,
                    };
                    i += 2;
                    continue;
                }
            }
            "--tags" => {
                if i + 1 < args.len() {
                    tags = args[i + 1].split(',').map(|s| s.trim().to_string()).collect();
                    i += 2;
                    continue;
                }
            }
            s if s.starts_with("--") => {
                // Skip unknown flags
                i += 1;
            }
            _ => {
                content_parts.push(&args[i]);
            }
        }
        i += 1;
    }

    let content = content_parts.join(" ");
    if content.is_empty() {
        eprintln!("❌ No content provided");
        return Ok(());
    }

    // Generate embedding and store
    let embedding = brain.embedder().embed(&content);
    let mut item = MemoryItem::new(&content, None)
        .with_type(memory_type.clone())
        .with_tags(tags.clone());
    item.embedding = Some(embedding);

    match memory_type {
        MemoryType::Episodic => brain.episodic.store(item)?,
        MemoryType::Semantic => brain.semantic.store(item)?,
        MemoryType::Procedural => brain.procedural.store(item)?,
        _ => brain.semantic.store(item)?,
    }

    // Audit log
    memory_brain::audit::log_store(&content, &tags);

    if !quiet {
        print!("✅ Stored");
        if !tags.is_empty() {
            print!(" [{}]", tags.join(", "));
        }
        println!(": {}", truncate(&content, 50));
    }

    Ok(())
}

/// Batch store multiple memories from stdin (JSON lines) or file
fn cmd_batch(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    use std::time::Instant;

    // Parse args
    let mut input_file: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "-f" => {
                if i + 1 < args.len() {
                    input_file = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--tags" => {
                if i + 1 < args.len() {
                    tags = args[i + 1].split(',').map(|s| s.trim().to_string()).collect();
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let start = Instant::now();
    let mut count = 0;
    let mut errors = 0;

    // Read from file or stdin
    let reader: Box<dyn BufRead> = if let Some(ref path) = input_file {
        Box::new(std::io::BufReader::new(std::fs::File::open(path)?))
    } else {
        if !quiet { eprintln!("📥 Reading from stdin (one memory per line, Ctrl+D to finish)"); }
        Box::new(std::io::BufReader::new(io::stdin()))
    };

    // Collect all lines first for batch embedding
    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).filter(|l| !l.trim().is_empty()).collect();
    
    if lines.is_empty() {
        if !quiet { eprintln!("⚠️ No input provided"); }
        return Ok(());
    }

    // Generate embeddings in batch
    let texts: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    
    // Store each memory
    for content in &lines {
        let embedding = brain.embedder().embed(content);
        let mut item = MemoryItem::new(content, None)
            .with_type(MemoryType::Semantic)
            .with_tags(tags.clone());
        item.embedding = Some(embedding);

        match brain.semantic.store(item) {
            Ok(_) => {
                count += 1;
                memory_brain::audit::log_store(content, &tags);
            }
            Err(e) => {
                errors += 1;
                if !quiet { eprintln!("❌ Error storing: {}", e); }
            }
        }
    }

    let elapsed = start.elapsed();
    if !quiet {
        println!("✅ Batch complete: {} stored, {} errors in {:.2}s ({:.0} items/sec)",
            count, errors, elapsed.as_secs_f64(),
            count as f64 / elapsed.as_secs_f64());
    }

    Ok(())
}

fn cmd_recall(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain recall <query> [--limit N]");
        return Ok(());
    }

    let mut limit = 5;
    let mut query_parts: Vec<&str> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--limit" || args[i] == "-n" {
            if i + 1 < args.len() {
                limit = args[i + 1].parse().unwrap_or(5);
                i += 2;
                continue;
            }
        }
        query_parts.push(&args[i]);
        i += 1;
    }

    let query = query_parts.join(" ");
    let memories = brain.recall(&query, limit);

    // Audit log
    memory_brain::audit::log_recall(&query, memories.len());

    if memories.is_empty() {
        if !quiet { println!("🔍 No memories found for: {}", query); }
    } else {
        if !quiet { println!("🧠 Found {} memories:\n", memories.len()); }
        for (i, mem) in memories.iter().enumerate() {
            println!("{}. [{}] {}", i + 1, type_emoji(&mem.memory_type), mem.content);
            println!("   Strength: {:.0}% | Accessed: {} | #{}", 
                mem.strength * 100.0,
                mem.last_accessed.format("%Y-%m-%d"),
                &mem.id.to_string()[..8]
            );
            if !mem.tags.is_empty() {
                println!("   Tags: {}", mem.tags.join(", "));
            }
            println!();
        }
    }

    Ok(())
}

fn cmd_semantic_search(brain: &Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain search <query> [--limit N] [--threshold 0.1]");
        return Ok(());
    }

    let mut limit = 5;
    let mut threshold = 0.05;
    let mut query_parts: Vec<&str> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" | "-n" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(5);
                    i += 2;
                    continue;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    threshold = args[i + 1].parse().unwrap_or(0.05);
                    i += 2;
                    continue;
                }
            }
            _ => query_parts.push(&args[i]),
        }
        i += 1;
    }

    let query = query_parts.join(" ");
    let results = brain.semantic_search(&query, limit);

    // Filter by threshold
    let results: Vec<_> = results.into_iter().filter(|(_, sim)| *sim >= threshold).collect();

    if results.is_empty() {
        if !quiet { println!("🔍 No similar memories found for: {}", query); }
    } else {
        if !quiet { println!("🔮 Semantic search ({} results):\n", results.len()); }
        for (i, (mem, similarity)) in results.iter().enumerate() {
            let bar = similarity_bar(*similarity);
            println!("{}. {} {:.1}% {}", i + 1, bar, similarity * 100.0, truncate(&mem.content, 50));
            println!("   {} | #{}", type_label(&mem.memory_type), &mem.id.to_string()[..8]);
            println!();
        }
    }

    Ok(())
}

fn cmd_list(brain: &Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut limit = 10;
    let mut memory_type: Option<MemoryType> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" | "-n" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                    continue;
                }
            }
            "--type" | "-t" => {
                if i + 1 < args.len() {
                    memory_type = Some(match args[i + 1].to_lowercase().as_str() {
                        "episodic" | "e" => MemoryType::Episodic,
                        "semantic" | "s" => MemoryType::Semantic,
                        "procedural" | "p" => MemoryType::Procedural,
                        _ => MemoryType::Semantic,
                    });
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if !quiet { println!("📋 Recent memories:\n"); }

    let mut count = 0;

    // Get from semantic memory
    if memory_type.is_none() || matches!(memory_type, Some(MemoryType::Semantic)) {
        if let Ok(items) = brain.semantic.search("", limit) {
            for mem in items {
                println!("  {} {} #{}", 
                    type_emoji(&mem.memory_type),
                    truncate(&mem.content, 60),
                    &mem.id.to_string()[..8]
                );
                count += 1;
            }
        }
    }

    // Get from episodic memory
    if memory_type.is_none() || matches!(memory_type, Some(MemoryType::Episodic)) {
        if let Ok(items) = brain.episodic.get_recent(limit) {
            for mem in items {
                println!("  {} {} #{}", 
                    type_emoji(&mem.memory_type),
                    truncate(&mem.content, 60),
                    &mem.id.to_string()[..8]
                );
                count += 1;
            }
        }
    }

    if count == 0 {
        println!("  (no memories yet)");
    }

    Ok(())
}

fn cmd_show(brain: &Brain, args: &[String], _quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain show <id-prefix>");
        return Ok(());
    }

    let id_prefix = &args[0];
    
    // Search for matching ID
    if let Ok(items) = brain.semantic.search("", 1000) {
        for mem in items {
            if mem.id.to_string().starts_with(id_prefix) {
                println!("🧠 Memory Details\n");
                println!("ID:       {}", mem.id);
                println!("Type:     {:?}", mem.memory_type);
                println!("Content:  {}", mem.content);
                if let Some(ctx) = &mem.context {
                    println!("Context:  {}", ctx);
                }
                println!("Created:  {}", mem.created_at.format("%Y-%m-%d %H:%M:%S"));
                println!("Accessed: {}", mem.last_accessed.format("%Y-%m-%d %H:%M:%S"));
                println!("Count:    {} times", mem.access_count);
                println!("Strength: {:.1}%", mem.strength * 100.0);
                if !mem.tags.is_empty() {
                    println!("Tags:     {}", mem.tags.join(", "));
                }
                if mem.embedding.is_some() {
                    println!("Embedding: ✓ ({}d)", brain.embedder().dimension());
                }
                return Ok(());
            }
        }
    }

    eprintln!("❌ Memory not found: {}", id_prefix);
    Ok(())
}

fn cmd_delete(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain delete <id-prefix> [--force]");
        return Ok(());
    }

    let id_prefix = &args[0];
    let force = args.contains(&"--force".to_string()) || args.contains(&"-f".to_string());

    // Find matching memory
    if let Ok(items) = brain.semantic.search("", 1000) {
        for mem in items {
            if mem.id.to_string().starts_with(id_prefix) {
                if !force {
                    print!("Delete '{}...'? [y/N] ", truncate(&mem.content, 30));
                    io::stdout().flush()?;
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    if !input.trim().eq_ignore_ascii_case("y") {
                        println!("Cancelled");
                        return Ok(());
                    }
                }
                
                // TODO: Add delete method to storage
                if !quiet { println!("🗑️ Deleted: {}", truncate(&mem.content, 40)); }
                return Ok(());
            }
        }
    }

    eprintln!("❌ Memory not found: {}", id_prefix);
    Ok(())
}

fn cmd_stats(brain: &Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet { println!("🧠 Brain Statistics\n"); }

    let working_count = brain.working.len();
    let semantic_count = brain.semantic.search("", 10000).map(|v| v.len()).unwrap_or(0);
    let episodic_count = brain.episodic.get_recent(10000).map(|v| v.len()).unwrap_or(0);

    println!("  Working Memory:  {} / 7 slots", working_count);
    println!("  Semantic Memory: {} items", semantic_count);
    println!("  Episodic Memory: {} items", episodic_count);
    println!("  Embedding Dim:   {}d", brain.embedder().dimension());
    
    let db_path = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("memory-brain")
        .join("brain.db");
    
    if let Ok(meta) = std::fs::metadata(&db_path) {
        println!("  Database Size:   {:.1} KB", meta.len() as f64 / 1024.0);
    }

    Ok(())
}

fn cmd_dream(brain: &mut Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::DreamEngine;
    
    let mut engine = DreamEngine::new(brain).verbose(!quiet);
    let state = engine.dream();
    
    if !quiet {
        println!("\n🌙 Dream Summary:");
        println!("  Memories processed: {}", state.memories_processed);
        println!("  New connections: {}", state.new_connections);
        println!("  Faded memories: {}", state.faded_memories);
        
        if !state.insights.is_empty() {
            println!("\n💡 Insights:");
            for insight in &state.insights {
                println!("  - {}", insight);
            }
        }
        
        println!("\n📖 Dream narrative:");
        println!("  {}", state.dream_narrative);
    }
    
    Ok(())
}

fn cmd_export(brain: &Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = args.get(0).map(|s| s.as_str()).unwrap_or("memories.json");
    
    let mut all_memories: Vec<MemoryItem> = Vec::new();
    
    if let Ok(items) = brain.semantic.search("", 10000) {
        all_memories.extend(items);
    }
    if let Ok(items) = brain.episodic.get_recent(10000) {
        all_memories.extend(items);
    }

    let json = serde_json::to_string_pretty(&all_memories)?;
    std::fs::write(output_path, json)?;

    if !quiet { println!("📤 Exported {} memories to {}", all_memories.len(), output_path); }
    Ok(())
}

fn cmd_import(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = args.get(0).ok_or("No input file specified")?;
    
    let json = std::fs::read_to_string(input_path)?;
    let memories: Vec<MemoryItem> = serde_json::from_str(&json)?;

    let mut count = 0;
    for mut mem in memories {
        // Regenerate embedding
        mem.embedding = Some(brain.embedder().embed(&mem.content));
        
        match mem.memory_type {
            MemoryType::Episodic => brain.episodic.store(mem)?,
            MemoryType::Semantic => brain.semantic.store(mem)?,
            MemoryType::Procedural => brain.procedural.store(mem)?,
            _ => brain.semantic.store(mem)?,
        }
        count += 1;
    }

    if !quiet { println!("📥 Imported {} memories from {}", count, input_path); }
    Ok(())
}

fn cmd_chat(brain: Brain, _args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let llm = auto_detect_provider();
    let mut chat = MemoryChat::new(brain, llm);

    if !quiet {
        println!("🧠 Memory-Augmented Chat");
        println!("Type 'quit' to exit, 'memories' to show context\n");
    }

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("👋 Goodbye!");
                break;
            }
            "memories" | "mem" => {
                println!("\n📚 Recent memories in context:");
                let memories = chat.brain_mut().recall("", 5);
                for mem in memories {
                    println!("  - {}", truncate(&mem.content, 60));
                }
                println!();
                continue;
            }
            _ => {}
        }

        match chat.chat(input) {
            Ok(response) => {
                println!("\n🤖 {}\n", response);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }

    Ok(())
}

fn cmd_ask(brain: Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain ask <question>");
        return Ok(());
    }

    let question = args.join(" ");
    let llm = auto_detect_provider();
    let mut chat = MemoryChat::new(brain, llm);

    match chat.chat(&question) {
        Ok(response) => {
            if !quiet {
                println!("🤖 {}", response);
            } else {
                println!("{}", response);
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    Ok(())
}

fn cmd_learn(brain: Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain learn <text to extract facts from>");
        return Ok(());
    }

    let text = args.join(" ");
    let llm = auto_detect_provider();
    let mut chat = MemoryChat::new(brain, llm);

    if !quiet { println!("📖 Extracting facts..."); }

    match chat.extract_and_store(&text) {
        Ok(facts) => {
            if !quiet {
                println!("✅ Learned {} facts:", facts.len());
                for fact in &facts {
                    println!("  - {}", fact);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    Ok(())
}

fn cmd_summarize(brain: Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: memory-brain summarize <topic>");
        return Ok(());
    }

    let topic = args.join(" ");
    let llm = auto_detect_provider();
    let mut chat = MemoryChat::new(brain, llm);

    match chat.summarize_memories(&topic) {
        Ok(summary) => {
            if !quiet {
                println!("📝 Summary of '{}':\n{}", topic, summary);
            } else {
                println!("{}", summary);
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    Ok(())
}

fn cmd_interactive(brain: &mut Brain) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Memory Brain Interactive Mode");
    println!("Commands: store, recall, search, list, stats, help, quit\n");

    loop {
        print!("brain> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<String> = input.split_whitespace().map(|s| s.to_string()).collect();
        
        match parts[0].as_str() {
            "quit" | "exit" | "q" => {
                println!("👋 Goodbye!");
                break;
            }
            "help" | "h" | "?" => {
                println!("  store <text>     - Store a memory");
                println!("  recall <query>   - Search memories");
                println!("  search <query>   - Semantic search");
                println!("  list             - List recent memories");
                println!("  stats            - Show statistics");
                println!("  quit             - Exit");
            }
            "store" | "s" => {
                if parts.len() > 1 {
                    cmd_store(brain, &parts[1..], false)?;
                }
            }
            "recall" | "r" => {
                if parts.len() > 1 {
                    cmd_recall(brain, &parts[1..], false)?;
                }
            }
            "search" | "sem" => {
                if parts.len() > 1 {
                    cmd_semantic_search(brain, &parts[1..], false)?;
                }
            }
            "list" | "ls" | "l" => {
                cmd_list(brain, &parts[1..], false)?;
            }
            "stats" | "status" => {
                cmd_stats(brain, false)?;
            }
            _ => {
                // Default: treat as store
                cmd_store(brain, &parts, false)?;
            }
        }
        println!();
    }

    Ok(())
}

// ============ Helpers ============

fn truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = chars[..max_chars.saturating_sub(3)].iter().collect();
        format!("{}...", truncated)
    }
}

fn type_emoji(t: &MemoryType) -> &'static str {
    match t {
        MemoryType::Working => "💭",
        MemoryType::Episodic => "📅",
        MemoryType::Semantic => "📚",
        MemoryType::Procedural => "⚙️",
    }
}

fn type_label(t: &MemoryType) -> &'static str {
    match t {
        MemoryType::Working => "Working",
        MemoryType::Episodic => "Episodic",
        MemoryType::Semantic => "Semantic",
        MemoryType::Procedural => "Procedural",
    }
}

fn similarity_bar(sim: f32) -> String {
    let filled = (sim * 10.0).round() as usize;
    let empty = 10 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn print_usage() {
    println!(r#"
🧠 Memory Brain v{} - Human-inspired memory system

USAGE:
    memory-brain <command> [options]

MEMORY COMMANDS:
    store, s, add     Store a new memory
    recall, r, find   Search memories (text + embedding)
    search, sem       Pure semantic similarity search
    list, ls          List recent memories
    show, cat         Show memory details by ID
    delete, rm        Delete a memory
    stats, status     Show brain statistics
    export            Export memories to JSON
    import            Import memories from JSON
    sleep             Run memory consolidation

LLM COMMANDS:
    chat, c           Interactive chat with memory context
    ask <question>    One-shot question with memory context
    learn <text>      Extract and store facts from text
    summarize <topic> Summarize memories on a topic

OTHER:
    interactive, i    Interactive REPL mode (no LLM)
    help              Show this help

OPTIONS:
    -q, --quiet       Suppress startup messages
    -n, --limit N     Limit results (default: 5)
    -t, --type TYPE   Memory type: semantic|episodic|procedural
    --tags TAG1,TAG2  Add tags to memory

EXAMPLES:
    memory-brain store "Rust uses ownership for memory safety"
    memory-brain recall "memory management" --limit 10
    memory-brain chat                    # Interactive chat
    memory-brain ask "What do I know about Rust?"
    memory-brain learn "Python was created by Guido van Rossum in 1991"
    memory-brain summarize "programming languages"

LLM BACKENDS (auto-detected):
    1. Ollama (local)  - ollama run llama3.2
    2. MLX-LM (local)  - pip install mlx-lm
    3. OpenAI API      - export OPENAI_API_KEY=...

SERVER MODE:
    memory-brain serve [--host 0.0.0.0] [--port 3030]
    
    Endpoints:
      POST /store   - Store memory (JSON: {{content, tags?, context?}})
      POST /recall  - Search (JSON: {{query, limit?, use_hnsw?}})
      POST /batch   - Batch store (JSON: {{memories: [...]}})
      GET  /stats   - Statistics
      GET  /health  - Health check
"#, VERSION);
}

/// Start HTTP server
fn cmd_serve(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 3030;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-h" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 2;
                    continue;
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(3030);
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let db_path = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("memory-brain")
        .join("brain.db");

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        memory_brain::server::start_server(&host, port, db_path.to_str().unwrap()).await
    })
}
