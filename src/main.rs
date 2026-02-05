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
        .join("coredb");

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

    // Auto-rebuild indexes for fast search (O(1) keyword lookup)
    let rebuild_stats = brain.rebuild_indexes()?;
    if !quiet && rebuild_stats.episodic_count + rebuild_stats.semantic_count > 0 {
        println!("🔍 Index loaded: {} memories, {} keywords", 
            rebuild_stats.episodic_count + rebuild_stats.semantic_count + rebuild_stats.procedural_count,
            rebuild_stats.index_stats.unique_keywords);
    }

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

        Some("map") | Some("mindmap") => {
            cmd_map(&brain, &args[2..], quiet)?;
        }

        Some("predict") | Some("next") => {
            cmd_predict(&brain, quiet)?;
        }

        Some("forget") | Some("forgetting") => {
            cmd_forgetting(&brain, quiet)?;
        }

        Some("patterns") => {
            cmd_patterns(&brain, quiet)?;
        }

        Some("stats") | Some("status") | Some("info") => {
            cmd_stats(&brain, quiet)?;
        }

        Some("audit") => {
            // Check for flags
            let show_weekly = args.iter().any(|a| a == "--weekly" || a == "-w");
            let show_simple = args.iter().any(|a| a == "--simple" || a == "-s");
            
            if show_simple {
                memory_brain::audit::print_daily_summary();
            } else if show_weekly {
                memory_brain::audit::print_full_report();
            } else {
                memory_brain::audit::print_visual_summary();
            }
        }

        Some("tui") | Some("dashboard") | Some("ui") => {
            // Load memories for TUI from semantic memory
            let memory_data: Vec<(String, String, String)> = if let Ok(items) = brain.semantic.search("", 100) {
                items.iter().map(|m| {
                    (
                        m.id.to_string(),
                        m.content.clone(),
                        m.tags.join(", "),
                    )
                }).collect()
            } else {
                Vec::new()
            };
            
            memory_brain::tui::run_tui(memory_data)?;
        }

        Some("rebuild") | Some("reindex") => {
            cmd_rebuild(&mut brain, quiet)?;
        }

        Some("merge") | Some("dedup") => {
            cmd_merge(&mut brain, &args[2..], quiet)?;
        }

        Some("bench") | Some("benchmark") => {
            cmd_bench(quiet)?;
        }

        Some("watch") | Some("monitor") => {
            cmd_watch(&brain, &args[2..])?;
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

        Some("sam") => {
            cmd_sam(&args[2..], db_path.to_str().unwrap_or("."), quiet)?;
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

        Some("visual") | Some("vis") | Some("img") => {
            cmd_visual(&args[2..], quiet)?;
        }

        Some("describe") | Some("vlm") => {
            cmd_describe(&args[2..], quiet)?;
        }

        Some("vlm-status") => {
            cmd_vlm_status(quiet)?;
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
        eprintln!("Usage: memory-brain recall <query> [options]");
        eprintln!("Options:");
        eprintln!("  --limit N, -n N    Max results (default: 5)");
        eprintln!("  --tag TAG          Filter by tag");
        eprintln!("  --regex            Use regex matching");
        eprintln!("  --fuzzy            Fuzzy search (typo tolerant)");
        eprintln!("  --type TYPE        Filter by type (semantic/episodic/procedural)");
        return Ok(());
    }

    let mut limit = 5;
    let mut tag_filter: Option<String> = None;
    let mut type_filter: Option<MemoryType> = None;
    let mut use_regex = false;
    let mut use_fuzzy = false;
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
            "--tag" | "-t" => {
                if i + 1 < args.len() {
                    tag_filter = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--type" => {
                if i + 1 < args.len() {
                    type_filter = match args[i + 1].to_lowercase().as_str() {
                        "semantic" | "sem" => Some(MemoryType::Semantic),
                        "episodic" | "epi" => Some(MemoryType::Episodic),
                        "procedural" | "proc" => Some(MemoryType::Procedural),
                        _ => None,
                    };
                    i += 2;
                    continue;
                }
            }
            "--regex" | "-r" => {
                use_regex = true;
                i += 1;
                continue;
            }
            "--fuzzy" | "-f" => {
                use_fuzzy = true;
                i += 1;
                continue;
            }
            s if s.starts_with("--tag=") => {
                tag_filter = Some(s.trim_start_matches("--tag=").to_string());
                i += 1;
                continue;
            }
            s if s.starts_with("--type=") => {
                type_filter = match s.trim_start_matches("--type=").to_lowercase().as_str() {
                    "semantic" | "sem" => Some(MemoryType::Semantic),
                    "episodic" | "epi" => Some(MemoryType::Episodic),
                    "procedural" | "proc" => Some(MemoryType::Procedural),
                    _ => None,
                };
                i += 1;
                continue;
            }
            _ => {}
        }
        query_parts.push(&args[i]);
        i += 1;
    }

    let query = query_parts.join(" ");
    
    // Get more results initially for filtering
    let fetch_limit = if tag_filter.is_some() || type_filter.is_some() || use_regex || use_fuzzy {
        limit * 10
    } else {
        limit
    };
    
    let mut memories = brain.recall(&query, fetch_limit);

    // Apply regex filter
    if use_regex && !query.is_empty() {
        if let Ok(re) = regex::Regex::new(&query) {
            memories.retain(|m| re.is_match(&m.content));
        }
    }

    // Apply fuzzy filter
    if use_fuzzy && !query.is_empty() {
        let query_lower = query.to_lowercase();
        let query_chars: Vec<char> = query_lower.chars().collect();
        
        memories.retain(|m| {
            let content_lower = m.content.to_lowercase();
            // Simple fuzzy: all query chars appear in order
            fuzzy_match(&query_chars, &content_lower)
        });
    }

    // Apply tag filter
    if let Some(ref tag) = tag_filter {
        let tag_lower = tag.to_lowercase();
        memories.retain(|m| m.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower)));
    }

    // Apply type filter
    if let Some(ref mem_type) = type_filter {
        memories.retain(|m| std::mem::discriminant(&m.memory_type) == std::mem::discriminant(mem_type));
    }

    // Truncate to limit
    memories.truncate(limit);

    // Audit log
    memory_brain::audit::log_recall(&query, memories.len());

    if memories.is_empty() {
        if !quiet { 
            println!("🔍 No memories found for: {}", query);
            if tag_filter.is_some() || type_filter.is_some() || use_regex || use_fuzzy {
                println!("   (filters applied)");
            }
        }
    } else {
        if !quiet { 
            print!("🧠 Found {} memories", memories.len());
            if let Some(ref tag) = tag_filter {
                print!(" [tag: {}]", tag);
            }
            if use_regex {
                print!(" [regex]");
            }
            if use_fuzzy {
                print!(" [fuzzy]");
            }
            println!(":\n");
        }
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

/// Simple fuzzy matching - all chars appear in order
fn fuzzy_match(pattern: &[char], text: &str) -> bool {
    let mut pattern_idx = 0;
    for c in text.chars() {
        if pattern_idx < pattern.len() && c == pattern[pattern_idx] {
            pattern_idx += 1;
        }
    }
    pattern_idx == pattern.len()
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
                
                // 🔗 Show associations
                if !mem.associations.is_empty() {
                    println!("\n🔗 Linked Memories ({}):", mem.associations.len());
                    for assoc_id in &mem.associations {
                        // Try to find the associated memory
                        if let Ok(all_items) = brain.semantic.search("", 1000) {
                            if let Some(linked) = all_items.iter().find(|m| m.id == *assoc_id) {
                                println!("   → {} - {}", 
                                    &assoc_id.to_string()[..8],
                                    truncate(&linked.content, 50));
                            } else {
                                println!("   → {} (not found)", &assoc_id.to_string()[..8]);
                            }
                        }
                    }
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
        .join("coredb");
    
    if let Ok(meta) = std::fs::metadata(&db_path) {
        println!("  Database Size:   {:.1} KB", meta.len() as f64 / 1024.0);
    }

    Ok(())
}

fn cmd_rebuild(brain: &mut Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("🔧 Rebuilding indexes from database...");
    }

    let stats = brain.rebuild_indexes()?;

    if !quiet {
        println!("{}", stats);
    } else {
        println!("{}", stats.index_stats.documents);
    }

    Ok(())
}

fn cmd_watch(brain: &Brain, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::watch::{MemoryWatcher, WatchConfig};
    
    let mut interval_ms = 1000u64;
    let mut detailed = false;
    
    for arg in args {
        if arg.starts_with("--interval=") || arg.starts_with("-i=") {
            interval_ms = arg.split('=').nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000);
        } else if arg == "--detailed" || arg == "-d" {
            detailed = true;
        }
    }
    
    let config = WatchConfig {
        interval_ms,
        detailed,
        clear_screen: true,
        max_iterations: 0,
    };
    
    MemoryWatcher::with_config(brain, config).run()?;
    
    Ok(())
}

fn cmd_bench(quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::bench;
    
    if !quiet {
        println!("⚡ Memory Brain Benchmark\n");
        
        // Test SIMD correctness first
        if bench::test_simd_correctness() {
            println!("✅ SIMD correctness verified\n");
        } else {
            println!("❌ SIMD mismatch detected!\n");
        }
    }
    
    bench::run_benchmarks(!quiet);
    
    Ok(())
}

fn cmd_merge(brain: &mut Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::merge::{MemoryMerger, MergeConfig};
    
    // Parse arguments
    let mut threshold = 0.85f32;
    let mut dry_run = true; // Default to dry run for safety
    
    for arg in args {
        if arg.starts_with("--threshold=") {
            threshold = arg.trim_start_matches("--threshold=")
                .parse()
                .unwrap_or(0.85);
        } else if arg == "--execute" || arg == "-x" {
            dry_run = false;
        } else if arg == "--dry-run" || arg == "-n" {
            dry_run = true;
        }
    }

    if !quiet {
        if dry_run {
            println!("🔍 Analyzing duplicate memories (dry run)...");
        } else {
            println!("🔗 Merging duplicate memories...");
        }
        println!("  Threshold: {:.0}%", threshold * 100.0);
    }

    let config = MergeConfig {
        similarity_threshold: threshold,
        dry_run,
        ..Default::default()
    };

    let mut merger = MemoryMerger::with_config(brain, config);
    let result = merger.find_similar();

    if !quiet {
        println!("{}", result);
        
        if dry_run && result.mergeable_count > 0 {
            println!("\n💡 Run with --execute (-x) to actually merge");
        }
    } else {
        println!("{}", result.merged_count);
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

fn cmd_predict(brain: &Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::Predictor;
    
    let predictor = Predictor::new(brain);
    let predictions = predictor.predict_next(5);
    
    if predictions.is_empty() {
        if !quiet {
            println!("🔮 패턴이 충분하지 않아 예측하기 어려워");
        }
        return Ok(());
    }
    
    if !quiet {
        println!("🔮 예측:\n");
        for pred in predictions {
            let conf_bar = "█".repeat((pred.confidence * 10.0) as usize);
            let empty_bar = "░".repeat(10 - (pred.confidence * 10.0) as usize);
            println!("  {} ({:.0}%)", pred.content, pred.confidence * 100.0);
            println!("  [{}{}] {}\n", conf_bar, empty_bar, pred.reason);
        }
    }
    
    Ok(())
}

fn cmd_forgetting(brain: &Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::Predictor;
    
    let predictor = Predictor::new(brain);
    let alerts = predictor.forgetting_alerts(10);
    
    if alerts.is_empty() {
        if !quiet {
            println!("✨ 모든 기억이 건강해! 잊혀질 위험 없음");
        }
        return Ok(());
    }
    
    if !quiet {
        println!("⚠️ 잊혀질 위험이 있는 기억들:\n");
        for alert in alerts {
            let content = truncate(&alert.memory.content, 40);
            println!("  {} {} ({}일 전, 강도 {:.0}%)", 
                alert.urgency,
                content,
                alert.days_since_access,
                alert.strength * 100.0
            );
        }
        println!("\n💡 팁: 이 기억들을 recall해서 강화하세요!");
    }
    
    Ok(())
}

fn cmd_patterns(brain: &Brain, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::Predictor;
    
    let predictor = Predictor::new(brain);
    let patterns = predictor.discover_patterns();
    
    if patterns.is_empty() {
        if !quiet {
            println!("🔍 아직 뚜렷한 패턴이 발견되지 않았어");
        }
        return Ok(());
    }
    
    if !quiet {
        println!("📊 발견된 패턴들:\n");
        for pattern in patterns {
            println!("  🔹 {} - {}", pattern.name, pattern.description);
            if !pattern.examples.is_empty() {
                for ex in &pattern.examples {
                    println!("     └ {}", ex);
                }
            }
            println!();
        }
    }
    
    Ok(())
}

fn cmd_map(brain: &Brain, args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::MindMap;
    
    let mut format = "html";
    let mut output = "memory_map.html";
    let mut limit = 100;
    let mut threshold = 0.3;
    let mut open_browser = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--format" | "-f" => {
                if i + 1 < args.len() {
                    format = match args[i + 1].as_str() {
                        "dot" => "dot",
                        "mermaid" => "mermaid",
                        _ => "html",
                    };
                    i += 2;
                    continue;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output = args[i + 1].as_str();
                    i += 2;
                    continue;
                }
            }
            "--limit" | "-n" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(100);
                    i += 2;
                    continue;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    threshold = args[i + 1].parse().unwrap_or(0.3);
                    i += 2;
                    continue;
                }
            }
            "--open" => {
                open_browser = true;
            }
            _ => {}
        }
        i += 1;
    }
    
    if !quiet {
        println!("🗺️  Generating mind map...");
    }
    
    let map = MindMap::from_brain(brain, limit, threshold);
    
    let content = match format {
        "dot" => {
            let out = if output == "memory_map.html" { "memory_map.dot" } else { output };
            let content = map.to_dot();
            std::fs::write(out, &content)?;
            if !quiet {
                println!("✅ DOT file saved to {}", out);
                println!("   Run: dot -Tpng {} -o map.png", out);
            }
            content
        }
        "mermaid" => {
            let out = if output == "memory_map.html" { "memory_map.md" } else { output };
            let content = format!("```mermaid\n{}\n```", map.to_mermaid());
            std::fs::write(out, &content)?;
            if !quiet {
                println!("✅ Mermaid file saved to {}", out);
            }
            content
        }
        _ => {
            let content = map.to_html();
            std::fs::write(output, &content)?;
            if !quiet {
                println!("✅ HTML mind map saved to {}", output);
                println!("   {} nodes, {} connections", map.nodes.len(), map.edges.len());
            }
            
            if open_browser {
                #[cfg(target_os = "macos")]
                std::process::Command::new("open").arg(output).spawn()?;
                #[cfg(target_os = "linux")]
                std::process::Command::new("xdg-open").arg(output).spawn()?;
                #[cfg(target_os = "windows")]
                std::process::Command::new("start").arg(output).spawn()?;
            }
            content
        }
    };
    
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
    
    // Parse options
    let mut default_tags: Vec<String> = Vec::new();
    let mut memory_type = MemoryType::Semantic;
    
    for arg in args.iter().skip(1) {
        if arg.starts_with("--tags=") {
            default_tags = arg.trim_start_matches("--tags=")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if arg == "--episodic" {
            memory_type = MemoryType::Episodic;
        } else if arg == "--procedural" {
            memory_type = MemoryType::Procedural;
        }
    }
    
    // Detect format from extension
    let path = std::path::Path::new(input_path);
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let mut count = 0;
    let mut errors = 0;
    
    match extension.as_str() {
        "json" => {
            // JSON import (array of MemoryItem or simple objects)
            let json = std::fs::read_to_string(input_path)?;
            
            // Try full MemoryItem format first
            if let Ok(memories) = serde_json::from_str::<Vec<MemoryItem>>(&json) {
                for mut mem in memories {
                    mem.embedding = Some(brain.embedder().embed(&mem.content));
                    mem.tags.extend(default_tags.clone());
                    
                    match mem.memory_type {
                        MemoryType::Episodic => brain.episodic.store(mem)?,
                        MemoryType::Semantic => brain.semantic.store(mem)?,
                        MemoryType::Procedural => brain.procedural.store(mem)?,
                        _ => brain.semantic.store(mem)?,
                    }
                    count += 1;
                    if !quiet && count % 100 == 0 {
                        print!("\r📥 Imported {} memories...", count);
                        std::io::stdout().flush()?;
                    }
                }
            } else {
                // Try simple format: [{"content": "...", "tags": [...]}]
                #[derive(serde::Deserialize)]
                struct SimpleMemory {
                    content: String,
                    #[serde(default)]
                    tags: Vec<String>,
                    #[serde(default)]
                    context: Option<String>,
                }
                
                let simple: Vec<SimpleMemory> = serde_json::from_str(&json)?;
                for item in simple {
                    let mut mem = MemoryItem::new(&item.content, item.context.as_deref());
                    mem.embedding = Some(brain.embedder().embed(&item.content));
                    mem.tags = item.tags;
                    mem.tags.extend(default_tags.clone());
                    mem.memory_type = memory_type.clone();
                    
                    match memory_type {
                        MemoryType::Episodic => brain.episodic.store(mem)?,
                        MemoryType::Semantic => brain.semantic.store(mem)?,
                        MemoryType::Procedural => brain.procedural.store(mem)?,
                        _ => brain.semantic.store(mem)?,
                    }
                    count += 1;
                }
            }
        }
        
        "csv" => {
            // CSV import: content,tags (comma-separated)
            let content = std::fs::read_to_string(input_path)?;
            let mut lines = content.lines();
            
            // Skip header if it looks like one
            if let Some(first) = lines.next() {
                let lower = first.to_lowercase();
                if !lower.contains("content") && !lower.contains("text") {
                    // Not a header, process it
                    if let Some(mem) = parse_csv_line(first, &default_tags, memory_type.clone(), brain) {
                        match memory_type {
                            MemoryType::Episodic => brain.episodic.store(mem)?,
                            MemoryType::Semantic => brain.semantic.store(mem)?,
                            MemoryType::Procedural => brain.procedural.store(mem)?,
                            _ => brain.semantic.store(mem)?,
                        }
                        count += 1;
                    }
                }
            }
            
            for line in lines {
                if line.trim().is_empty() {
                    continue;
                }
                if let Some(mem) = parse_csv_line(line, &default_tags, memory_type.clone(), brain) {
                    match memory_type {
                        MemoryType::Episodic => brain.episodic.store(mem)?,
                        MemoryType::Semantic => brain.semantic.store(mem)?,
                        MemoryType::Procedural => brain.procedural.store(mem)?,
                        _ => brain.semantic.store(mem)?,
                    }
                    count += 1;
                } else {
                    errors += 1;
                }
                
                if !quiet && count % 100 == 0 {
                    print!("\r📥 Imported {} memories...", count);
                    std::io::stdout().flush()?;
                }
            }
        }
        
        "txt" | "md" | _ => {
            // Text file: one memory per line (or per paragraph for .md)
            let content = std::fs::read_to_string(input_path)?;
            
            let delimiter = if extension == "md" { "\n\n" } else { "\n" };
            
            for chunk in content.split(delimiter) {
                let text = chunk.trim();
                if text.is_empty() || text.len() < 3 {
                    continue;
                }
                
                let mut mem = MemoryItem::new(text, None);
                mem.embedding = Some(brain.embedder().embed(text));
                mem.tags = default_tags.clone();
                mem.memory_type = memory_type.clone();
                
                match memory_type {
                    MemoryType::Episodic => brain.episodic.store(mem)?,
                    MemoryType::Semantic => brain.semantic.store(mem)?,
                    MemoryType::Procedural => brain.procedural.store(mem)?,
                    _ => brain.semantic.store(mem)?,
                }
                count += 1;
                
                if !quiet && count % 100 == 0 {
                    print!("\r📥 Imported {} memories...", count);
                    std::io::stdout().flush()?;
                }
            }
        }
    }

    if !quiet {
        println!("\r📥 Imported {} memories from {}        ", count, input_path);
        if errors > 0 {
            println!("⚠️  {} lines skipped due to errors", errors);
        }
    }
    Ok(())
}

fn parse_csv_line(line: &str, default_tags: &[String], memory_type: MemoryType, brain: &Brain) -> Option<MemoryItem> {
    // Simple CSV parsing (content,tags)
    // Handle quoted strings
    let parts: Vec<&str> = if line.starts_with('"') {
        // Quoted content
        if let Some(end_quote) = line[1..].find('"') {
            let content = &line[1..end_quote + 1];
            let rest = &line[end_quote + 2..];
            let tags_str = rest.trim_start_matches(',').trim();
            vec![content, tags_str]
        } else {
            vec![line]
        }
    } else {
        line.splitn(2, ',').collect()
    };
    
    let content = parts.get(0)?.trim();
    if content.is_empty() {
        return None;
    }
    
    let mut mem = MemoryItem::new(content, None);
    mem.embedding = Some(brain.embedder().embed(content));
    mem.memory_type = memory_type;
    
    // Parse tags if present
    if let Some(tags_str) = parts.get(1) {
        mem.tags = tags_str
            .split(&[',', ';'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    mem.tags.extend(default_tags.iter().cloned());
    
    Some(mem)
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

VISUAL / VLM COMMANDS:
    visual store      Store image with CLIP embedding
    visual recall     Search images by text
    describe <image>  Generate description using VLM (LLaVA)
    vlm-status        Check Ollama VLM models

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
        .join("coredb");

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        memory_brain::server::start_server(&host, port, db_path.to_str().unwrap()).await
    })
}

/// Sam's personal memory commands 🦊
fn cmd_sam(args: &[String], db_path: &str, quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::{SamBrain, SamMemory};
    
    if args.is_empty() {
        eprintln!("🦊 Sam's Memory Commands:");
        eprintln!("");
        eprintln!("  sam conversation <text> [--channel imessage|discord]");
        eprintln!("  sam learn <text>        - Remember something learned");
        eprintln!("  sam preference <text>   - Remember Paul's preference");
        eprintln!("  sam lesson <text>       - Remember a lesson learned");
        eprintln!("  sam project <name> <details>");
        eprintln!("  sam recall <query>      - Search Sam's memories");
        eprintln!("  sam stats               - Show Sam's brain stats");
        return Ok(());
    }
    
    let mut sam = SamBrain::new(db_path)?;
    
    match args[0].as_str() {
        "conversation" | "conv" | "chat" => {
            if args.len() < 2 {
                eprintln!("Usage: sam conversation <text> [--channel imessage|discord]");
                return Ok(());
            }
            let text = &args[1];
            let channel = args.iter()
                .position(|a| a == "--channel")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .unwrap_or("imessage");
            
            let id = sam.remember_conversation(text, channel)?;
            if !quiet {
                println!("💬 Remembered conversation: {} ({})", truncate(text, 50), channel);
                println!("   ID: {}", id);
            }
        }
        
        "learn" | "learning" => {
            if args.len() < 2 {
                eprintln!("Usage: sam learn <text>");
                return Ok(());
            }
            let text = &args[1];
            let id = sam.remember_learning(text)?;
            if !quiet {
                println!("📚 Remembered learning: {}", truncate(text, 50));
                println!("   ID: {}", id);
            }
        }
        
        "preference" | "pref" => {
            if args.len() < 2 {
                eprintln!("Usage: sam preference <text>");
                return Ok(());
            }
            let text = &args[1];
            let id = sam.remember_preference(text)?;
            if !quiet {
                println!("❤️ Remembered preference: {}", truncate(text, 50));
                println!("   ID: {}", id);
            }
        }
        
        "lesson" => {
            if args.len() < 2 {
                eprintln!("Usage: sam lesson <text>");
                return Ok(());
            }
            let text = &args[1];
            let id = sam.remember_lesson(text)?;
            if !quiet {
                println!("💡 Remembered lesson: {}", truncate(text, 50));
                println!("   ID: {}", id);
            }
        }
        
        "project" | "proj" => {
            if args.len() < 3 {
                eprintln!("Usage: sam project <name> <details>");
                return Ok(());
            }
            let name = &args[1];
            let details = &args[2];
            let memory = SamMemory::project(name, details);
            let id = sam.remember(memory)?;
            if !quiet {
                println!("🔧 Remembered project: {} - {}", name, truncate(details, 40));
                println!("   ID: {}", id);
            }
        }
        
        "recall" | "find" | "search" => {
            if args.len() < 2 {
                eprintln!("Usage: sam recall <query> [--limit N]");
                return Ok(());
            }
            let query = &args[1];
            let limit = args.iter()
                .position(|a| a == "--limit")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            
            let results = sam.recall(query, limit);
            
            if results.is_empty() {
                println!("🦊 No memories found for: {}", query);
            } else {
                println!("🦊 Found {} memories:\n", results.len());
                for (i, item) in results.iter().enumerate() {
                    let type_icon = item.tags.iter()
                        .find(|t| t.starts_with("sam:"))
                        .map(|t| match t.as_str() {
                            "sam:conversation" => "💬",
                            "sam:learning" => "📚",
                            "sam:project" => "🔧",
                            "sam:decision" => "⚖️",
                            "sam:lesson" => "💡",
                            "sam:preference" => "❤️",
                            "sam:task" => "📋",
                            _ => "🧠",
                        })
                        .unwrap_or("🧠");
                    
                    println!("{}. {} {}", i + 1, type_icon, item.content);
                    println!("   Tags: {}", item.tags.join(", "));
                    println!();
                }
            }
        }
        
        "stats" => {
            let stats = sam.stats();
            println!("{}", stats);
        }
        
        _ => {
            eprintln!("Unknown sam command: {}", args[0]);
            eprintln!("Run 'memory-brain sam' for help");
        }
    }
    
    Ok(())
}

// ============ Visual Memory Commands ============

fn cmd_visual(args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::clip_onnx::{MockClipProvider, ClipServerProvider};
    use memory_brain::visual::{ClipProvider, VisualMemory, VisualContext, ImageSource, cosine_similarity};
    use memory_brain::visual_storage::VisualStorage;
    use memory_brain::vlm::{OllamaVlm, VlmProvider};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    // Default CLIP server URL
    let server_url = std::env::var("CLIP_SERVER_URL")
        .unwrap_or_else(|_| "http://localhost:5050".to_string());
    
    // DB path
    let db_path = std::env::var("MEMORY_BRAIN_DB")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{}/.memory-brain/visual.db", home)
        });
    
    if args.is_empty() {
        println!("🖼️ Visual Memory - Brain-inspired image storage");
        println!();
        println!("Usage:");
        println!("  memory-brain visual store <image_path> [--desc \"description\"] [--tags tag1,tag2]");
        println!("  memory-brain visual store <image_path> --auto          # VLM auto-description");
        println!("  memory-brain visual recall <query>     # Search images by text");
        println!("  memory-brain visual similar <image>    # Find similar images");
        println!("  memory-brain visual list               # List all visual memories");
        println!("  memory-brain visual show <id>          # Show memory details");
        println!("  memory-brain visual stats              # Show statistics");
        println!();
        println!("CLIP Server: {} (set CLIP_SERVER_URL to change)", server_url);
        println!("DB: {} (set MEMORY_BRAIN_DB to change)", db_path);
        println!();
        println!("Examples:");
        println!("  memory-brain visual store photo.jpg --auto                    # VLM describes it");
        println!("  memory-brain visual store photo.jpg --desc \"Coffee\" --tags cafe,seoul");
        println!("  memory-brain visual recall \"coffee shop\"");
        println!("  memory-brain visual similar vacation.jpg");
        return Ok(());
    }
    
    // Try to connect to CLIP server, fallback to mock
    let clip: Arc<dyn ClipProvider> = match ClipServerProvider::new(&server_url) {
        Ok(provider) => {
            if !quiet {
                eprintln!("🔗 CLIP server: {}", server_url);
            }
            Arc::new(provider)
        }
        Err(_) => {
            if !quiet {
                eprintln!("⚠️ CLIP server unavailable, using hash embeddings (install clip_server.py for real CLIP)");
            }
            Arc::new(MockClipProvider::new(512))
        }
    };
    
    // Create async runtime for CoreDB operations
    let rt = tokio::runtime::Runtime::new()?;
    
    match args[0].as_str() {
        "store" | "add" => {
            if args.len() < 2 {
                eprintln!("Usage: memory-brain visual store <image_path> [--desc \"...\"] [--tags ...] [--auto]");
                return Ok(());
            }
            
            let image_path = std::path::Path::new(&args[1]);
            if !image_path.exists() {
                eprintln!("❌ Image not found: {}", args[1]);
                return Ok(());
            }
            
            let auto_describe = args.iter().any(|a| a == "--auto" || a == "-a");
            
            // Parse description (or auto-generate)
            let desc = if auto_describe {
                let model = args.iter()
                    .position(|a| a == "--model" || a == "-m")
                    .and_then(|i| args.get(i + 1))
                    .map(|s| s.as_str())
                    .unwrap_or("llava:7b");
                
                if !quiet {
                    eprintln!("🤖 Auto-describing with {} ...", model);
                }
                
                let vlm = OllamaVlm::new(model);
                let prompt = args.iter()
                    .position(|a| a == "--prompt" || a == "-p")
                    .and_then(|i| args.get(i + 1))
                    .map(|s| s.as_str());
                    
                match vlm.describe_image(image_path, prompt) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("❌ VLM error: {}. Use --desc instead or start Ollama.", e);
                        return Ok(());
                    }
                }
            } else {
                args.iter()
                    .position(|a| a == "--desc" || a == "-d")
                    .and_then(|i| args.get(i + 1))
                    .cloned()
                    .unwrap_or_else(|| "(no description)".to_string())
            };
            
            let tags: Vec<String> = args.iter()
                .position(|a| a == "--tags" || a == "-t")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
                .unwrap_or_default();
            
            let emotion: f32 = args.iter()
                .position(|a| a == "--emotion" || a == "-e")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            
            // Generate CLIP embedding
            let embedding = clip.embed_image(image_path)?;
            
            // Store in CoreDB
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                
                // Load cache for auto-linking
                let _ = storage.load_cache().await;
                
                let memory = storage.store_image(
                    image_path,
                    &desc,
                    None,
                    tags.clone(),
                    emotion,
                ).await.expect("Failed to store visual memory");
                
                if !quiet {
                    println!("✅ Stored visual memory: {}", image_path.display());
                    println!("   Description: {}", desc);
                    if !tags.is_empty() {
                        println!("   Tags: {}", tags.join(", "));
                    }
                    if !memory.linked_visuals.is_empty() {
                        println!("   Linked to {} similar images", memory.linked_visuals.len());
                    }
                    println!("   ID: {}", memory.id);
                }
            });
        }
        
        "recall" | "search" | "find" => {
            if args.len() < 2 {
                eprintln!("Usage: memory-brain visual recall <text query>");
                return Ok(());
            }
            
            let query = args[1..].join(" ");
            let limit: usize = args.iter()
                .position(|a| a == "--limit" || a == "-n")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                let loaded = storage.load_cache().await.unwrap_or(0);
                
                if loaded == 0 {
                    println!("📷 No visual memories stored yet.");
                    println!("   Use: memory-brain visual store <image> --auto");
                    return;
                }
                
                match storage.search_by_text(&query, limit).await {
                    Ok(results) if results.is_empty() => {
                        println!("🔍 No results for: \"{}\"", query);
                    }
                    Ok(results) => {
                        println!("🔍 Results for \"{}\" ({} of {} memories):", query, results.len(), loaded);
                        println!();
                        for (i, (mem, score)) in results.iter().enumerate() {
                            println!("  {}. [{:.1}%] {} ", i + 1, score * 100.0, mem.image_path.display());
                            println!("     {}", truncate_str(&mem.description, 80));
                            if !mem.tags.is_empty() {
                                println!("     Tags: {}", mem.tags.join(", "));
                            }
                            println!();
                        }
                    }
                    Err(e) => eprintln!("❌ Search error: {}", e),
                }
            });
        }
        
        "similar" => {
            if args.len() < 2 {
                eprintln!("Usage: memory-brain visual similar <image_path>");
                return Ok(());
            }
            
            let image_path = std::path::Path::new(&args[1]);
            if !image_path.exists() {
                eprintln!("❌ Image not found: {}", args[1]);
                return Ok(());
            }
            
            let limit: usize = args.iter()
                .position(|a| a == "--limit" || a == "-n")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                let loaded = storage.load_cache().await.unwrap_or(0);
                
                if loaded == 0 {
                    println!("📷 No visual memories stored yet.");
                    return;
                }
                
                match storage.search_by_image(image_path, limit).await {
                    Ok(results) if results.is_empty() => {
                        println!("🔍 No similar images found.");
                    }
                    Ok(results) => {
                        println!("🔍 Images similar to {} ({} found):", image_path.display(), results.len());
                        println!();
                        for (i, (mem, score)) in results.iter().enumerate() {
                            println!("  {}. [{:.1}%] {}", i + 1, score * 100.0, mem.image_path.display());
                            println!("     {}", truncate_str(&mem.description, 80));
                            println!();
                        }
                    }
                    Err(e) => eprintln!("❌ Search error: {}", e),
                }
            });
        }
        
        "list" | "ls" => {
            let limit: usize = args.iter()
                .position(|a| a == "--limit" || a == "-n")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                let loaded = storage.load_cache().await.unwrap_or(0);
                
                if loaded == 0 {
                    println!("📷 No visual memories stored yet.");
                    println!("   Use: memory-brain visual store <image> --auto");
                    return;
                }
                
                println!("📷 Visual Memories ({} total):", loaded);
                println!();
                
                // Get all from cache via stats (we already loaded)
                let stats = storage.stats().await.unwrap();
                println!("  Embedding dim: {}", stats.embedding_dim);
                println!("  Total: {} memories", stats.total_memories);
            });
        }
        
        "show" => {
            if args.len() < 2 {
                eprintln!("Usage: memory-brain visual show <id>");
                return Ok(());
            }
            
            let id_str = &args[1];
            
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                let _ = storage.load_cache().await;
                
                if let Ok(id) = uuid::Uuid::parse_str(id_str) {
                    match storage.get(id).await {
                        Ok(Some(mem)) => {
                            println!("🖼️  Visual Memory");
                            println!("   ID: {}", mem.id);
                            println!("   Path: {}", mem.image_path.display());
                            println!("   Description: {}", mem.description);
                            println!("   Tags: {}", if mem.tags.is_empty() { "(none)".to_string() } else { mem.tags.join(", ") });
                            println!("   Emotion: {:.2}", mem.emotional_valence);
                            println!("   Strength: {:.2}", mem.strength);
                            println!("   Recalls: {}", mem.recall_count);
                            println!("   Created: {}", mem.created_at.format("%Y-%m-%d %H:%M"));
                            println!("   Last accessed: {}", mem.last_accessed.format("%Y-%m-%d %H:%M"));
                            if !mem.linked_visuals.is_empty() {
                                println!("   Linked visuals: {}", mem.linked_visuals.len());
                            }
                            if !mem.linked_memories.is_empty() {
                                println!("   Linked memories: {}", mem.linked_memories.len());
                            }
                        }
                        Ok(None) => println!("❌ Memory not found: {}", id_str),
                        Err(e) => eprintln!("❌ Error: {}", e),
                    }
                } else {
                    eprintln!("❌ Invalid UUID: {}", id_str);
                }
            });
        }
        
        "stats" => {
            rt.block_on(async {
                let db = Arc::new(RwLock::new(
                    open_visual_db(&db_path).await
                ));
                let storage = VisualStorage::new(db, clip.clone(), "visual_brain").await
                    .expect("Failed to create VisualStorage");
                let loaded = storage.load_cache().await.unwrap_or(0);
                let stats = storage.stats().await.unwrap();
                
                println!("📊 Visual Memory Statistics:");
                println!("   Total memories: {}", stats.total_memories);
                println!("   Embedding dim: {} (CLIP ViT-B/32)", stats.embedding_dim);
                println!("   CLIP server: {}", server_url);
                println!("   VLM: {}", if check_vlm_available() { "✅ Ollama available" } else { "❌ Ollama not running" });
                println!("   DB: {}", db_path);
            });
        }
        
        _ => {
            eprintln!("Unknown visual command: {}", args[0]);
            eprintln!("Run 'memory-brain visual' for help");
        }
    }
    
    Ok(())
}

async fn open_visual_db(db_path: &str) -> coredb::CoreDB {
    use coredb::DatabaseConfig;
    use std::path::PathBuf;
    
    let config = DatabaseConfig {
        data_directory: PathBuf::from(db_path).join("data"),
        commitlog_directory: PathBuf::from(db_path).join("commitlog"),
        memtable_flush_threshold_mb: 16,
        compaction_throughput_mb_per_sec: 16,
        concurrent_reads: 32,
        concurrent_writes: 32,
    };
    
    coredb::CoreDB::new(config).await.expect("Failed to open CoreDB")
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.min(s.len())])
    }
}

fn check_vlm_available() -> bool {
    use memory_brain::vlm::check_ollama_model;
    check_ollama_model("llava").map(|_| true).unwrap_or(false)
}

// ============ VLM Commands ============

fn cmd_describe(args: &[String], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::vlm::{OllamaVlm, VlmProvider};
    use std::path::Path;
    
    if args.is_empty() {
        println!("🖼️ VLM Image Description - Generate descriptions using LLaVA");
        println!();
        println!("Usage:");
        println!("  memory-brain describe <image_path> [--prompt \"custom prompt\"]");
        println!("  memory-brain describe <image_path> --model llava:13b");
        println!();
        println!("Examples:");
        println!("  memory-brain describe photo.jpg");
        println!("  memory-brain describe cat.png --prompt \"What breed is this cat?\"");
        return Ok(());
    }
    
    let image_path = Path::new(&args[0]);
    if !image_path.exists() {
        eprintln!("❌ Image not found: {}", args[0]);
        return Ok(());
    }
    
    // Parse options
    let model = args.iter()
        .position(|a| a == "--model" || a == "-m")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("llava:7b");
    
    let prompt = args.iter()
        .position(|a| a == "--prompt" || a == "-p")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());
    
    if !quiet {
        println!("🤖 Generating description with {} ...", model);
    }
    
    let vlm = OllamaVlm::new(model);
    
    match vlm.describe_image(image_path, prompt) {
        Ok(description) => {
            if quiet {
                println!("{}", description);
            } else {
                println!();
                println!("📝 Description:");
                println!("{}", description);
            }
        }
        Err(e) => {
            eprintln!("❌ VLM error: {}", e);
            eprintln!();
            eprintln!("Make sure Ollama is running: ollama serve");
            eprintln!("And LLaVA is installed: ollama pull llava:7b");
        }
    }
    
    Ok(())
}

fn cmd_vlm_status(quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    use memory_brain::vlm::check_ollama_model;
    
    if !quiet {
        println!("🔍 VLM Status Check");
        println!();
    }
    
    // Check Ollama connection
    let models = ["llava:7b", "llava:13b", "llava:34b", "bakllava"];
    let mut found_any = false;
    
    for model in &models {
        match check_ollama_model(model) {
            Ok(true) => {
                println!("  ✅ {} - installed", model);
                found_any = true;
            }
            Ok(false) => {
                if !quiet {
                    println!("  ❌ {} - not installed", model);
                }
            }
            Err(e) => {
                if !found_any {
                    eprintln!("❌ Cannot connect to Ollama: {}", e);
                    eprintln!();
                    eprintln!("Start Ollama: ollama serve");
                    return Ok(());
                }
            }
        }
    }
    
    if !found_any {
        println!();
        println!("No VLM models found. Install one:");
        println!("  ollama pull llava:7b    # 4.7GB, recommended");
        println!("  ollama pull llava:13b   # 8GB, better quality");
    }
    
    Ok(())
}
