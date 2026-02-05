//! HNSW vs Brute-Force 벤치마크

use memory_brain::{HnswIndex, HashEmbedder, Embedder};
use std::time::Instant;
use uuid::Uuid;

fn main() {
    println!("🔍 HNSW vs Brute-Force 벤치마크\n");

    let embedder = HashEmbedder::new(256);
    let index = HnswIndex::new(256);
    
    // Generate test data
    println!("📦 테스트 데이터 생성 (10,000개)...");
    let mut vectors: Vec<(Uuid, Vec<f32>)> = Vec::new();
    for i in 0..10_000 {
        let id = Uuid::new_v4();
        let text = format!("메모리 항목 {} 테스트 데이터 콘텐츠", i);
        let embedding = embedder.embed(&text);
        vectors.push((id, embedding));
    }

    // Build index
    println!("🏗️  HNSW 인덱스 구축...");
    let start = Instant::now();
    for (id, embedding) in &vectors {
        index.add(*id, embedding.clone()).unwrap();
    }
    let build_time = start.elapsed();
    println!("   구축 시간: {:?} ({:.0} items/sec)\n", 
        build_time, 10_000.0 / build_time.as_secs_f64());

    // Generate query
    let query = embedder.embed("메모리 테스트 데이터");

    // HNSW search benchmark
    println!("⚡ HNSW 검색 (1,000회)...");
    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = index.search(&query, 10);
    }
    let hnsw_time = start.elapsed();
    println!("   HNSW: {:?} ({:.0} searches/sec)", 
        hnsw_time, 1_000.0 / hnsw_time.as_secs_f64());

    // Brute-force search benchmark
    println!("🐢 Brute-Force 검색 (1,000회)...");
    let start = Instant::now();
    for _ in 0..1_000 {
        let mut results: Vec<(Uuid, f32)> = vectors
            .iter()
            .map(|(id, v)| {
                let sim = cosine_similarity(&query, v);
                (*id, sim)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(10);
    }
    let brute_time = start.elapsed();
    println!("   Brute-Force: {:?} ({:.0} searches/sec)", 
        brute_time, 1_000.0 / brute_time.as_secs_f64());

    // Comparison
    let speedup = brute_time.as_secs_f64() / hnsw_time.as_secs_f64();
    println!("\n🚀 HNSW가 {:.1}x 빠름!", speedup);

    // Verify accuracy
    println!("\n📊 정확도 검증...");
    let hnsw_results = index.search(&query, 10);
    let mut brute_results: Vec<(Uuid, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, cosine_similarity(&query, v)))
        .collect();
    brute_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    brute_results.truncate(10);

    // Count unique similarities
    let hnsw_sims: std::collections::HashSet<u32> = hnsw_results.iter()
        .map(|(_, s)| (s * 10000.0) as u32).collect();
    let brute_sims: std::collections::HashSet<u32> = brute_results.iter()
        .map(|(_, s)| (s * 10000.0) as u32).collect();
    
    println!("   HNSW Top-1 sim:  {:.4}", hnsw_results.first().map(|(_, s)| *s).unwrap_or(0.0));
    println!("   Brute Top-1 sim: {:.4}", brute_results.first().map(|(_, s)| *s).unwrap_or(0.0));
    
    // Check if top-1 similarity matches
    let hnsw_top1 = hnsw_results.first().map(|(_, s)| (s * 10000.0) as u32);
    let brute_top1 = brute_results.first().map(|(_, s)| (s * 10000.0) as u32);
    if hnsw_top1 == brute_top1 {
        println!("   ✅ Top-1 similarity 일치!");
    }
    
    // Check similarity overlap (same sim values found)
    let sim_overlap = hnsw_sims.intersection(&brute_sims).count();
    println!("   Similarity 값 일치: {}/10", sim_overlap);

    println!("\n✅ 벤치마크 완료!");
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}
