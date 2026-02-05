use memory_brain::{HnswIndex, HashEmbedder, Embedder};
use uuid::Uuid;

fn main() {
    let embedder = HashEmbedder::new(16);
    let index = HnswIndex::new(16);
    
    // 간단한 테스트 - 3개 벡터
    let v1 = embedder.embed("hello world test");
    let v2 = embedder.embed("hello there friend");  
    let v3 = embedder.embed("goodbye everyone");
    
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();
    
    println!("v1: {:?}", &v1[..4]);
    println!("v2: {:?}", &v2[..4]);
    println!("v3: {:?}", &v3[..4]);
    
    index.add(id1, v1.clone()).unwrap();
    index.add(id2, v2.clone()).unwrap();
    index.add(id3, v3.clone()).unwrap();
    
    // 쿼리
    let query = embedder.embed("hello world test");
    println!("\nQuery: {:?}", &query[..4]);
    
    // HNSW 검색
    let results = index.search(&query, 3);
    println!("\nHNSW Results:");
    for (id, sim) in &results {
        let label = if *id == id1 { "v1 (exact)" } 
            else if *id == id2 { "v2 (similar)" }
            else { "v3 (different)" };
        println!("  {} - sim={:.4}", label, sim);
    }
    
    // Brute force 검색
    println!("\nBrute Force:");
    let sim1 = cosine(&query, &v1);
    let sim2 = cosine(&query, &v2);
    let sim3 = cosine(&query, &v3);
    println!("  v1 (exact):     sim={:.4}", sim1);
    println!("  v2 (similar):   sim={:.4}", sim2);
    println!("  v3 (different): sim={:.4}", sim3);
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}
