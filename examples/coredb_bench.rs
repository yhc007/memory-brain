//! CoreDB vs SQLite 벤치마크

use memory_brain::{MemoryItem, MemoryType};
use std::time::Instant;

#[cfg(feature = "coredb-backend")]
use memory_brain::CoreDBStorage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "coredb-backend"))]
    {
        println!("❌ coredb-backend feature가 필요합니다");
        println!("   cargo run --features coredb-backend --example coredb_bench");
        return Ok(());
    }

    #[cfg(feature = "coredb-backend")]
    {
        println!("🧪 CoreDB Storage 벤치마크\n");
        
        let dir = tempfile::tempdir()?;
        let storage = CoreDBStorage::new(
            dir.path().to_str().unwrap(),
            "bench",
            "memories"
        ).await?;

        // 1. 단일 저장 벤치마크
        println!("📝 단일 저장 테스트 (100개)...");
        let start = Instant::now();
        for i in 0..100 {
            let item = MemoryItem::new(&format!("테스트 메모리 {} 데이터", i), None);
            storage.save(&item).await?;
        }
        let elapsed = start.elapsed();
        println!("   100 saves: {:?} ({:.0} ops/sec)\n", elapsed, 100.0 / elapsed.as_secs_f64());

        // 2. 배치 저장 벤치마크
        println!("📦 배치 저장 테스트 (1000개)...");
        let items: Vec<MemoryItem> = (0..1000)
            .map(|i| MemoryItem::new(&format!("배치 메모리 {} 데이터", i), None))
            .collect();
        
        let start = Instant::now();
        let count = storage.save_batch(&items).await?;
        let elapsed = start.elapsed();
        println!("   {} saves: {:?} ({:.0} ops/sec)\n", count, elapsed, count as f64 / elapsed.as_secs_f64());

        // 3. 검색 벤치마크
        println!("🔍 검색 테스트...");
        let start = Instant::now();
        for _ in 0..100 {
            let _ = storage.search("메모리", 10).await?;
        }
        let elapsed = start.elapsed();
        println!("   100 searches: {:?} ({:.0} ops/sec)\n", elapsed, 100.0 / elapsed.as_secs_f64());

        // 4. 전체 조회
        println!("📊 전체 조회...");
        let start = Instant::now();
        let all = storage.get_all().await?;
        let elapsed = start.elapsed();
        println!("   {} items loaded in {:?}\n", all.len(), elapsed);

        println!("✅ 벤치마크 완료!");
    }
    Ok(())
}
