//! Sam's Brain 예제
//!
//! Sam의 기억 시스템 테스트

use memory_brain::{SamBrain, SamMemory};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦊 Sam's Brain 테스트\n");

    let dir = tempdir()?;
    let db_path = dir.path().join("sam.db");
    let mut brain = SamBrain::new(db_path.to_str().unwrap())?;

    // 1. Paul의 선호도 기억
    println!("❤️ Paul의 선호도 저장...");
    brain.remember_preference("Paul은 반말 선호, 친구처럼 대화")?;
    brain.remember_preference("Paul의 GitHub: yhc007")?;
    brain.remember_preference("Paul의 주 언어: Rust 🦀")?;

    // 2. 학습 내용 저장
    println!("📚 학습 내용 저장...");
    brain.remember_learning("CoreDB는 Cassandra 스타일 NoSQL DB")?;
    brain.remember_learning("HNSW는 O(log n) 근사 최근접 이웃 검색")?;
    brain.remember_learning("memory-brain은 인간 뇌 구조 모방")?;

    // 3. 교훈 저장
    println!("💡 교훈 저장...");
    brain.remember_lesson("truncate 함수에서 유니코드 경계 주의")?;
    brain.remember_lesson("항상 커밋 전에 테스트 돌리기")?;

    // 4. 대화 기억
    println!("💬 대화 저장...");
    brain.remember_conversation("2026-01-30: CoreDB + Pekko Actor 통합 성공!", "imessage")?;
    brain.remember_conversation("memory-brain 성능 850x 향상", "imessage")?;

    // 5. 프로젝트 컨텍스트
    println!("🔧 프로젝트 저장...");
    let project = SamMemory::project("memory-brain", "인간 뇌 구조 모방 메모리 시스템");
    brain.remember(project)?;
    let project = SamMemory::project("CoreDB", "Rust로 만든 Cassandra 스타일 NoSQL");
    brain.remember(project)?;

    // 통계
    println!("\n{}\n", brain.stats());

    // 6. Recall 테스트
    println!("🔍 Recall 테스트:");
    
    println!("\n  Query: 'Paul'");
    let results = brain.recall("Paul", 3);
    for item in results.iter().take(3) {
        println!("    - {}", item.content);
    }

    println!("\n  Query: 'Rust'");
    let results = brain.recall("Rust", 3);
    for item in results.iter().take(3) {
        println!("    - {}", item.content);
    }

    println!("\n  Query: '성능'");
    let results = brain.recall("성능", 3);
    for item in results.iter().take(3) {
        println!("    - {}", item.content);
    }

    // 7. Fast recall (HNSW)
    println!("\n⚡ Fast Recall (HNSW):");
    let results = brain.fast_recall("memory", 5);
    println!("  Found {} similar memories", results.len());

    println!("\n✅ Sam's Brain 테스트 완료! 🦊");
    Ok(())
}
