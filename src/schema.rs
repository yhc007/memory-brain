//! Semantic Schema Layer - CoreDB + CoreVecDB 통합 스키마
//!
//! 이 레이어는 구조화된 데이터(CoreDB)와 벡터 데이터(CoreVecDB)를
//! 통합하여 메모리 시스템의 단일 인터페이스를 제공합니다.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::types::{MemoryType, Emotion};

// ============================================================================
// Core Schema Types
// ============================================================================

/// 메모리 관계 타입
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    /// 의미적으로 유사한 기억
    Similar,
    /// 인과 관계 (A가 B를 유발)
    Causes,
    /// 부분-전체 관계 (A는 B의 일부)
    PartOf,
    /// 시간적 순서 (A 다음에 B)
    Temporal,
    /// 반대/대조 관계
    Contrasts,
    /// 사용자 정의 관계
    Custom(String),
}

impl RelationType {
    pub fn as_str(&self) -> &str {
        match self {
            RelationType::Similar => "similar",
            RelationType::Causes => "causes",
            RelationType::PartOf => "part_of",
            RelationType::Temporal => "temporal",
            RelationType::Contrasts => "contrasts",
            RelationType::Custom(s) => s.as_str(),
        }
    }
    
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "similar" => RelationType::Similar,
            "causes" => RelationType::Causes,
            "part_of" | "partof" => RelationType::PartOf,
            "temporal" => RelationType::Temporal,
            "contrasts" => RelationType::Contrasts,
            other => RelationType::Custom(other.to_string()),
        }
    }
}

/// 메모리 링크 (관계)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    /// 출발 메모리 ID
    pub from_id: Uuid,
    /// 도착 메모리 ID
    pub to_id: Uuid,
    /// 관계 타입
    pub relation: RelationType,
    /// 관계 강도 (0.0 - 1.0)
    pub weight: f32,
    /// 생성 시간
    pub created_at: DateTime<Utc>,
    /// 메타데이터
    pub metadata: Option<HashMap<String, String>>,
}

impl MemoryLink {
    pub fn new(from_id: Uuid, to_id: Uuid, relation: RelationType) -> Self {
        Self {
            from_id,
            to_id,
            relation,
            weight: 1.0,
            created_at: Utc::now(),
            metadata: None,
        }
    }
    
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// 컨텍스트 정보 (메모리가 생성된 맥락)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub id: Uuid,
    /// 세션 ID (대화 세션)
    pub session_id: Option<String>,
    /// 소스 (예: "chat", "document", "observation")
    pub source: String,
    /// 위치 정보
    pub location: Option<String>,
    /// 참여자들
    pub participants: Vec<String>,
    /// 시작 시간
    pub started_at: DateTime<Utc>,
    /// 종료 시간
    pub ended_at: Option<DateTime<Utc>>,
    /// 추가 메타데이터
    pub metadata: HashMap<String, String>,
}

impl MemoryContext {
    pub fn new(source: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: None,
            source: source.to_string(),
            location: None,
            participants: Vec::new(),
            started_at: Utc::now(),
            ended_at: None,
            metadata: HashMap::new(),
        }
    }
}

/// 통합 메모리 스키마 (CoreDB + CoreVecDB 참조)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    // === CoreDB에 저장되는 필드 ===
    /// 고유 식별자
    pub id: Uuid,
    /// 메모리 내용
    pub content: String,
    /// 메모리 타입 (Episodic, Semantic, Procedural)
    pub memory_type: MemoryType,
    /// 태그들
    pub tags: Vec<String>,
    /// 기억 강도 (0.0 - 1.0)
    pub strength: f32,
    /// 감정
    pub emotion: Emotion,
    /// 생성 시간
    pub created_at: DateTime<Utc>,
    /// 마지막 접근 시간
    pub accessed_at: DateTime<Utc>,
    /// 접근 횟수
    pub access_count: u32,
    /// 컨텍스트 ID (선택적)
    pub context_id: Option<Uuid>,
    
    // === CoreVecDB 참조 ===
    /// CoreVecDB의 벡터 ID
    pub vector_id: Option<u64>,
    
    // === 메모리 내 캐시 (DB에 저장 안 함) ===
    /// 임베딩 벡터 (캐시용)
    #[serde(skip)]
    pub embedding_cache: Option<Vec<f32>>,
    /// 연결된 링크들 (캐시용)
    #[serde(skip)]
    pub links_cache: Option<Vec<MemoryLink>>,
}

impl Memory {
    pub fn new(content: &str, memory_type: MemoryType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content: content.to_string(),
            memory_type,
            tags: Vec::new(),
            strength: 1.0,
            emotion: Emotion::Neutral,
            created_at: now,
            accessed_at: now,
            access_count: 1,
            context_id: None,
            vector_id: None,
            embedding_cache: None,
            links_cache: None,
        }
    }
    
    /// Builder: 태그 추가
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Builder: 감정 설정
    pub fn with_emotion(mut self, emotion: Emotion) -> Self {
        self.emotion = emotion.clone();
        // 감정적 기억은 더 강함
        if !matches!(emotion, Emotion::Neutral) {
            self.strength = (self.strength * 1.5).min(1.0);
        }
        self
    }
    
    /// Builder: 컨텍스트 설정
    pub fn with_context(mut self, context_id: Uuid) -> Self {
        self.context_id = Some(context_id);
        self
    }
    
    /// Builder: 벡터 ID 설정
    pub fn with_vector_id(mut self, vector_id: u64) -> Self {
        self.vector_id = Some(vector_id);
        self
    }
    
    /// 메모리 접근 (강화)
    pub fn access(&mut self) {
        self.accessed_at = Utc::now();
        self.access_count += 1;
        self.strength = (self.strength + 0.1).min(1.0);
    }
    
    /// 메모리 약화 (망각)
    pub fn decay(&mut self, factor: f32) {
        self.strength *= factor;
    }
    
    /// 잊혀졌는지 확인
    pub fn is_forgotten(&self) -> bool {
        self.strength < 0.1
    }
    
    /// 관련성 점수 계산
    pub fn relevance_score(&self) -> f32 {
        let recency = self.recency_factor();
        let frequency = (self.access_count as f32).ln() / 10.0;
        self.strength * 0.5 + recency * 0.3 + frequency * 0.2
    }
    
    fn recency_factor(&self) -> f32 {
        let hours_since = (Utc::now() - self.accessed_at).num_hours() as f32;
        (-hours_since / 168.0).exp() // 반감기 ~1주일
    }
}

// ============================================================================
// CQL Schema Definitions (CoreDB용)
// ============================================================================

/// CoreDB 테이블 생성 CQL
pub mod cql {
    pub const CREATE_KEYSPACE: &str = r#"
        CREATE KEYSPACE IF NOT EXISTS memory_brain
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    "#;
    
    pub const CREATE_MEMORIES_TABLE: &str = r#"
        CREATE TABLE IF NOT EXISTS memory_brain.memories (
            id UUID PRIMARY KEY,
            content TEXT,
            memory_type TEXT,
            tags LIST<TEXT>,
            strength DOUBLE,
            emotion TEXT,
            created_at TIMESTAMP,
            accessed_at TIMESTAMP,
            access_count INT,
            context_id UUID,
            vector_id BIGINT
        )
    "#;
    
    pub const CREATE_MEMORY_LINKS_TABLE: &str = r#"
        CREATE TABLE IF NOT EXISTS memory_brain.memory_links (
            from_id UUID,
            to_id UUID,
            relation TEXT,
            weight DOUBLE,
            created_at TIMESTAMP,
            metadata MAP<TEXT, TEXT>,
            PRIMARY KEY (from_id, to_id)
        )
    "#;
    
    pub const CREATE_CONTEXTS_TABLE: &str = r#"
        CREATE TABLE IF NOT EXISTS memory_brain.contexts (
            id UUID PRIMARY KEY,
            session_id TEXT,
            source TEXT,
            location TEXT,
            participants LIST<TEXT>,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            metadata MAP<TEXT, TEXT>
        )
    "#;
    
    // === 인덱스 ===
    pub const CREATE_MEMORY_TYPE_INDEX: &str = r#"
        CREATE INDEX IF NOT EXISTS ON memory_brain.memories (memory_type)
    "#;
    
    pub const CREATE_CONTEXT_ID_INDEX: &str = r#"
        CREATE INDEX IF NOT EXISTS ON memory_brain.memories (context_id)
    "#;
}

// ============================================================================
// Semantic Store Trait
// ============================================================================

use async_trait::async_trait;

/// 통합 메모리 저장소 인터페이스
#[async_trait]
pub trait SemanticStore: Send + Sync {
    /// 메모리 저장 (CoreDB + CoreVecDB 동시 저장)
    async fn store(&self, memory: Memory, embedding: Vec<f32>) -> anyhow::Result<Uuid>;
    
    /// 시맨틱 검색 (벡터 유사도 기반)
    async fn recall(&self, query: &str, k: usize) -> anyhow::Result<Vec<Memory>>;
    
    /// ID로 메모리 조회
    async fn get(&self, id: Uuid) -> anyhow::Result<Option<Memory>>;
    
    /// 여러 ID로 메모리 조회
    async fn get_many(&self, ids: Vec<Uuid>) -> anyhow::Result<Vec<Memory>>;
    
    /// 메모리 업데이트
    async fn update(&self, memory: &Memory) -> anyhow::Result<()>;
    
    /// 메모리 삭제
    async fn delete(&self, id: Uuid) -> anyhow::Result<()>;
    
    /// 메모리 간 링크 생성
    async fn link(&self, from: Uuid, to: Uuid, relation: RelationType, weight: f32) -> anyhow::Result<()>;
    
    /// 특정 메모리의 링크 조회
    async fn get_links(&self, memory_id: Uuid) -> anyhow::Result<Vec<MemoryLink>>;
    
    /// 메모리 강화 (접근 시)
    async fn strengthen(&self, id: Uuid) -> anyhow::Result<()>;
    
    /// 전체 메모리 약화 (망각 시뮬레이션)
    async fn decay_all(&self, factor: f32) -> anyhow::Result<u64>;
    
    /// 통계
    async fn stats(&self) -> anyhow::Result<StoreStats>;
}

/// 저장소 통계
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub total_memories: u64,
    pub total_links: u64,
    pub total_vectors: u64,
    pub memories_by_type: HashMap<String, u64>,
    pub avg_strength: f32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let memory = Memory::new("테스트 기억", MemoryType::Episodic)
            .with_tags(vec!["test".to_string(), "rust".to_string()])
            .with_emotion(Emotion::Positive);
        
        assert_eq!(memory.content, "테스트 기억");
        assert_eq!(memory.tags.len(), 2);
        assert!(memory.strength > 1.0 - 0.01); // 감정으로 강화됨
    }

    #[test]
    fn test_memory_access_strengthens() {
        let mut memory = Memory::new("테스트", MemoryType::Semantic);
        memory.strength = 0.5;
        
        memory.access();
        
        assert_eq!(memory.access_count, 2);
        assert!(memory.strength > 0.5);
    }

    #[test]
    fn test_memory_decay() {
        let mut memory = Memory::new("테스트", MemoryType::Semantic);
        
        memory.decay(0.5);
        
        assert!((memory.strength - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_link_creation() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        let link = MemoryLink::new(id1, id2, RelationType::Similar)
            .with_weight(0.8);
        
        assert_eq!(link.from_id, id1);
        assert_eq!(link.to_id, id2);
        assert!((link.weight - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_relation_type_conversion() {
        assert_eq!(RelationType::from_str("similar"), RelationType::Similar);
        assert_eq!(RelationType::from_str("CAUSES"), RelationType::Causes);
        assert_eq!(RelationType::from_str("custom_rel"), RelationType::Custom("custom_rel".to_string()));
    }
}
