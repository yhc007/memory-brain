//! CoreDB 기반 영속 저장소
//!
//! SemanticLayer의 인메모리 저장소를 CoreDB로 교체

use anyhow::{Result, Context, bail};
use chrono::{DateTime, Utc, TimeZone};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use coredb::{CoreDB, DatabaseConfig, CassandraValue};
use coredb::query::{QueryResult, Row};
use crate::schema::{Memory, MemoryLink, RelationType};
use crate::types::{MemoryType, Emotion};

/// CoreDB 기반 메모리 저장소
pub struct CoreDBStore {
    db: Arc<RwLock<CoreDB>>,
    keyspace: String,
}

impl CoreDBStore {
    /// 새 CoreDBStore 생성
    pub async fn new(db_path: &str) -> Result<Self> {
        let config = DatabaseConfig {
            data_directory: PathBuf::from(db_path).join("data"),
            commitlog_directory: PathBuf::from(db_path).join("commitlog"),
            memtable_flush_threshold_mb: 16,
            compaction_throughput_mb_per_sec: 16,
            concurrent_reads: 32,
            concurrent_writes: 32,
            block_cache_size_mb: 64,
            block_cache_max_entries: 5_000,
        };

        let db = CoreDB::new(config).await
            .context("Failed to create CoreDB")?;
        
        let store = Self {
            db: Arc::new(RwLock::new(db)),
            keyspace: "memory_brain".to_string(),
        };
        
        store.init_tables().await?;
        Ok(store)
    }
    
    /// 테이블 초기화
    async fn init_tables(&self) -> Result<()> {
        let db = self.db.read().await;
        
        // Keyspace 생성
        let _ = db.execute_cql(&format!(
            "CREATE KEYSPACE {} WITH REPLICATION = {{'class': 'SimpleStrategy', 'replication_factor': 1}}",
            self.keyspace
        )).await;
        
        // memories 테이블
        let _ = db.execute_cql(&format!(
            "CREATE TABLE {}.memories (
                id UUID PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                tags TEXT,
                strength DOUBLE,
                emotion TEXT,
                created_at BIGINT,
                accessed_at BIGINT,
                access_count INT,
                context_id UUID,
                vector_id BIGINT
            )",
            self.keyspace
        )).await;
        
        // memory_links 테이블
        let _ = db.execute_cql(&format!(
            "CREATE TABLE {}.memory_links (
                from_id UUID,
                to_id UUID,
                relation TEXT,
                weight DOUBLE,
                created_at BIGINT,
                PRIMARY KEY (from_id, to_id)
            )",
            self.keyspace
        )).await;
        
        Ok(())
    }
    
    /// 메모리 저장
    pub async fn insert(&self, memory: &Memory) -> Result<()> {
        let db = self.db.read().await;
        
        let tags_json = serde_json::to_string(&memory.tags)?;
        let query = format!(
            "INSERT INTO {}.memories (id, content, memory_type, tags, strength, emotion, created_at, accessed_at, access_count, context_id, vector_id) VALUES ({}, '{}', '{}', '{}', {}, '{}', {}, {}, {}, {}, {})",
            self.keyspace,
            memory.id,
            escape_string(&memory.content),
            format!("{:?}", memory.memory_type),
            escape_string(&tags_json),
            memory.strength,
            format!("{:?}", memory.emotion),
            memory.created_at.timestamp_millis(),
            memory.accessed_at.timestamp_millis(),
            memory.access_count,
            memory.context_id.map(|id| id.to_string()).unwrap_or_else(|| "null".to_string()),
            memory.vector_id.map(|id| id.to_string()).unwrap_or_else(|| "null".to_string()),
        );
        
        db.execute_cql(&query).await
            .context("Failed to insert memory")?;
        
        Ok(())
    }
    
    /// 메모리 조회
    pub async fn get(&self, id: Uuid) -> Result<Option<Memory>> {
        let db = self.db.read().await;
        
        let query = format!(
            "SELECT * FROM {}.memories WHERE id = {}",
            self.keyspace, id
        );
        
        let result = db.execute_cql(&query).await
            .context("Failed to get memory")?;
        
        match result {
            QueryResult::Rows(rows) if !rows.is_empty() => {
                Ok(Some(row_to_memory(&rows[0])?))
            }
            _ => Ok(None),
        }
    }
    
    /// 메모리 업데이트
    pub async fn update(&self, memory: &Memory) -> Result<()> {
        // CoreDB는 INSERT가 UPSERT처럼 동작
        self.insert(memory).await
    }
    
    /// 메모리 삭제
    pub async fn delete(&self, id: Uuid) -> Result<()> {
        let db = self.db.read().await;
        
        let query = format!(
            "DELETE FROM {}.memories WHERE id = {}",
            self.keyspace, id
        );
        
        db.execute_cql(&query).await
            .context("Failed to delete memory")?;
        
        Ok(())
    }
    
    /// 여러 메모리 조회
    pub async fn get_many(&self, ids: &[Uuid]) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();
        for id in ids {
            if let Some(m) = self.get(*id).await? {
                memories.push(m);
            }
        }
        Ok(memories)
    }
    
    /// 링크 저장
    pub async fn insert_link(&self, link: &MemoryLink) -> Result<()> {
        let db = self.db.read().await;
        
        let query = format!(
            "INSERT INTO {}.memory_links (from_id, to_id, relation, weight, created_at) VALUES ({}, {}, '{}', {}, {})",
            self.keyspace,
            link.from_id,
            link.to_id,
            link.relation.as_str(),
            link.weight,
            link.created_at.timestamp_millis(),
        );
        
        db.execute_cql(&query).await
            .context("Failed to insert link")?;
        
        Ok(())
    }
    
    /// 링크 조회
    pub async fn get_links(&self, from_id: Uuid) -> Result<Vec<MemoryLink>> {
        let db = self.db.read().await;
        
        let query = format!(
            "SELECT * FROM {}.memory_links WHERE from_id = {}",
            self.keyspace, from_id
        );
        
        let result = db.execute_cql(&query).await
            .context("Failed to get links")?;
        
        match result {
            QueryResult::Rows(rows) => {
                let links: Vec<MemoryLink> = rows.iter()
                    .filter_map(|row| row_to_link(row).ok())
                    .collect();
                Ok(links)
            }
            _ => Ok(Vec::new()),
        }
    }
    
    /// 링크 삭제
    pub async fn delete_links(&self, from_id: Uuid) -> Result<()> {
        let db = self.db.read().await;
        
        let query = format!(
            "DELETE FROM {}.memory_links WHERE from_id = {}",
            self.keyspace, from_id
        );
        
        db.execute_cql(&query).await
            .context("Failed to delete links")?;
        
        Ok(())
    }
    
    /// 전체 메모리 조회 (주의: 대량 데이터 시 느림)
    pub async fn get_all(&self) -> Result<Vec<Memory>> {
        let db = self.db.read().await;
        
        let query = format!("SELECT * FROM {}.memories", self.keyspace);
        let result = db.execute_cql(&query).await
            .context("Failed to get all memories")?;
        
        match result {
            QueryResult::Rows(rows) => {
                let memories: Vec<Memory> = rows.iter()
                    .filter_map(|row| row_to_memory(row).ok())
                    .collect();
                Ok(memories)
            }
            _ => Ok(Vec::new()),
        }
    }
    
    /// 통계
    pub async fn count(&self) -> Result<u64> {
        let memories = self.get_all().await?;
        Ok(memories.len() as u64)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn escape_string(s: &str) -> String {
    s.replace('\'', "''")
}

fn row_to_memory(row: &Row) -> Result<Memory> {
    let id = get_uuid(&row.columns, "id")?;
    let content = get_string(&row.columns, "content")?;
    let memory_type = parse_memory_type(&get_string(&row.columns, "memory_type")?);
    let tags: Vec<String> = serde_json::from_str(&get_string(&row.columns, "tags").unwrap_or_else(|_| "[]".to_string()))?;
    let strength = get_double(&row.columns, "strength").unwrap_or(1.0) as f32;
    let emotion = parse_emotion(&get_string(&row.columns, "emotion").unwrap_or_else(|_| "Neutral".to_string()));
    let created_at = Utc.timestamp_millis_opt(get_bigint(&row.columns, "created_at")?).unwrap();
    let accessed_at = Utc.timestamp_millis_opt(get_bigint(&row.columns, "accessed_at")?).unwrap();
    let access_count = get_int(&row.columns, "access_count").unwrap_or(1) as u32;
    let context_id = get_uuid(&row.columns, "context_id").ok();
    let vector_id = get_bigint(&row.columns, "vector_id").ok().map(|v| v as u64);
    
    Ok(Memory {
        id,
        content,
        memory_type,
        tags,
        strength,
        emotion,
        created_at,
        accessed_at,
        access_count,
        context_id,
        vector_id,
        embedding_cache: None,
        links_cache: None,
    })
}

fn row_to_link(row: &Row) -> Result<MemoryLink> {
    let from_id = get_uuid(&row.columns, "from_id")?;
    let to_id = get_uuid(&row.columns, "to_id")?;
    let relation = RelationType::from_str(&get_string(&row.columns, "relation")?);
    let weight = get_double(&row.columns, "weight").unwrap_or(1.0) as f32;
    let created_at = Utc.timestamp_millis_opt(get_bigint(&row.columns, "created_at")?).unwrap();
    
    Ok(MemoryLink {
        from_id,
        to_id,
        relation,
        weight,
        created_at,
        metadata: None,
    })
}

fn get_string(cols: &HashMap<String, CassandraValue>, key: &str) -> Result<String> {
    match cols.get(key) {
        Some(CassandraValue::Text(s)) => Ok(s.clone()),
        _ => bail!("Missing or invalid field: {}", key),
    }
}

fn get_uuid(cols: &HashMap<String, CassandraValue>, key: &str) -> Result<Uuid> {
    match cols.get(key) {
        Some(CassandraValue::UUID(u)) => Ok(*u),
        Some(CassandraValue::Text(s)) => Uuid::parse_str(s).context("Invalid UUID"),
        _ => bail!("Missing or invalid UUID: {}", key),
    }
}

fn get_bigint(cols: &HashMap<String, CassandraValue>, key: &str) -> Result<i64> {
    match cols.get(key) {
        Some(CassandraValue::BigInt(n)) => Ok(*n),
        Some(CassandraValue::Int(n)) => Ok(*n as i64),
        Some(CassandraValue::Timestamp(n)) => Ok(*n),
        _ => bail!("Missing or invalid BigInt: {}", key),
    }
}

fn get_int(cols: &HashMap<String, CassandraValue>, key: &str) -> Result<i32> {
    match cols.get(key) {
        Some(CassandraValue::Int(n)) => Ok(*n),
        Some(CassandraValue::BigInt(n)) => Ok(*n as i32),
        _ => bail!("Missing or invalid Int: {}", key),
    }
}

fn get_double(cols: &HashMap<String, CassandraValue>, key: &str) -> Result<f64> {
    match cols.get(key) {
        Some(CassandraValue::Double(n)) => Ok(*n),
        Some(CassandraValue::Text(s)) => s.parse().context("Invalid double"),
        _ => bail!("Missing or invalid Double: {}", key),
    }
}

fn parse_memory_type(s: &str) -> MemoryType {
    match s.to_lowercase().as_str() {
        "working" => MemoryType::Working,
        "episodic" => MemoryType::Episodic,
        "semantic" => MemoryType::Semantic,
        "procedural" => MemoryType::Procedural,
        _ => MemoryType::Episodic,
    }
}

fn parse_emotion(s: &str) -> Emotion {
    match s.to_lowercase().as_str() {
        "positive" => Emotion::Positive,
        "negative" => Emotion::Negative,
        "surprise" => Emotion::Surprise,
        _ => Emotion::Neutral,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("it's"), "it''s");
        assert_eq!(escape_string("a'b'c"), "a''b''c");
    }
    
    #[test]
    fn test_parse_memory_type() {
        assert_eq!(parse_memory_type("Episodic"), MemoryType::Episodic);
        assert_eq!(parse_memory_type("semantic"), MemoryType::Semantic);
        assert_eq!(parse_memory_type("PROCEDURAL"), MemoryType::Procedural);
        assert_eq!(parse_memory_type("unknown"), MemoryType::Episodic);
    }
}
