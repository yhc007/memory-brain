//! SemanticStore 구현체 - CoreDB + CoreVecDB 통합 저장소
//!
//! 이 모듈은 SemanticStore trait의 실제 구현을 제공합니다.
//! CoreDB (구조화 데이터)와 CoreVecDB (벡터)를 동기화하여 관리합니다.

use anyhow::{Result, Context, bail};
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::schema::{Memory, MemoryLink, RelationType, SemanticStore, StoreStats};
use crate::types::MemoryType;
use crate::embedding::Embedder;

// ============================================================================
// CoreVecDB Client
// ============================================================================

/// CoreVecDB HTTP 클라이언트
#[derive(Clone)]
pub struct VecDBClient {
    base_url: String,
    collection: String,
}

impl VecDBClient {
    pub fn new(base_url: &str, collection: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            collection: collection.to_string(),
        }
    }
    
    /// 벡터 삽입
    pub async fn insert(&self, id: &str, vector: Vec<f32>, payload: serde_json::Value) -> Result<u64> {
        let url = format!("{}/collections/{}/points", self.base_url, self.collection);
        
        let body = serde_json::json!({
            "points": [{
                "id": id,
                "vector": vector,
                "payload": payload
            }]
        });
        
        let response: serde_json::Value = ureq::post(&url)
            .send_json(&body)
            .context("VecDB insert failed")?
            .into_json()
            .context("Failed to parse VecDB response")?;
        
        // VecDB returns the internal ID
        let vec_id = response["ids"][0].as_u64().unwrap_or(0);
        Ok(vec_id)
    }
    
    /// 벡터 검색
    pub async fn search(&self, vector: Vec<f32>, k: usize) -> Result<Vec<SearchResult>> {
        let url = format!("{}/collections/{}/search", self.base_url, self.collection);
        
        let body = serde_json::json!({
            "vector": vector,
            "top_k": k,
            "with_payload": true
        });
        
        let response: serde_json::Value = ureq::post(&url)
            .send_json(&body)
            .context("VecDB search failed")?
            .into_json()
            .context("Failed to parse search response")?;
        
        let results = response["results"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        Some(SearchResult {
                            id: r["payload"]["id"].as_str()?.to_string(),
                            score: r["score"].as_f64()? as f32,
                            payload: r["payload"].clone(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        
        Ok(results)
    }
    
    /// 벡터 삭제
    pub async fn delete(&self, id: &str) -> Result<()> {
        let url = format!("{}/collections/{}/points/{}", self.base_url, self.collection, id);
        
        ureq::delete(&url)
            .call()
            .context("VecDB delete failed")?;
        
        Ok(())
    }
    
    /// 컬렉션 통계
    pub async fn stats(&self) -> Result<u64> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        
        let response: serde_json::Value = ureq::get(&url)
            .call()
            .context("VecDB stats failed")?
            .into_json()
            .context("Failed to parse stats response")?;
        
        Ok(response["vectors_count"].as_u64().unwrap_or(0))
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: serde_json::Value,
}

// ============================================================================
// In-Memory Store (CoreDB 연동 전 임시)
// ============================================================================

/// 메모리 기반 저장소 (CoreDB 연동 전 임시 구현)
struct MemoryStore {
    memories: HashMap<Uuid, Memory>,
    links: HashMap<Uuid, Vec<MemoryLink>>,
}

impl MemoryStore {
    fn new() -> Self {
        Self {
            memories: HashMap::new(),
            links: HashMap::new(),
        }
    }
}

// ============================================================================
// SemanticLayer 구현
// ============================================================================

/// SemanticStore의 실제 구현체
pub struct SemanticLayer {
    /// CoreVecDB 클라이언트
    vecdb: VecDBClient,
    /// 임베딩 모델
    embedder: Arc<dyn Embedder>,
    /// 인메모리 저장소 (CoreDB 연동 전 임시)
    store: Arc<RwLock<MemoryStore>>,
}

impl SemanticLayer {
    /// 새 SemanticLayer 생성
    pub fn new(
        vecdb_url: &str,
        collection: &str,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            vecdb: VecDBClient::new(vecdb_url, collection),
            embedder,
            store: Arc::new(RwLock::new(MemoryStore::new())),
        }
    }
    
    /// 임베딩 생성
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embedder.embed(text)
    }
}

#[async_trait]
impl SemanticStore for SemanticLayer {
    async fn store(&self, mut memory: Memory, embedding: Vec<f32>) -> Result<Uuid> {
        let id = memory.id;
        
        // 1. CoreVecDB에 벡터 저장
        let payload = serde_json::json!({
            "id": id.to_string(),
            "memory_type": format!("{:?}", memory.memory_type),
            "tags": memory.tags,
            "strength": memory.strength,
            "created_at": memory.created_at.to_rfc3339(),
        });
        
        let vec_id = self.vecdb.insert(&id.to_string(), embedding.clone(), payload).await?;
        memory.vector_id = Some(vec_id);
        memory.embedding_cache = Some(embedding);
        
        // 2. 인메모리 저장소에 저장 (나중에 CoreDB로 교체)
        let mut store = self.store.write().await;
        store.memories.insert(id, memory);
        
        Ok(id)
    }
    
    async fn recall(&self, query: &str, k: usize) -> Result<Vec<Memory>> {
        // 1. 쿼리 임베딩
        let query_vec = self.embed(query);
        
        // 2. CoreVecDB 검색
        let results = self.vecdb.search(query_vec, k).await?;
        
        // 3. ID로 메모리 조회
        let store = self.store.read().await;
        let memories: Vec<Memory> = results
            .iter()
            .filter_map(|r| {
                Uuid::parse_str(&r.id).ok()
                    .and_then(|id| store.memories.get(&id).cloned())
            })
            .collect();
        
        Ok(memories)
    }
    
    async fn get(&self, id: Uuid) -> Result<Option<Memory>> {
        let store = self.store.read().await;
        Ok(store.memories.get(&id).cloned())
    }
    
    async fn get_many(&self, ids: Vec<Uuid>) -> Result<Vec<Memory>> {
        let store = self.store.read().await;
        let memories: Vec<Memory> = ids
            .iter()
            .filter_map(|id| store.memories.get(id).cloned())
            .collect();
        Ok(memories)
    }
    
    async fn update(&self, memory: &Memory) -> Result<()> {
        let mut store = self.store.write().await;
        store.memories.insert(memory.id, memory.clone());
        Ok(())
    }
    
    async fn delete(&self, id: Uuid) -> Result<()> {
        // 1. CoreVecDB에서 삭제
        self.vecdb.delete(&id.to_string()).await?;
        
        // 2. 인메모리에서 삭제
        let mut store = self.store.write().await;
        store.memories.remove(&id);
        store.links.remove(&id);
        
        Ok(())
    }
    
    async fn link(&self, from: Uuid, to: Uuid, relation: RelationType, weight: f32) -> Result<()> {
        let link = MemoryLink::new(from, to, relation).with_weight(weight);
        
        let mut store = self.store.write().await;
        store.links
            .entry(from)
            .or_insert_with(Vec::new)
            .push(link);
        
        Ok(())
    }
    
    async fn get_links(&self, memory_id: Uuid) -> Result<Vec<MemoryLink>> {
        let store = self.store.read().await;
        Ok(store.links.get(&memory_id).cloned().unwrap_or_default())
    }
    
    async fn strengthen(&self, id: Uuid) -> Result<()> {
        let mut store = self.store.write().await;
        if let Some(memory) = store.memories.get_mut(&id) {
            memory.access();
        }
        Ok(())
    }
    
    async fn decay_all(&self, factor: f32) -> Result<u64> {
        let mut store = self.store.write().await;
        let mut count = 0u64;
        
        for memory in store.memories.values_mut() {
            memory.decay(factor);
            count += 1;
        }
        
        // 잊혀진 기억 제거
        let forgotten: Vec<Uuid> = store.memories
            .iter()
            .filter(|(_, m)| m.is_forgotten())
            .map(|(id, _)| *id)
            .collect();
        
        for id in &forgotten {
            store.memories.remove(id);
            store.links.remove(id);
            // VecDB에서도 삭제 (비동기로 처리)
            let _ = self.vecdb.delete(&id.to_string()).await;
        }
        
        Ok(count)
    }
    
    async fn stats(&self) -> Result<StoreStats> {
        let store = self.store.read().await;
        let vec_count = self.vecdb.stats().await.unwrap_or(0);
        
        let mut by_type: HashMap<String, u64> = HashMap::new();
        let mut total_strength = 0.0f32;
        
        for memory in store.memories.values() {
            let type_str = format!("{:?}", memory.memory_type);
            *by_type.entry(type_str).or_insert(0) += 1;
            total_strength += memory.strength;
        }
        
        let total = store.memories.len() as u64;
        let avg_strength = if total > 0 {
            total_strength / total as f32
        } else {
            0.0
        };
        
        Ok(StoreStats {
            total_memories: total,
            total_links: store.links.values().map(|v| v.len() as u64).sum(),
            total_vectors: vec_count,
            memories_by_type: by_type,
            avg_strength,
        })
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// SemanticLayer 빌더
pub struct SemanticLayerBuilder {
    vecdb_url: String,
    collection: String,
    embedder: Option<Arc<dyn Embedder>>,
}

impl SemanticLayerBuilder {
    pub fn new() -> Self {
        Self {
            vecdb_url: "http://localhost:3100".to_string(),
            collection: "memories".to_string(),
            embedder: None,
        }
    }
    
    pub fn vecdb_url(mut self, url: &str) -> Self {
        self.vecdb_url = url.to_string();
        self
    }
    
    pub fn collection(mut self, name: &str) -> Self {
        self.collection = name.to_string();
        self
    }
    
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }
    
    pub fn build(self) -> Result<SemanticLayer> {
        let embedder = self.embedder
            .context("Embedder is required")?;
        
        Ok(SemanticLayer::new(&self.vecdb_url, &self.collection, embedder))
    }
}

impl Default for SemanticLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::HashEmbedder;
    
    fn make_test_layer() -> SemanticLayer {
        let embedder = Arc::new(HashEmbedder::new(128));
        SemanticLayer::new("http://localhost:3100", "test", embedder)
    }
    
    #[tokio::test]
    async fn test_store_stats_empty() {
        let layer = make_test_layer();
        let stats = layer.stats().await.unwrap();
        
        assert_eq!(stats.total_memories, 0);
        assert_eq!(stats.total_links, 0);
    }
    
    #[tokio::test]
    async fn test_link_memories() {
        let layer = make_test_layer();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        layer.link(id1, id2, RelationType::Similar, 0.9).await.unwrap();
        
        let links = layer.get_links(id1).await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].to_id, id2);
    }
}
