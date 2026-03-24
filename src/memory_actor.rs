//! MemoryActor - pekko-actor 기반 메모리 액터
//!
//! SemanticLayer를 Actor로 감싸서 비동기 메시지 기반 인터페이스 제공

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::schema::{Memory, MemoryLink, RelationType, SemanticStore, StoreStats};
use crate::semantic_store::SemanticLayer;
use crate::types::MemoryType;

// ============================================================================
// Actor Messages
// ============================================================================

/// MemoryActor가 처리하는 메시지 타입
#[derive(Debug)]
pub enum MemoryMessage {
    /// 메모리 저장
    Store {
        memory: Memory,
        embedding: Vec<f32>,
        reply: oneshot::Sender<Result<Uuid, String>>,
    },
    
    /// 시맨틱 검색
    Recall {
        query: String,
        k: usize,
        reply: oneshot::Sender<Result<Vec<Memory>, String>>,
    },
    
    /// ID로 메모리 조회
    Get {
        id: Uuid,
        reply: oneshot::Sender<Result<Option<Memory>, String>>,
    },
    
    /// 메모리 삭제
    Delete {
        id: Uuid,
        reply: oneshot::Sender<Result<(), String>>,
    },
    
    /// 메모리 간 링크 생성
    Link {
        from: Uuid,
        to: Uuid,
        relation: RelationType,
        weight: f32,
        reply: oneshot::Sender<Result<(), String>>,
    },
    
    /// 특정 메모리의 링크 조회
    GetLinks {
        memory_id: Uuid,
        reply: oneshot::Sender<Result<Vec<MemoryLink>, String>>,
    },
    
    /// 메모리 강화
    Strengthen {
        id: Uuid,
        reply: oneshot::Sender<Result<(), String>>,
    },
    
    /// 전체 약화 (망각)
    DecayAll {
        factor: f32,
        reply: oneshot::Sender<Result<u64, String>>,
    },
    
    /// 통계 조회
    Stats {
        reply: oneshot::Sender<Result<StoreStats, String>>,
    },
    
    /// 액터 종료
    Shutdown,
}

// ============================================================================
// MemoryActor
// ============================================================================

/// SemanticLayer를 감싸는 Actor
pub struct MemoryActor {
    store: Arc<SemanticLayer>,
    receiver: mpsc::Receiver<MemoryMessage>,
}

impl MemoryActor {
    /// 새 MemoryActor 생성
    fn new(store: Arc<SemanticLayer>, receiver: mpsc::Receiver<MemoryMessage>) -> Self {
        Self { store, receiver }
    }
    
    /// 메시지 처리 루프 실행
    pub async fn run(mut self) {
        eprintln!("[MemoryActor] started");
        
        while let Some(msg) = self.receiver.recv().await {
            match msg {
                MemoryMessage::Shutdown => {
                    eprintln!("[MemoryActor] shutting down");
                    break;
                }
                _ => self.handle_message(msg).await,
            }
        }
        
        eprintln!("[MemoryActor] stopped");
    }
    
    /// 개별 메시지 처리
    async fn handle_message(&self, msg: MemoryMessage) {
        match msg {
            MemoryMessage::Store { memory, embedding, reply } => {
                let result = self.store.store(memory, embedding).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Recall { query, k, reply } => {
                let result = self.store.recall(&query, k).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Get { id, reply } => {
                let result = self.store.get(id).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Delete { id, reply } => {
                let result = self.store.delete(id).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Link { from, to, relation, weight, reply } => {
                let result = self.store.link(from, to, relation, weight).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::GetLinks { memory_id, reply } => {
                let result = self.store.get_links(memory_id).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Strengthen { id, reply } => {
                let result = self.store.strengthen(id).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::DecayAll { factor, reply } => {
                let result = self.store.decay_all(factor).await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Stats { reply } => {
                let result = self.store.stats().await
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            
            MemoryMessage::Shutdown => unreachable!(),
        }
    }
}

// ============================================================================
// ActorRef (클라이언트 핸들)
// ============================================================================

/// MemoryActor에 메시지를 보내는 핸들
#[derive(Clone)]
pub struct MemoryActorRef {
    sender: mpsc::Sender<MemoryMessage>,
}

impl MemoryActorRef {
    /// 메모리 저장
    pub async fn store(&self, memory: Memory, embedding: Vec<f32>) -> Result<Uuid, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Store { memory, embedding, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 시맨틱 검색
    pub async fn recall(&self, query: &str, k: usize) -> Result<Vec<Memory>, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Recall { query: query.to_string(), k, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// ID로 메모리 조회
    pub async fn get(&self, id: Uuid) -> Result<Option<Memory>, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Get { id, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 메모리 삭제
    pub async fn delete(&self, id: Uuid) -> Result<(), String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Delete { id, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 메모리 간 링크 생성
    pub async fn link(&self, from: Uuid, to: Uuid, relation: RelationType, weight: f32) -> Result<(), String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Link { from, to, relation, weight, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 특정 메모리의 링크 조회
    pub async fn get_links(&self, memory_id: Uuid) -> Result<Vec<MemoryLink>, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::GetLinks { memory_id, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 메모리 강화
    pub async fn strengthen(&self, id: Uuid) -> Result<(), String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Strengthen { id, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 전체 약화
    pub async fn decay_all(&self, factor: f32) -> Result<u64, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::DecayAll { factor, reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 통계 조회
    pub async fn stats(&self) -> Result<StoreStats, String> {
        let (reply, rx) = oneshot::channel();
        self.sender.send(MemoryMessage::Stats { reply })
            .await
            .map_err(|_| "Actor channel closed".to_string())?;
        rx.await.map_err(|_| "Actor reply failed".to_string())?
    }
    
    /// 액터 종료
    pub async fn shutdown(&self) -> Result<(), String> {
        self.sender.send(MemoryMessage::Shutdown)
            .await
            .map_err(|_| "Actor channel closed".to_string())
    }
}

// ============================================================================
// Actor System Integration
// ============================================================================

/// MemoryActor를 생성하고 ActorRef 반환
pub fn spawn_memory_actor(store: SemanticLayer, buffer_size: usize) -> (MemoryActorRef, tokio::task::JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel(buffer_size);
    let actor = MemoryActor::new(Arc::new(store), receiver);
    
    let handle = tokio::spawn(async move {
        actor.run().await;
    });
    
    (MemoryActorRef { sender }, handle)
}

/// 기본 버퍼 크기로 MemoryActor 생성
pub fn spawn_memory_actor_default(store: SemanticLayer) -> (MemoryActorRef, tokio::task::JoinHandle<()>) {
    spawn_memory_actor(store, 256)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::HashEmbedder;
    use crate::semantic_store::SemanticLayer;
    
    fn make_test_store() -> SemanticLayer {
        let embedder = Arc::new(HashEmbedder::new(128));
        SemanticLayer::new("http://localhost:3100", "test", embedder)
    }
    
    #[tokio::test]
    async fn test_actor_stats() {
        let store = make_test_store();
        let (actor_ref, handle) = spawn_memory_actor_default(store);
        
        let stats = actor_ref.stats().await.unwrap();
        assert_eq!(stats.total_memories, 0);
        
        actor_ref.shutdown().await.unwrap();
        handle.await.unwrap();
    }
    
    #[tokio::test]
    async fn test_actor_link() {
        let store = make_test_store();
        let (actor_ref, handle) = spawn_memory_actor_default(store);
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        actor_ref.link(id1, id2, RelationType::Similar, 0.8).await.unwrap();
        
        let links = actor_ref.get_links(id1).await.unwrap();
        assert_eq!(links.len(), 1);
        
        actor_ref.shutdown().await.unwrap();
        handle.await.unwrap();
    }
}
