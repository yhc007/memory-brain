//! Memory Brain HTTP Server
//!
//! REST API for memory operations without CLI overhead.
//!
//! ## Endpoints
//! - POST /store - Store a memory
//! - POST /recall - Search memories
//! - POST /batch - Batch store memories
//! - GET /stats - Get statistics
//! - DELETE /memory/:id - Delete a memory

use axum::{
    extract::{Path, State},
    http::{StatusCode, Method},
    response::Json,
    routing::{get, post, delete},
    Router,
};
use tower_http::cors::{CorsLayer, Any};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{Brain, MemoryItem, MemoryType, GloVeEmbedder, HnswIndex, Embedder};

/// Server state
pub struct AppState {
    pub brain: RwLock<Brain>,
    pub hnsw: HnswIndex,
    pub embedder: Arc<dyn Embedder>,
}

/// Store request
#[derive(Debug, Deserialize)]
pub struct StoreRequest {
    content: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    memory_type: Option<String>,
}

/// Store response
#[derive(Debug, Serialize)]
pub struct StoreResponse {
    id: String,
    success: bool,
}

/// Recall request
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    use_hnsw: bool,
}

fn default_limit() -> usize { 5 }

/// Memory response
#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    id: String,
    content: String,
    context: Option<String>,
    tags: Vec<String>,
    strength: f32,
    similarity: Option<f32>,
    created_at: String,
}

/// Batch store request
#[derive(Debug, Deserialize)]
pub struct BatchStoreRequest {
    memories: Vec<StoreRequest>,
}

/// Batch response
#[derive(Debug, Serialize)]
pub struct BatchResponse {
    stored: usize,
    errors: usize,
}

/// Stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    working_memory: usize,
    semantic_memory: usize,
    episodic_memory: usize,
    hnsw_indexed: usize,
    embedding_dim: usize,
}

/// Create the router
pub fn create_router(state: Arc<AppState>) -> Router {
    // CORS configuration - allow all origins for API access
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);

    // API routes
    let api = Router::new()
        .route("/store", post(store_handler))
        .route("/recall", post(recall_handler))
        .route("/batch", post(batch_handler))
        .route("/stats", get(stats_handler))
        .route("/memory/:id", delete(delete_handler))
        .route("/health", get(health_handler));
    
    // Web UI routes
    let web = crate::web_ui::create_web_router();
    
    Router::new()
        .nest("/api", api)
        .merge(web)
        .layer(cors)
        .with_state(state)
}

/// Store a memory
async fn store_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StoreRequest>,
) -> Result<Json<StoreResponse>, StatusCode> {
    let mut brain = state.brain.write().await;
    
    // Generate embedding
    let embedding = state.embedder.embed(&req.content);
    
    // Create memory item
    let mut item = MemoryItem::new(&req.content, req.context.as_deref());
    item.tags = req.tags;
    item.embedding = Some(embedding.clone());
    
    if let Some(ref mt) = req.memory_type {
        item.memory_type = match mt.to_lowercase().as_str() {
            "episodic" => MemoryType::Episodic,
            "procedural" => MemoryType::Procedural,
            _ => MemoryType::Semantic,
        };
    }
    
    let id = item.id.to_string();
    
    // Store in brain
    match brain.semantic.store(item.clone()) {
        Ok(_) => {
            // Also add to HNSW index
            let _ = state.hnsw.add(item.id, embedding);
            Ok(Json(StoreResponse { id, success: true }))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Recall memories
async fn recall_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Vec<MemoryResponse>>, StatusCode> {
    let mut brain = state.brain.write().await;
    
    let results = if req.use_hnsw {
        // Use HNSW for fast search
        let query_embedding = state.embedder.embed(&req.query);
        let hnsw_results = state.hnsw.search(&query_embedding, req.limit);
        
        // Convert to MemoryResponse (need to fetch full items)
        // For now, just return IDs with similarity
        hnsw_results
            .into_iter()
            .map(|(id, sim)| MemoryResponse {
                id: id.to_string(),
                content: String::new(), // TODO: fetch from storage
                context: None,
                tags: Vec::new(),
                strength: sim,
                similarity: Some(sim),
                created_at: String::new(),
            })
            .collect()
    } else {
        // Use brain's recall
        let memories = brain.recall(&req.query, req.limit);
        
        memories
            .into_iter()
            .map(|m| MemoryResponse {
                id: m.id.to_string(),
                content: m.content,
                context: m.context,
                tags: m.tags,
                strength: m.strength,
                similarity: None,
                created_at: m.created_at.to_rfc3339(),
            })
            .collect()
    };
    
    Ok(Json(results))
}

/// Batch store
async fn batch_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchStoreRequest>,
) -> Result<Json<BatchResponse>, StatusCode> {
    let mut brain = state.brain.write().await;
    let mut stored = 0;
    let mut errors = 0;
    
    for mem_req in req.memories {
        let embedding = state.embedder.embed(&mem_req.content);
        let mut item = MemoryItem::new(&mem_req.content, mem_req.context.as_deref());
        item.tags = mem_req.tags;
        item.embedding = Some(embedding.clone());
        
        match brain.semantic.store(item.clone()) {
            Ok(_) => {
                let _ = state.hnsw.add(item.id, embedding);
                stored += 1;
            }
            Err(_) => errors += 1,
        }
    }
    
    Ok(Json(BatchResponse { stored, errors }))
}

/// Get stats
async fn stats_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatsResponse>, StatusCode> {
    let brain = state.brain.read().await;
    let hnsw_stats = state.hnsw.stats();
    
    // Get counts by searching with empty query
    let semantic_count = brain.semantic.search("", 10000).map(|v| v.len()).unwrap_or(0);
    let episodic_count = brain.episodic.search("", 10000).map(|v| v.len()).unwrap_or(0);
    
    Ok(Json(StatsResponse {
        working_memory: brain.working.len(),
        semantic_memory: semantic_count,
        episodic_memory: episodic_count,
        hnsw_indexed: hnsw_stats.count,
        embedding_dim: hnsw_stats.dimension,
    }))
}

/// Delete a memory
async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> StatusCode {
    if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        state.hnsw.remove(&uuid);
        // TODO: also delete from brain storage
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

/// Health check
async fn health_handler() -> &'static str {
    "OK"
}

/// Start the server
pub async fn start_server(host: &str, port: u16, db_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize brain
    let embedder: Arc<dyn Embedder> = Arc::new(GloVeEmbedder::test_embedder());
    let dimension = embedder.dimension();
    
    let mut brain = Brain::with_embedder(db_path, embedder.clone())?;
    
    // Rebuild indexes for search (critical for recall to work!)
    let stats = brain.rebuild_indexes()?;
    println!("🔍 Index loaded: {} memories, {} keywords", 
        stats.episodic_count + stats.semantic_count + stats.procedural_count,
        stats.index_stats.unique_keywords);
    
    let state = Arc::new(AppState {
        brain: RwLock::new(brain),
        hnsw: HnswIndex::new(dimension),
        embedder,
    });
    
    let app = create_router(state);
    
    let addr = format!("{}:{}", host, port);
    println!("🧠 Memory Brain Server starting on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_health() {
        let embedder: Arc<dyn Embedder> = Arc::new(GloVeEmbedder::test_embedder());
        let dim = embedder.dimension();
        let dir = tempfile::tempdir().unwrap();
        let brain = Brain::with_embedder(dir.path().join("test.db").to_str().unwrap(), embedder.clone()).unwrap();
        
        let state = Arc::new(AppState {
            brain: RwLock::new(brain),
            hnsw: HnswIndex::new(dim),
            embedder,
        });
        
        let app = create_router(state);
        
        let response = app
            .oneshot(Request::builder().uri("/api/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }
}
