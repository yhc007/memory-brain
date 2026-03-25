//! VecDB Storage - CoreVecDB HTTP backend for memory persistence
//!
//! Connects to CoreVecDB at localhost:3100 for vector similarity search.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use chrono::Utc;
use uuid::Uuid;

use crate::messages::{Memory, MemoryContext};

/// Default CoreVecDB URL
pub const DEFAULT_VECDB_URL: &str = "http://localhost:3100";

/// Default collection name
pub const DEFAULT_COLLECTION: &str = "memory_actor";

/// Default embedding dimension (BGE-M3)
pub const DEFAULT_DIM: usize = 1024;

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Serialize)]
struct CreateCollectionReq {
    name: String,
    dim: usize,
    distance: String,
    indexed_fields: Vec<String>,
    numeric_fields: Vec<String>,
}

#[derive(Serialize)]
struct BatchVectorReq {
    vector: Vec<f32>,
    metadata: HashMap<String, String>,
}

#[derive(Serialize)]
struct UpsertBatchReq {
    vectors: Vec<BatchVectorReq>,
}

#[derive(Deserialize)]
struct UpsertBatchResp {
    start_id: u64,
    #[allow(dead_code)]
    count: usize,
}

#[derive(Serialize)]
struct SearchReq {
    vector: Vec<f32>,
    k: u32,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    filter: HashMap<String, String>,
    include_metadata: bool,
}

#[derive(Deserialize)]
struct SearchResp {
    results: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    id: u64,
    score: f32,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
struct CollectionInfo {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    dim: u32,
    vector_count: u64,
}

// ============================================================================
// VecDbStorage Implementation
// ============================================================================

/// CoreVecDB storage backend for memories
pub struct VecDbStorage {
    base_url: String,
    collection: String,
}

impl VecDbStorage {
    /// Create a new VecDbStorage connecting to CoreVecDB.
    pub fn new(base_url: &str, collection: &str) -> Result<Self, String> {
        let storage = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            collection: collection.to_string(),
        };

        storage.ensure_collection()?;
        info!("Connected to CoreVecDB: {} (collection: {})", base_url, collection);
        Ok(storage)
    }

    /// Create with default settings
    pub fn default() -> Result<Self, String> {
        Self::new(DEFAULT_VECDB_URL, DEFAULT_COLLECTION)
    }

    /// Ensure collection exists
    fn ensure_collection(&self) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        match ureq::get(&url).call() {
            Ok(resp) if resp.status() == 200 => Ok(()),
            _ => {
                // Create collection
                let req = CreateCollectionReq {
                    name: self.collection.clone(),
                    dim: DEFAULT_DIM,
                    distance: "cosine".to_string(),
                    indexed_fields: vec![
                        "source".to_string(),
                        "tags".to_string(),
                    ],
                    numeric_fields: vec![
                        "timestamp".to_string(),
                        "strength".to_string(),
                        "access_count".to_string(),
                    ],
                };

                match ureq::post(&format!("{}/collections", self.base_url)).send_json(&req) {
                    Ok(_) => {
                        info!("Created collection: {}", self.collection);
                        Ok(())
                    }
                    Err(ureq::Error::Status(409, _)) => Ok(()), // Already exists
                    Err(e) => Err(format!("Failed to create collection: {}", e)),
                }
            }
        }
    }

    /// Store a memory with its embedding
    pub fn store(&self, memory: &Memory, embedding: &[f32]) -> Result<u64, String> {
        let mut metadata = HashMap::new();

        // Required fields
        metadata.insert("id".to_string(), memory.id.to_string());
        metadata.insert("content".to_string(), memory.content.clone());
        metadata.insert("source".to_string(), memory.context.source.clone());

        // Tags as comma-separated
        if !memory.context.tags.is_empty() {
            metadata.insert("tags".to_string(), memory.context.tags.join(","));
        }

        // Timestamps
        metadata.insert("created_at".to_string(), memory.created_at.timestamp_millis().to_string());
        metadata.insert("last_accessed".to_string(), memory.last_accessed.timestamp_millis().to_string());

        // Numeric fields
        metadata.insert("access_count".to_string(), memory.access_count.to_string());
        metadata.insert("strength".to_string(), memory.strength.to_string());

        let url = format!("{}/collections/{}/upsert_batch", self.base_url, self.collection);
        let req = UpsertBatchReq {
            vectors: vec![BatchVectorReq {
                vector: embedding.to_vec(),
                metadata,
            }],
        };

        let resp = ureq::post(&url)
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() == 200 {
            let result: UpsertBatchResp = resp.into_json()
                .map_err(|e| format!("JSON error: {}", e))?;
            debug!("Stored memory {} -> vec_id {}", memory.id, result.start_id);
            Ok(result.start_id)
        } else {
            Err(format!("Store failed: status {}", resp.status()))
        }
    }

    /// Search for similar memories
    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<MemorySearchResult>, String> {
        self.search_with_filter(query_embedding, k, None)
    }

    /// Search with optional source filter
    pub fn search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<MemorySearchResult>, String> {
        let url = format!("{}/collections/{}/search", self.base_url, self.collection);

        let mut filter = HashMap::new();
        if let Some(src) = source_filter {
            filter.insert("source".to_string(), src.to_string());
        }

        let req = SearchReq {
            vector: query_embedding.to_vec(),
            k: k as u32,
            filter,
            include_metadata: true,
        };

        let resp = ureq::post(&url)
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() != 200 {
            return Err(format!("Search failed: status {}", resp.status()));
        }

        let search_resp: SearchResp = resp.into_json()
            .map_err(|e| format!("JSON error: {}", e))?;

        let results: Vec<MemorySearchResult> = search_resp.results
            .into_iter()
            .map(|sr| {
                let memory = self.metadata_to_memory(sr.id, sr.metadata.as_ref());
                MemorySearchResult {
                    memory,
                    score: sr.score,
                    vec_id: sr.id,
                }
            })
            .collect();

        debug!("Search returned {} results", results.len());
        Ok(results)
    }

    /// Convert metadata to Memory struct
    fn metadata_to_memory(&self, vec_id: u64, metadata: Option<&HashMap<String, String>>) -> Memory {
        let meta = metadata.cloned().unwrap_or_default();

        let id = meta.get("id")
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::new_v4);

        let content = meta.get("content").cloned()
            .unwrap_or_else(|| format!("[Vector {}]", vec_id));

        let source = meta.get("source").cloned().unwrap_or_default();

        let tags: Vec<String> = meta.get("tags")
            .map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect())
            .unwrap_or_default();

        let created_at = meta.get("created_at")
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(|ms| chrono::DateTime::from_timestamp_millis(ms))
            .unwrap_or_else(Utc::now);

        let last_accessed = meta.get("last_accessed")
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(|ms| chrono::DateTime::from_timestamp_millis(ms))
            .unwrap_or_else(Utc::now);

        let access_count = meta.get("access_count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let strength = meta.get("strength")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        Memory {
            id,
            content,
            context: MemoryContext {
                source,
                tags,
                ..Default::default()
            },
            created_at,
            last_accessed,
            access_count,
            strength,
            links: vec![],
            embedding: None,
        }
    }

    /// Get collection statistics
    pub fn stats(&self) -> Result<VecDbStats, String> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        let resp = ureq::get(&url)
            .call()
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() == 200 {
            let info: CollectionInfo = resp.into_json()
                .map_err(|e| format!("JSON error: {}", e))?;
            Ok(VecDbStats {
                vector_count: info.vector_count,
            })
        } else {
            Err(format!("Stats failed: status {}", resp.status()))
        }
    }

    /// Check if storage is healthy
    pub fn health_check(&self) -> bool {
        ureq::get(&format!("{}/collections", self.base_url))
            .call()
            .map(|r| r.status() == 200)
            .unwrap_or(false)
    }
}

/// Search result with memory and score
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub memory: Memory,
    pub score: f32,
    pub vec_id: u64,
}

/// VecDB statistics
#[derive(Debug, Clone)]
pub struct VecDbStats {
    pub vector_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_VECDB_URL, "http://localhost:3100");
        assert_eq!(DEFAULT_DIM, 1024);
    }

    // Integration tests require CoreVecDB to be running
    #[test]
    #[ignore]
    fn test_storage_connection() {
        let storage = VecDbStorage::default().expect("Should connect");
        assert!(storage.health_check());
    }
}
