//! VecDB Storage - CoreVecDB HTTP backend for persistent memory
//!
//! Uses CoreVecDB's HTTP API instead of CQL-based CoreDB.
//! Benefits:
//! - No CQL parsing issues with special characters
//! - Native vector similarity search
//! - Metadata filtering with indexed fields

use crate::types::{MemoryItem, MemoryType, Emotion};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

const DEFAULT_COLLECTION: &str = "memories";
const DEFAULT_DIM: usize = 1024;

/// Storage backend using CoreVecDB HTTP API.
pub struct VecDbStorage {
    base_url: String,
    collection: String,
}

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
    #[allow(dead_code)]
    success: bool,
}

#[derive(Serialize)]
struct SearchReq {
    vector: Vec<f32>,
    k: u32,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    filter: HashMap<String, String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
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
    dim: u32,
    vector_count: u64,
}

// ============================================================================
// VecDbStorage Implementation
// ============================================================================

impl VecDbStorage {
    /// Create a new VecDbStorage connecting to CoreVecDB.
    ///
    /// # Arguments
    /// * `base_url` - CoreVecDB HTTP API URL (e.g., "http://localhost:3100")
    /// * `collection` - Collection name (default: "memories")
    pub fn new(base_url: &str, collection: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let collection_name = collection.unwrap_or(DEFAULT_COLLECTION).to_string();

        let storage = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            collection: collection_name,
        };

        // Ensure collection exists
        storage.ensure_collection()?;

        Ok(storage)
    }

    /// Ensure the collection exists, create if not.
    fn ensure_collection(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        match ureq::get(&url).call() {
            Ok(resp) if resp.status() == 200 => {
                // Collection exists
                Ok(())
            }
            _ => {
                // Create collection
                let create_url = format!("{}/collections", self.base_url);
                let req = CreateCollectionReq {
                    name: self.collection.clone(),
                    dim: DEFAULT_DIM,
                    distance: "cosine".to_string(),
                    indexed_fields: vec![
                        "type".to_string(),
                        "tags".to_string(),
                        "emotion".to_string(),
                    ],
                    numeric_fields: vec![
                        "timestamp".to_string(),
                        "strength".to_string(),
                        "access_count".to_string(),
                    ],
                };

                match ureq::post(&create_url).send_json(&req) {
                    Ok(_) => Ok(()),
                    Err(ureq::Error::Status(409, _)) => Ok(()), // Already exists
                    Err(e) => Err(format!("Failed to create collection: {}", e).into()),
                }
            }
        }
    }

    /// Store a memory item with its embedding.
    pub fn store(
        &self,
        item: &MemoryItem,
        embedding: &[f32],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();

        // Required fields
        metadata.insert("id".to_string(), item.id.to_string());
        metadata.insert("content".to_string(), item.content.clone());
        metadata.insert("type".to_string(), format!("{:?}", item.memory_type));
        metadata.insert("emotion".to_string(), format!("{:?}", item.emotion));

        // Optional fields
        if let Some(ref ctx) = item.context {
            metadata.insert("context".to_string(), ctx.clone());
        }

        // Tags as comma-separated
        if !item.tags.is_empty() {
            metadata.insert("tags".to_string(), item.tags.join(","));
        }

        // Timestamps as epoch millis (stored as string for metadata)
        metadata.insert(
            "created_at".to_string(),
            item.created_at.timestamp_millis().to_string(),
        );
        metadata.insert(
            "last_accessed".to_string(),
            item.last_accessed.timestamp_millis().to_string(),
        );

        // Numeric fields (stored as string for metadata, indexed as numeric)
        metadata.insert("access_count".to_string(), item.access_count.to_string());
        metadata.insert("strength".to_string(), item.strength.to_string());

        let url = format!("{}/collections/{}/upsert_batch", self.base_url, self.collection);
        let req = UpsertBatchReq {
            vectors: vec![BatchVectorReq {
                vector: embedding.to_vec(),
                metadata,
            }],
        };

        let resp = ureq::post(&url)
            .send_json(&req)?;

        if resp.status() == 200 {
            let result: UpsertBatchResp = resp.into_json()?;
            Ok(result.start_id)
        } else {
            Err(format!("Failed to store: status {}", resp.status()).into())
        }
    }

    /// Store multiple memory items in batch.
    pub fn store_batch(
        &self,
        items: &[(MemoryItem, Vec<f32>)],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if items.is_empty() {
            return Ok(0);
        }

        let vectors: Vec<BatchVectorReq> = items
            .iter()
            .map(|(item, embedding)| {
                let mut metadata = HashMap::new();
                metadata.insert("id".to_string(), item.id.to_string());
                metadata.insert("content".to_string(), item.content.clone());
                metadata.insert("type".to_string(), format!("{:?}", item.memory_type));
                metadata.insert("emotion".to_string(), format!("{:?}", item.emotion));

                if let Some(ref ctx) = item.context {
                    metadata.insert("context".to_string(), ctx.clone());
                }
                if !item.tags.is_empty() {
                    metadata.insert("tags".to_string(), item.tags.join(","));
                }

                metadata.insert("created_at".to_string(), item.created_at.timestamp_millis().to_string());
                metadata.insert("last_accessed".to_string(), item.last_accessed.timestamp_millis().to_string());
                metadata.insert("access_count".to_string(), item.access_count.to_string());
                metadata.insert("strength".to_string(), item.strength.to_string());

                BatchVectorReq {
                    vector: embedding.clone(),
                    metadata,
                }
            })
            .collect();

        let url = format!("{}/collections/{}/upsert_batch", self.base_url, self.collection);
        let req = UpsertBatchReq { vectors };

        let resp = ureq::post(&url)
            .send_json(&req)?;

        if resp.status() == 200 {
            let result: UpsertBatchResp = resp.into_json()?;
            Ok(result.start_id)
        } else {
            Err(format!("Failed to store batch: status {}", resp.status()).into())
        }
    }

    /// Search for similar memories using vector similarity.
    ///
    /// Returns (id, score, metadata) tuples.
    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        type_filter: Option<&str>,
    ) -> Result<Vec<(u64, f32, HashMap<String, String>)>, Box<dyn std::error::Error>> {
        self.search_with_metadata(query_embedding, k, type_filter, true)
    }

    /// Search with optional metadata inclusion.
    pub fn search_with_metadata(
        &self,
        query_embedding: &[f32],
        k: usize,
        type_filter: Option<&str>,
        include_metadata: bool,
    ) -> Result<Vec<(u64, f32, HashMap<String, String>)>, Box<dyn std::error::Error>> {
        let url = format!("{}/collections/{}/search", self.base_url, self.collection);

        let mut filter = HashMap::new();
        if let Some(t) = type_filter {
            filter.insert("type".to_string(), t.to_string());
        }

        let req = SearchReq {
            vector: query_embedding.to_vec(),
            k: k as u32,
            filter,
            include_metadata,
        };

        let resp = ureq::post(&url)
            .send_json(&req)?;

        if resp.status() == 200 {
            let result: SearchResp = resp.into_json()?;

            let results = result.results
                .into_iter()
                .map(|sr| (sr.id, sr.score, sr.metadata.unwrap_or_default()))
                .collect();

            Ok(results)
        } else {
            Err(format!("Search failed: status {}", resp.status()).into())
        }
    }

    /// Search and return full MemoryItems.
    pub fn search_memories(
        &self,
        query_embedding: &[f32],
        k: usize,
        type_filter: Option<&str>,
    ) -> Result<Vec<(MemoryItem, f32)>, Box<dyn std::error::Error>> {
        let url = format!("{}/collections/{}/search", self.base_url, self.collection);

        let mut filter = HashMap::new();
        if let Some(t) = type_filter {
            filter.insert("type".to_string(), t.to_string());
        }

        let req = SearchReq {
            vector: query_embedding.to_vec(),
            k: k as u32,
            filter,
            include_metadata: true,  // Request metadata!
        };

        let resp = ureq::post(&url)
            .send_json(&req)?;

        if resp.status() != 200 {
            return Err(format!("Search failed: status {}", resp.status()).into());
        }

        let search_result: SearchResp = resp.into_json()?;

        let mut results = Vec::with_capacity(search_result.results.len());
        for sr in search_result.results {
            let item = self.metadata_to_memory_item(sr.id, sr.metadata.as_ref());
            results.push((item, sr.score));
        }

        Ok(results)
    }

    /// Convert metadata HashMap to MemoryItem.
    fn metadata_to_memory_item(&self, vec_id: u64, metadata: Option<&HashMap<String, String>>) -> MemoryItem {
        let meta = metadata.cloned().unwrap_or_default();

        let id = meta.get("id")
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::new_v4);

        let content = meta.get("content").cloned().unwrap_or_else(|| format!("[Vector ID {}]", vec_id));
        let context = meta.get("context").cloned();

        let memory_type = meta.get("type")
            .map(|s| match s.as_str() {
                "Semantic" => MemoryType::Semantic,
                "Procedural" => MemoryType::Procedural,
                _ => MemoryType::Episodic,
            })
            .unwrap_or(MemoryType::Episodic);

        let emotion = meta.get("emotion")
            .map(|s| match s.as_str() {
                "Positive" => Emotion::Positive,
                "Negative" => Emotion::Negative,
                "Surprise" => Emotion::Surprise,
                _ => Emotion::Neutral,
            })
            .unwrap_or(Emotion::Neutral);

        let tags: Vec<String> = meta.get("tags")
            .map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect())
            .unwrap_or_default();

        let created_at = meta.get("created_at")
            .and_then(|s| s.parse::<i64>().ok())
            .map(|ms| chrono::DateTime::from_timestamp_millis(ms).unwrap_or_else(Utc::now))
            .unwrap_or_else(Utc::now);

        let last_accessed = meta.get("last_accessed")
            .and_then(|s| s.parse::<i64>().ok())
            .map(|ms| chrono::DateTime::from_timestamp_millis(ms).unwrap_or_else(Utc::now))
            .unwrap_or_else(Utc::now);

        let access_count = meta.get("access_count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let strength = meta.get("strength")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        MemoryItem {
            id,
            content,
            context,
            memory_type,
            emotion,
            created_at,
            last_accessed,
            access_count,
            strength,
            embedding: None,
            tags,
            associations: vec![],
        }
    }

    /// Get collection statistics.
    pub fn stats(&self) -> Result<(u64, u32), Box<dyn std::error::Error>> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);

        let resp = ureq::get(&url).call()?;

        if resp.status() == 200 {
            let info: CollectionInfo = resp.into_json()?;
            Ok((info.vector_count, info.dim))
        } else {
            Err(format!("Failed to get stats: status {}", resp.status()).into())
        }
    }

    /// Check if storage is healthy.
    pub fn health_check(&self) -> bool {
        let url = format!("{}/collections", self.base_url);
        ureq::get(&url).call().map(|r| r.status() == 200).unwrap_or(false)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        // This test requires CoreVecDB to be running
        if let Ok(storage) = VecDbStorage::new("http://localhost:3100", Some("test_memories")) {
            assert!(storage.health_check());
        }
    }
}
