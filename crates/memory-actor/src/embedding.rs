//! Embedding Client - BGE-M3 HTTP API
//!
//! Connects to the embedding server at localhost:3201 for text → vector conversion.

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default embedding server URL
pub const DEFAULT_EMBEDDING_URL: &str = "http://localhost:3201";

/// BGE-M3 embedding dimension
pub const EMBEDDING_DIM: usize = 1024;

/// Embedding client for BGE-M3 server
pub struct EmbeddingClient {
    url: String,
}

#[derive(Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_sparse: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_colbert: bool,
}

#[derive(Deserialize)]
struct EmbedResponse {
    dense: Vec<Vec<f32>>,
}

impl EmbeddingClient {
    /// Create a new embedding client
    pub fn new(url: &str) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
        }
    }

    /// Create with default URL (localhost:3201)
    pub fn default() -> Self {
        Self::new(DEFAULT_EMBEDDING_URL)
    }

    /// Check if the server is available
    pub fn health_check(&self) -> bool {
        ureq::get(&format!("{}/health", self.url))
            .call()
            .is_ok()
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        match self.embed_batch(&[text.to_string()]) {
            Ok(mut vecs) if !vecs.is_empty() => Ok(vecs.remove(0)),
            Ok(_) => Err("Empty response from embedding server".to_string()),
            Err(e) => Err(e),
        }
    }

    /// Embed multiple texts in batch (more efficient)
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let req = EmbedRequest {
            texts: texts.to_vec(),
            return_sparse: false,
            return_colbert: false,
        };

        let resp = ureq::post(&format!("{}/embed", self.url))
            .set("Content-Type", "application/json")
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {}", e))?;

        if resp.status() != 200 {
            return Err(format!("Server error: status {}", resp.status()));
        }

        let body: EmbedResponse = resp
            .into_json()
            .map_err(|e| format!("JSON parse error: {}", e))?;

        debug!("Embedded {} texts", body.dense.len());
        Ok(body.dense)
    }

    /// Compute cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

/// Fallback embedder using simple hash (no server required)
pub struct HashEmbedder {
    pub dimension: usize,
}

impl HashEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Simple tokenization
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    /// Simple hash function
    fn simple_hash(s: &str) -> u32 {
        let mut hash: u32 = 5381;
        for c in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(c as u32);
        }
        hash
    }

    /// Normalize vector to unit length
    fn normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Embed text using hash-based approach
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimension];
        let tokens = Self::tokenize(text);

        for token in tokens {
            let hash = Self::simple_hash(&token);
            let idx = (hash as usize) % self.dimension;
            let sign = if (hash >> 16) & 1 == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }

        Self::normalize(&mut vec);
        vec
    }
}

/// Trait for embedding providers
pub trait Embedder: Send + Sync {
    /// Convert text to embedding vector
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Embedding dimension
    fn dimension(&self) -> usize;

    /// Compute cosine similarity
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        EmbeddingClient::cosine_similarity(a, b)
    }
}

impl Embedder for EmbeddingClient {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed(text).unwrap_or_else(|e| {
            warn!("Embedding failed: {}, using zero vector", e);
            vec![0.0; EMBEDDING_DIM]
        })
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed(text)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedder() {
        let embedder = HashEmbedder::new(128);

        let v1 = embedder.embed("rust programming");
        let v2 = embedder.embed("rust programming");
        let v3 = embedder.embed("python scripting");

        // Same text = same embedding
        assert_eq!(v1, v2);

        // Different text = different embedding
        assert_ne!(v1, v3);

        // Dimension check
        assert_eq!(v1.len(), 128);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((EmbeddingClient::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((EmbeddingClient::cosine_similarity(&a, &c)).abs() < 0.001);
    }
}
