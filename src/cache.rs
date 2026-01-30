//! Embedding Cache Module
//!
//! LRU cache for embeddings to avoid recomputation.
//! Also supports batch processing for efficiency.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use crate::embedding::Embedder;

/// Cached embedder wrapper with LRU cache
pub struct CachedEmbedder<E: Embedder> {
    inner: E,
    cache: Arc<RwLock<LruCache<u64, Vec<f32>>>>,
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl<E: Embedder> CachedEmbedder<E> {
    /// Create a new cached embedder with specified cache size
    pub fn new(inner: E, cache_size: usize) -> Self {
        let size = NonZeroUsize::new(cache_size).unwrap_or(NonZeroUsize::new(1000).unwrap());
        Self {
            inner,
            cache: Arc::new(RwLock::new(LruCache::new(size))),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Create with default cache size (10000 entries)
    pub fn with_default_cache(inner: E) -> Self {
        Self::new(inner, 10000)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let cache = self.cache.read().unwrap();
        CacheStats {
            hits,
            misses,
            size: cache.len(),
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        *self.hits.write().unwrap() = 0;
        *self.misses.write().unwrap() = 0;
    }

    /// Hash text for cache key
    fn hash_text(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Batch embed multiple texts efficiently
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut to_compute: Vec<(usize, &str)> = Vec::new();

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            for (i, text) in texts.iter().enumerate() {
                let key = Self::hash_text(text);
                if let Some(embedding) = cache.peek(&key) {
                    results.push((i, embedding.clone()));
                } else {
                    to_compute.push((i, *text));
                }
            }
        }

        // Update hit/miss counts
        {
            *self.hits.write().unwrap() += results.len() as u64;
            *self.misses.write().unwrap() += to_compute.len() as u64;
        }

        // Compute missing embeddings
        let mut computed: Vec<(usize, Vec<f32>)> = Vec::new();
        for (i, text) in &to_compute {
            let embedding = self.inner.embed(text);
            computed.push((*i, embedding));
        }

        // Update cache with new embeddings
        {
            let mut cache = self.cache.write().unwrap();
            for ((_, text), (_, ref embedding)) in to_compute.iter().zip(computed.iter()) {
                let key = Self::hash_text(text);
                cache.put(key, embedding.clone());
            }
        }

        // Merge results
        results.extend(computed);
        results.sort_by_key(|(i, _)| *i);
        results.into_iter().map(|(_, v)| v).collect()
    }

    /// Preload cache with texts (useful for warmup)
    pub fn preload(&self, texts: &[&str]) {
        let _ = self.embed_batch(texts);
    }
}

impl<E: Embedder> Embedder for CachedEmbedder<E> {
    fn embed(&self, text: &str) -> Vec<f32> {
        let key = Self::hash_text(text);

        // Try to get from cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(embedding) = cache.get(&key) {
                *self.hits.write().unwrap() += 1;
                return embedding.clone();
            }
        }

        // Cache miss - compute embedding
        *self.misses.write().unwrap() += 1;
        let embedding = self.inner.embed(text);

        // Store in cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(key, embedding.clone());
        }

        embedding
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub hit_rate: f64,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache: {} entries, {:.1}% hit rate ({} hits, {} misses)",
            self.size,
            self.hit_rate * 100.0,
            self.hits,
            self.misses
        )
    }
}

/// Batch processor for memory operations
pub struct BatchProcessor<E: Embedder> {
    embedder: CachedEmbedder<E>,
    batch_size: usize,
}

impl<E: Embedder> BatchProcessor<E> {
    pub fn new(embedder: E, cache_size: usize, batch_size: usize) -> Self {
        Self {
            embedder: CachedEmbedder::new(embedder, cache_size),
            batch_size,
        }
    }

    /// Process texts in batches
    pub fn process_batch<F, T>(&self, texts: &[&str], processor: F) -> Vec<T>
    where
        F: Fn(&str, Vec<f32>) -> T,
    {
        let mut results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let embeddings = self.embedder.embed_batch(chunk);
            for (text, embedding) in chunk.iter().zip(embeddings) {
                results.push(processor(text, embedding));
            }
        }

        results
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.embedder.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::HashEmbedder;

    #[test]
    fn test_cached_embedder() {
        let inner = HashEmbedder::new(128);
        let cached = CachedEmbedder::new(inner, 100);

        // First call - cache miss
        let v1 = cached.embed("hello world");
        let stats = cached.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Second call - cache hit
        let v2 = cached.embed("hello world");
        let stats = cached.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);

        // Same embedding
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_batch_embed() {
        let inner = HashEmbedder::new(128);
        let cached = CachedEmbedder::new(inner, 100);

        let texts = vec!["hello", "world", "rust", "hello"]; // "hello" appears twice
        let embeddings = cached.embed_batch(&texts.iter().map(|s| *s).collect::<Vec<_>>());

        assert_eq!(embeddings.len(), 4);
        assert_eq!(embeddings[0], embeddings[3]); // Same text = same embedding

        let stats = cached.stats();
        assert_eq!(stats.size, 3); // 3 unique texts cached
    }

    #[test]
    fn test_cache_stats() {
        let inner = HashEmbedder::new(128);
        let cached = CachedEmbedder::new(inner, 100);

        for _ in 0..10 {
            cached.embed("same text");
        }

        let stats = cached.stats();
        assert_eq!(stats.hits, 9);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.9).abs() < 0.01);
    }
}
