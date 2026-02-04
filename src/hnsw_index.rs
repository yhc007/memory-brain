//! HNSW Index for Fast Vector Search
//!
//! Hierarchical Navigable Small World graph for O(log n) approximate nearest neighbor search.
//! Much faster than brute-force O(n) search for large memory collections.

use hnsw::{Hnsw, Searcher};
use space::Metric;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Cosine distance metric for HNSW
#[derive(Clone, Copy)]
pub struct CosineDistance;

impl Metric<Vec<f32>> for CosineDistance {
    type Unit = u32;

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        // Use SIMD-accelerated cosine similarity
        // Cosine distance = 1 - cosine_similarity
        // We scale to u32 for HNSW (0 = identical, u32::MAX = opposite)
        
        if a.len() != b.len() || a.is_empty() {
            return u32::MAX;
        }

        let cosine_sim = crate::simd_ops::cosine_similarity_simd(a, b);
        
        // Convert from [-1, 1] to [0, 2], then scale to u32
        let distance = (1.0 - cosine_sim) / 2.0; // [0, 1]
        (distance * (u32::MAX as f32)) as u32
    }
}

/// HNSW-based vector index
pub struct HnswIndex {
    /// The HNSW graph
    hnsw: Arc<RwLock<Hnsw<CosineDistance, Vec<f32>, rand_pcg::Pcg64, 12, 24>>>,
    /// Mapping from internal index to UUID
    id_map: Arc<RwLock<HashMap<usize, Uuid>>>,
    /// Reverse mapping from UUID to internal index
    uuid_to_idx: Arc<RwLock<HashMap<Uuid, usize>>>,
    /// Next available index
    next_idx: Arc<RwLock<usize>>,
    /// Dimension of embeddings
    dimension: usize,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(dimension: usize) -> Self {
        use rand::SeedableRng;
        let rng = rand_pcg::Pcg64::seed_from_u64(42);
        
        Self {
            hnsw: Arc::new(RwLock::new(Hnsw::new_prng(CosineDistance, rng))),
            id_map: Arc::new(RwLock::new(HashMap::new())),
            uuid_to_idx: Arc::new(RwLock::new(HashMap::new())),
            next_idx: Arc::new(RwLock::new(0)),
            dimension,
        }
    }

    /// Add a vector to the index
    pub fn add(&self, id: Uuid, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            ));
        }

        let mut hnsw = self.hnsw.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut uuid_to_idx = self.uuid_to_idx.write().unwrap();
        let next_idx = self.next_idx.write().unwrap();

        // Check if already exists
        if uuid_to_idx.contains_key(&id) {
            return Ok(()); // Already indexed
        }

        // Create a searcher for insertion
        let mut searcher = Searcher::default();
        // insert returns the actual index used
        let idx = hnsw.insert(embedding, &mut searcher);

        id_map.insert(idx, id);
        uuid_to_idx.insert(id, idx);

        Ok(())
    }

    /// Add multiple vectors in batch
    pub fn add_batch(&self, items: &[(Uuid, Vec<f32>)]) -> Result<usize, String> {
        let mut count = 0;
        for (id, embedding) in items {
            if self.add(*id, embedding.clone()).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(Uuid, f32)> {
        use space::Neighbor;
        
        if query.len() != self.dimension {
            return Vec::new();
        }

        let hnsw = self.hnsw.read().unwrap();
        let id_map = self.id_map.read().unwrap();

        let mut searcher = Searcher::default();
        let query_vec = query.to_vec();
        
        // Prepare output buffer
        let mut neighbors: Vec<Neighbor<u32>> = vec![Neighbor { index: 0, distance: u32::MAX }; k];
        
        // Search returns slice of found neighbors
        let found = hnsw.nearest(&query_vec, k, &mut searcher, &mut neighbors);

        found
            .iter()
            .filter_map(|neighbor| {
                let idx = neighbor.index;
                let distance = neighbor.distance as f32 / u32::MAX as f32 * 2.0;
                let similarity = 1.0 - distance;
                
                id_map.get(&idx).map(|uuid| (*uuid, similarity))
            })
            .collect()
    }

    /// Remove a vector from the index (mark as deleted)
    pub fn remove(&self, id: &Uuid) -> bool {
        let mut uuid_to_idx = self.uuid_to_idx.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();

        if let Some(idx) = uuid_to_idx.remove(id) {
            id_map.remove(&idx);
            // Note: HNSW doesn't support true deletion, just remove from mappings
            true
        } else {
            false
        }
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let id_map = self.id_map.read().unwrap();
        IndexStats {
            count: id_map.len(),
            dimension: self.dimension,
        }
    }

    /// Clear the index
    pub fn clear(&self) {
        use rand::SeedableRng;
        
        let mut hnsw = self.hnsw.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut uuid_to_idx = self.uuid_to_idx.write().unwrap();
        let mut next_idx = self.next_idx.write().unwrap();

        let rng = rand_pcg::Pcg64::seed_from_u64(42);
        *hnsw = Hnsw::new_prng(CosineDistance, rng);
        id_map.clear();
        uuid_to_idx.clear();
        *next_idx = 0;
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub count: usize,
    pub dimension: usize,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HNSW Index: {} vectors, {}d", self.count, self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic() {
        let index = HnswIndex::new(4);

        // Add some vectors
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        index.add(id1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add(id2, vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        index.add(id3, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search for nearest to id1
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2);
        
        assert!(!results.is_empty());
        // First result should be id1 (exact match)
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn test_hnsw_stats() {
        let index = HnswIndex::new(128);

        for i in 0..100 {
            let id = Uuid::new_v4();
            let vec: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 1000.0).collect();
            index.add(id, vec).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.dimension, 128);
    }

    #[test]
    fn test_cosine_distance() {
        let metric = CosineDistance;

        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = metric.distance(&a, &b);
        assert!(dist < 1000); // Very close to 0

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = metric.distance(&a, &b);
        assert!(dist > u32::MAX / 4); // Around 0.5

        // Opposite vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = metric.distance(&a, &b);
        assert!(dist > u32::MAX / 2); // Close to 1.0
    }
}
