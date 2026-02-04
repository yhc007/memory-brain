//! Bloom Filter for Fast Existence Checks
//!
//! Probabilistic data structure for O(1) "does this keyword exist?" queries.
//! - No false negatives (if says "no", definitely no)
//! - Possible false positives (if says "yes", probably yes)
//! - Very memory efficient

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// Bloom filter for fast membership testing
pub struct BloomFilter {
    /// Bit array
    bits: RwLock<Vec<bool>>,
    /// Number of bits
    size: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Number of items added
    count: RwLock<usize>,
}

impl BloomFilter {
    /// Create a new bloom filter
    /// 
    /// - `expected_items`: Expected number of items to add
    /// - `false_positive_rate`: Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal size: m = -n*ln(p) / (ln(2)^2)
        let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let size = (-(expected_items as f64) * false_positive_rate.ln() / ln2_sq).ceil() as usize;
        let size = size.max(64); // Minimum size
        
        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes = ((size as f64 / expected_items as f64) * std::f64::consts::LN_2).ceil() as usize;
        let num_hashes = num_hashes.max(1).min(16); // Between 1 and 16

        Self {
            bits: RwLock::new(vec![false; size]),
            size,
            num_hashes,
            count: RwLock::new(0),
        }
    }

    /// Create with specific size and hash count
    pub fn with_size(size: usize, num_hashes: usize) -> Self {
        Self {
            bits: RwLock::new(vec![false; size]),
            size,
            num_hashes,
            count: RwLock::new(0),
        }
    }

    /// Generate hash values for an item
    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut results = Vec::with_capacity(self.num_hashes);
        
        // Use double hashing: h(i) = h1 + i*h2
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish() as usize;
        
        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(h1);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish() as usize;
        
        for i in 0..self.num_hashes {
            let hash = (h1.wrapping_add(i.wrapping_mul(h2))) % self.size;
            results.push(hash);
        }
        
        results
    }

    /// Add an item to the filter
    pub fn add<T: Hash>(&self, item: &T) {
        let hashes = self.hashes(item);
        let mut bits = self.bits.write().unwrap();
        
        for hash in hashes {
            bits[hash] = true;
        }
        
        let mut count = self.count.write().unwrap();
        *count += 1;
    }

    /// Add a string (convenience method)
    pub fn add_str(&self, s: &str) {
        self.add(&s.to_lowercase());
    }

    /// Check if an item might be in the filter
    /// 
    /// Returns:
    /// - `false`: Definitely NOT in the set
    /// - `true`: Probably in the set (may be false positive)
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hashes(item);
        let bits = self.bits.read().unwrap();
        
        hashes.iter().all(|&hash| bits[hash])
    }

    /// Check if a string might be in the filter
    pub fn contains_str(&self, s: &str) -> bool {
        self.contains(&s.to_lowercase())
    }

    /// Get the approximate false positive rate
    pub fn false_positive_rate(&self) -> f64 {
        let count = *self.count.read().unwrap();
        if count == 0 {
            return 0.0;
        }
        
        // FPR ≈ (1 - e^(-k*n/m))^k
        let k = self.num_hashes as f64;
        let n = count as f64;
        let m = self.size as f64;
        
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Get statistics
    pub fn stats(&self) -> BloomStats {
        let bits = self.bits.read().unwrap();
        let set_bits = bits.iter().filter(|&&b| b).count();
        
        BloomStats {
            size: self.size,
            num_hashes: self.num_hashes,
            items_added: *self.count.read().unwrap(),
            bits_set: set_bits,
            fill_ratio: set_bits as f64 / self.size as f64,
            estimated_fpr: self.false_positive_rate(),
        }
    }

    /// Clear the filter
    pub fn clear(&self) {
        let mut bits = self.bits.write().unwrap();
        bits.fill(false);
        let mut count = self.count.write().unwrap();
        *count = 0;
    }

    /// Merge another bloom filter into this one (OR operation)
    pub fn merge(&self, other: &BloomFilter) -> Result<(), &'static str> {
        if self.size != other.size || self.num_hashes != other.num_hashes {
            return Err("Bloom filters must have same size and hash count");
        }
        
        let mut self_bits = self.bits.write().unwrap();
        let other_bits = other.bits.read().unwrap();
        
        for (i, &bit) in other_bits.iter().enumerate() {
            self_bits[i] = self_bits[i] || bit;
        }
        
        Ok(())
    }
}

impl Default for BloomFilter {
    fn default() -> Self {
        // Default: 10000 items, 1% false positive rate
        Self::new(10000, 0.01)
    }
}

/// Bloom filter statistics
#[derive(Debug, Clone, Default)]
pub struct BloomStats {
    pub size: usize,
    pub num_hashes: usize,
    pub items_added: usize,
    pub bits_set: usize,
    pub fill_ratio: f64,
    pub estimated_fpr: f64,
}

impl std::fmt::Display for BloomStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bloom Filter: {} bits, {} hashes, {} items, {:.1}% full, {:.2}% FPR",
            self.size,
            self.num_hashes,
            self.items_added,
            self.fill_ratio * 100.0,
            self.estimated_fpr * 100.0
        )
    }
}

/// Counting Bloom Filter - supports removal
pub struct CountingBloomFilter {
    /// Counter array (instead of bits)
    counters: RwLock<Vec<u8>>,
    /// Size
    size: usize,
    /// Number of hash functions
    num_hashes: usize,
}

impl CountingBloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let size = (-(expected_items as f64) * false_positive_rate.ln() / ln2_sq).ceil() as usize;
        let size = size.max(64);
        
        let num_hashes = ((size as f64 / expected_items as f64) * std::f64::consts::LN_2).ceil() as usize;
        let num_hashes = num_hashes.max(1).min(16);

        Self {
            counters: RwLock::new(vec![0u8; size]),
            size,
            num_hashes,
        }
    }

    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut results = Vec::with_capacity(self.num_hashes);
        
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish() as usize;
        
        let mut hasher2 = DefaultHasher::new();
        hasher2.write_usize(h1);
        item.hash(&mut hasher2);
        let h2 = hasher2.finish() as usize;
        
        for i in 0..self.num_hashes {
            let hash = (h1.wrapping_add(i.wrapping_mul(h2))) % self.size;
            results.push(hash);
        }
        
        results
    }

    pub fn add<T: Hash>(&self, item: &T) {
        let hashes = self.hashes(item);
        let mut counters = self.counters.write().unwrap();
        
        for hash in hashes {
            counters[hash] = counters[hash].saturating_add(1);
        }
    }

    pub fn remove<T: Hash>(&self, item: &T) {
        let hashes = self.hashes(item);
        let mut counters = self.counters.write().unwrap();
        
        for hash in hashes {
            counters[hash] = counters[hash].saturating_sub(1);
        }
    }

    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hashes(item);
        let counters = self.counters.read().unwrap();
        
        hashes.iter().all(|&hash| counters[hash] > 0)
    }

    pub fn add_str(&self, s: &str) {
        self.add(&s.to_lowercase());
    }

    pub fn remove_str(&self, s: &str) {
        self.remove(&s.to_lowercase());
    }

    pub fn contains_str(&self, s: &str) -> bool {
        self.contains(&s.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_basic() {
        let bloom = BloomFilter::new(1000, 0.01);
        
        bloom.add_str("hello");
        bloom.add_str("world");
        bloom.add_str("rust");
        
        assert!(bloom.contains_str("hello"));
        assert!(bloom.contains_str("world"));
        assert!(bloom.contains_str("rust"));
        
        // These should (almost certainly) not be in the filter
        // Note: false positives are possible but unlikely with 1% FPR
        assert!(!bloom.contains_str("python"));
        assert!(!bloom.contains_str("java"));
    }

    #[test]
    fn test_bloom_case_insensitive() {
        let bloom = BloomFilter::new(100, 0.01);
        
        bloom.add_str("Hello");
        
        assert!(bloom.contains_str("hello"));
        assert!(bloom.contains_str("HELLO"));
        assert!(bloom.contains_str("HeLLo"));
    }

    #[test]
    fn test_bloom_stats() {
        let bloom = BloomFilter::new(100, 0.05);
        
        for i in 0..50 {
            bloom.add_str(&format!("item{}", i));
        }
        
        let stats = bloom.stats();
        assert_eq!(stats.items_added, 50);
        assert!(stats.fill_ratio > 0.0);
        assert!(stats.fill_ratio < 1.0);
    }

    #[test]
    fn test_counting_bloom() {
        let bloom = CountingBloomFilter::new(100, 0.01);
        
        bloom.add_str("test");
        assert!(bloom.contains_str("test"));
        
        bloom.remove_str("test");
        assert!(!bloom.contains_str("test"));
    }

    #[test]
    fn test_bloom_merge() {
        let bloom1 = BloomFilter::with_size(1000, 3);
        let bloom2 = BloomFilter::with_size(1000, 3);
        
        bloom1.add_str("hello");
        bloom2.add_str("world");
        
        bloom1.merge(&bloom2).unwrap();
        
        assert!(bloom1.contains_str("hello"));
        assert!(bloom1.contains_str("world"));
    }
}
