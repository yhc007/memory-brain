//! Embedding Module
//! 
//! Converts text to vector representations for semantic search.
//! 
//! Supports multiple backends:
//! - Simple TF-IDF (built-in, no dependencies)
//! - Hash-based (fast, consistent)
//! - MLX models (requires mlx-rs, Apple Silicon optimized)

use std::collections::HashMap;

#[cfg(feature = "mlx")]
use mlx_rs;

/// Trait for embedding providers
pub trait Embedder: Send + Sync {
    /// Convert text to embedding vector
    fn embed(&self, text: &str) -> Vec<f32>;
    
    /// Embedding dimension
    fn dimension(&self) -> usize;
    
    /// Compute cosine similarity between two embeddings
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b)
    }
}

/// Simple TF-IDF based embedder (no external dependencies)
pub struct TfIdfEmbedder {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f32>,
    dimension: usize,
}

impl TfIdfEmbedder {
    /// Create a new TF-IDF embedder with fixed vocabulary size
    pub fn new(dimension: usize) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: vec![1.0; dimension],
            dimension,
        }
    }

    /// Create embedder from a corpus (learns vocabulary)
    pub fn from_corpus(texts: &[&str], dimension: usize) -> Self {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        
        // Count words and document frequencies
        for text in texts {
            let mut seen = std::collections::HashSet::new();
            for word in tokenize(text) {
                *word_counts.entry(word.clone()).or_insert(0) += 1;
                if seen.insert(word.clone()) {
                    *doc_freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Select top words by frequency
        let mut words: Vec<_> = word_counts.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        words.truncate(dimension);

        let vocabulary: HashMap<String, usize> = words
            .iter()
            .enumerate()
            .map(|(i, (word, _))| (word.clone(), i))
            .collect();

        // Calculate IDF
        let n = texts.len() as f32;
        let idf: Vec<f32> = words
            .iter()
            .map(|(word, _)| {
                let df = doc_freq.get(word).copied().unwrap_or(1) as f32;
                (n / df).ln() + 1.0
            })
            .collect();

        Self {
            vocabulary,
            idf,
            dimension: words.len().min(dimension),
        }
    }

    /// Add a word to vocabulary (for incremental learning)
    pub fn add_word(&mut self, word: &str) {
        if !self.vocabulary.contains_key(word) && self.vocabulary.len() < self.dimension {
            let idx = self.vocabulary.len();
            self.vocabulary.insert(word.to_string(), idx);
            if idx < self.idf.len() {
                self.idf[idx] = 1.0;
            } else {
                self.idf.push(1.0);
            }
        }
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimension];
        let tokens = tokenize(text);
        let total = tokens.len() as f32;
        
        if total == 0.0 {
            return vec;
        }

        // Count term frequencies
        let mut tf: HashMap<&str, f32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
        }

        // Build TF-IDF vector
        for (word, count) in tf {
            if let Some(&idx) = self.vocabulary.get(word) {
                if idx < self.dimension {
                    let tf_val = count / total;
                    let idf_val = self.idf.get(idx).copied().unwrap_or(1.0);
                    vec[idx] = tf_val * idf_val;
                }
            }
        }

        // Normalize
        normalize(&mut vec);
        vec
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Hash-based embedder (consistent across runs, no training needed)
pub struct HashEmbedder {
    dimension: usize,
}

impl HashEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimension];
        let tokens = tokenize(text);
        
        for token in tokens {
            // Hash the token to get indices
            let hash = simple_hash(&token);
            let idx = (hash as usize) % self.dimension;
            let sign = if (hash >> 16) & 1 == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }

        normalize(&mut vec);
        vec
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// MLX-based embedder using learned word embeddings
/// 
/// This uses MLX for efficient embedding lookup on Apple Silicon.
/// For production use with transformer models, extend this with a proper
/// tokenizer and model architecture.
#[cfg(feature = "mlx")]
pub struct MlxEmbedder {
    /// Embedding table: (vocab_size, embed_dim)
    embedding_table: mlx_rs::Array,
    /// Vocabulary mapping
    vocab: HashMap<String, i32>,
    dimension: usize,
}

#[cfg(feature = "mlx")]
impl MlxEmbedder {
    /// Create a new MLX embedder with random embeddings
    /// 
    /// For production, load pre-trained weights instead.
    pub fn new(vocab_size: usize, dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        mlx_rs::init();
        
        // Initialize embedding table with random values
        let embedding_table = mlx_rs::random::normal::<f32>(
            &[vocab_size as i32, dimension as i32],
            None,
        )?;
        
        Ok(Self {
            embedding_table,
            vocab: HashMap::new(),
            dimension,
        })
    }

    /// Create embedder from pre-trained safetensors file
    pub fn from_safetensors(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        mlx_rs::init();
        
        let tensors = mlx_rs::serialize::load_safetensors(path)?;
        
        let embedding_table = tensors.get("embeddings")
            .or_else(|| tensors.get("word_embeddings"))
            .or_else(|| tensors.get("embedding.weight"))
            .ok_or("No embedding table found in safetensors")?
            .clone();
        
        let shape = embedding_table.shape();
        let dimension = shape.get(1).copied().unwrap_or(256) as usize;
        
        Ok(Self {
            embedding_table,
            vocab: HashMap::new(),
            dimension,
        })
    }

    /// Add words to vocabulary
    pub fn add_to_vocab(&mut self, words: &[&str]) {
        for word in words {
            if !self.vocab.contains_key(*word) {
                let idx = self.vocab.len() as i32;
                self.vocab.insert(word.to_string(), idx);
            }
        }
    }

    /// Get or create vocabulary index for a word
    fn get_or_create_index(&mut self, word: &str) -> i32 {
        if let Some(&idx) = self.vocab.get(word) {
            idx
        } else {
            // Use hash to get consistent index for unknown words
            let hash = simple_hash(word);
            let vocab_size = self.embedding_table.shape()[0] as u32;
            (hash % vocab_size) as i32
        }
    }
}

#[cfg(feature = "mlx")]
impl Embedder for MlxEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return vec![0.0; self.dimension];
        }

        // Convert tokens to indices
        let vocab_size = self.embedding_table.shape()[0] as u32;
        let indices: Vec<i32> = tokens
            .iter()
            .map(|t| {
                self.vocab.get(t).copied().unwrap_or_else(|| {
                    let hash = simple_hash(t);
                    (hash % vocab_size) as i32
                })
            })
            .collect();

        // Create MLX array from indices
        let indices_array = match mlx_rs::Array::from_slice(&indices, &[indices.len() as i32]) {
            Ok(arr) => arr,
            Err(_) => return vec![0.0; self.dimension],
        };

        // Look up embeddings using MLX
        let embeddings = match mlx_rs::nn::embedding(&self.embedding_table, &indices_array) {
            Ok(emb) => emb,
            Err(_) => return vec![0.0; self.dimension],
        };

        // Mean pooling: average all token embeddings
        // Sum along axis 0 then divide by token count
        let sum_embedding = match embeddings.sum_axes(&[0], false) {
            Ok(sum) => sum,
            Err(_) => return vec![0.0; self.dimension],
        };
        
        let token_count = indices.len() as f32;
        let scale = mlx_rs::Array::from_float(1.0 / token_count);
        let mean_embedding = &sum_embedding * &scale;

        // Evaluate and convert to Vec<f32>
        mean_embedding.eval();
        match mean_embedding.to_vec::<f32>() {
            Ok(vec) => {
                let mut result = vec;
                normalize(&mut result);
                result
            }
            Err(_) => vec![0.0; self.dimension],
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Create a default MLX embedder
#[cfg(feature = "mlx")]
pub fn create_mlx_embedder(dimension: usize) -> Result<MlxEmbedder, Box<dyn std::error::Error>> {
    // Use 10000 vocabulary size by default (covers most common words)
    MlxEmbedder::new(10000, dimension)
}

// ============ Helper Functions ============

/// Simple tokenization
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| s.len() > 2)
        .filter(|s| !is_stop_word(s))
        .map(|s| s.to_string())
        .collect()
}

/// Check if word is a stop word
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "this",
        "that", "these", "those", "what", "which", "who", "whom",
    ];
    STOP_WORDS.contains(&word)
}

/// Simple hash function for consistency
fn simple_hash(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    hash
}

/// Normalize a vector to unit length
pub fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two vectors
/// 
/// Now uses SIMD acceleration (NEON on Apple Silicon, AVX on x86_64)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::simd_ops::cosine_similarity_simd(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedder() {
        let embedder = HashEmbedder::new(128);
        
        let v1 = embedder.embed("rust programming language");
        let v2 = embedder.embed("rust programming language");
        let v3 = embedder.embed("python scripting language");
        
        // Same text should give same embedding
        assert_eq!(v1, v2);
        
        // Similar text should have positive similarity
        let sim_same = embedder.similarity(&v1, &v2);
        let sim_diff = embedder.similarity(&v1, &v3);
        
        assert!(sim_same > sim_diff);
    }

    #[test]
    fn test_tfidf_embedder() {
        let corpus = vec![
            "rust is a systems programming language",
            "python is a scripting language",
            "rust and python are both popular",
        ];
        
        let embedder = TfIdfEmbedder::from_corpus(&corpus, 100);
        
        let v1 = embedder.embed("rust programming");
        let v2 = embedder.embed("python scripting");
        
        // Both should have non-zero embeddings
        assert!(v1.iter().any(|&x| x != 0.0));
        assert!(v2.iter().any(|&x| x != 0.0));
    }
}
