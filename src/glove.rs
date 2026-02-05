//! GloVe Embeddings Loader
//! 
//! Loads pre-trained GloVe word embeddings for semantic similarity.
//! Download from: https://nlp.stanford.edu/projects/glove/
//! 
//! Recommended: glove.6B.100d.txt (100-dimensional, smaller file)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::embedding::{Embedder, normalize, tokenize};

/// GloVe word embedding model
pub struct GloVeEmbedder {
    embeddings: HashMap<String, Vec<f32>>,
    dimension: usize,
    /// OOV (out-of-vocabulary) embedding - average of all embeddings
    oov_embedding: Vec<f32>,
}

impl GloVeEmbedder {
    /// Load GloVe embeddings from a text file
    /// 
    /// # Arguments
    /// * `path` - Path to GloVe text file (e.g., glove.6B.100d.txt)
    /// * `max_words` - Maximum number of words to load (None = all)
    /// 
    /// # Example
    /// ```ignore
    /// let embedder = GloVeEmbedder::load("glove.6B.100d.txt", Some(50000))?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P, max_words: Option<usize>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        let mut dimension = 0;
        let mut sum_embedding: Vec<f32> = Vec::new();
        let mut count = 0;

        for (i, line) in reader.lines().enumerate() {
            if let Some(max) = max_words {
                if i >= max {
                    break;
                }
            }

            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.len() < 2 {
                continue;
            }

            let word = parts[0].to_lowercase();
            let values: Vec<f32> = parts[1..]
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();

            if dimension == 0 {
                dimension = values.len();
                sum_embedding = vec![0.0; dimension];
            }

            if values.len() == dimension {
                // Add to running sum for OOV embedding
                for (i, v) in values.iter().enumerate() {
                    sum_embedding[i] += v;
                }
                count += 1;
                
                embeddings.insert(word, values);
            }
        }

        // Calculate OOV embedding as average
        let oov_embedding: Vec<f32> = if count > 0 {
            sum_embedding.iter().map(|v| v / count as f32).collect()
        } else {
            vec![0.0; dimension]
        };

        println!("📚 Loaded {} GloVe embeddings ({}d)", embeddings.len(), dimension);

        Ok(Self {
            embeddings,
            dimension,
            oov_embedding,
        })
    }

    /// Create a small test embedder with hardcoded common word embeddings
    /// (for testing without downloading GloVe files)
    pub fn test_embedder() -> Self {
        let dimension = 50;
        let mut embeddings = HashMap::new();
        
        // Add some common words with pseudo-embeddings based on semantic categories
        // These are NOT real GloVe vectors, just for testing structure
        
        // Programming languages - similar vectors
        embeddings.insert("rust".to_string(), Self::make_category_vec(dimension, 0, 0.8));
        embeddings.insert("python".to_string(), Self::make_category_vec(dimension, 0, 0.75));
        embeddings.insert("programming".to_string(), Self::make_category_vec(dimension, 0, 0.9));
        embeddings.insert("code".to_string(), Self::make_category_vec(dimension, 0, 0.7));
        embeddings.insert("language".to_string(), Self::make_category_vec(dimension, 0, 0.5));
        
        // Memory/safety - similar vectors
        embeddings.insert("memory".to_string(), Self::make_category_vec(dimension, 1, 0.85));
        embeddings.insert("safety".to_string(), Self::make_category_vec(dimension, 1, 0.8));
        embeddings.insert("ownership".to_string(), Self::make_category_vec(dimension, 1, 0.75));
        embeddings.insert("management".to_string(), Self::make_category_vec(dimension, 1, 0.6));
        
        // ML/AI - similar vectors
        embeddings.insert("machine".to_string(), Self::make_category_vec(dimension, 2, 0.8));
        embeddings.insert("learning".to_string(), Self::make_category_vec(dimension, 2, 0.85));
        embeddings.insert("data".to_string(), Self::make_category_vec(dimension, 2, 0.7));
        embeddings.insert("science".to_string(), Self::make_category_vec(dimension, 2, 0.65));
        embeddings.insert("mlx".to_string(), Self::make_category_vec(dimension, 2, 0.9));
        
        // Apple/hardware - similar vectors
        embeddings.insert("apple".to_string(), Self::make_category_vec(dimension, 3, 0.85));
        embeddings.insert("silicon".to_string(), Self::make_category_vec(dimension, 3, 0.8));
        embeddings.insert("gpu".to_string(), Self::make_category_vec(dimension, 3, 0.75));
        embeddings.insert("hardware".to_string(), Self::make_category_vec(dimension, 3, 0.7));
        
        // General words
        embeddings.insert("uses".to_string(), Self::make_category_vec(dimension, 4, 0.3));
        embeddings.insert("great".to_string(), Self::make_category_vec(dimension, 4, 0.4));
        embeddings.insert("runs".to_string(), Self::make_category_vec(dimension, 4, 0.35));
        
        // Calculate OOV as average
        let mut sum = vec![0.0f32; dimension];
        for emb in embeddings.values() {
            for (i, v) in emb.iter().enumerate() {
                sum[i] += v;
            }
        }
        let count = embeddings.len() as f32;
        let oov_embedding: Vec<f32> = sum.iter().map(|v| v / count).collect();
        
        Self {
            embeddings,
            dimension,
            oov_embedding,
        }
    }

    /// Helper to create pseudo-embeddings for testing
    fn make_category_vec(dim: usize, category: usize, strength: f32) -> Vec<f32> {
        let mut vec = vec![0.0f32; dim];
        // Set category-specific dimensions high
        let start = (category * 10) % dim;
        for i in 0..10 {
            let idx = (start + i) % dim;
            vec[idx] = strength * (1.0 - (i as f32 * 0.05));
        }
        // Add some noise
        for i in 0..dim {
            vec[i] += (i as f32 * 0.01).sin() * 0.1;
        }
        normalize(&mut vec);
        vec
    }

    /// Get embedding for a single word
    pub fn get_word_embedding(&self, word: &str) -> &[f32] {
        self.embeddings
            .get(&word.to_lowercase())
            .map(|v| v.as_slice())
            .unwrap_or(&self.oov_embedding)
    }

    /// Check if word is in vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.embeddings.contains_key(&word.to_lowercase())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.embeddings.len()
    }
}

impl Embedder for GloVeEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        
        if tokens.is_empty() {
            return self.oov_embedding.clone();
        }

        // Average word embeddings
        let mut sum = vec![0.0f32; self.dimension];
        let mut count = 0;

        for token in &tokens {
            let emb = self.get_word_embedding(token);
            for (i, v) in emb.iter().enumerate() {
                sum[i] += v;
            }
            count += 1;
        }

        if count > 0 {
            for v in sum.iter_mut() {
                *v /= count as f32;
            }
        }

        normalize(&mut sum);
        sum
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine_similarity;

    #[test]
    fn test_glove_test_embedder() {
        let embedder = GloVeEmbedder::test_embedder();
        
        let rust_emb = embedder.embed("rust programming");
        let python_emb = embedder.embed("python programming");
        let ml_emb = embedder.embed("machine learning");
        
        // Rust and Python should be similar (both programming)
        let sim_programming = cosine_similarity(&rust_emb, &python_emb);
        
        // Rust and ML should be less similar
        let sim_different = cosine_similarity(&rust_emb, &ml_emb);
        
        println!("Rust-Python similarity: {:.3}", sim_programming);
        println!("Rust-ML similarity: {:.3}", sim_different);
        
        // Programming languages should be more similar to each other
        assert!(sim_programming > sim_different);
    }
}
