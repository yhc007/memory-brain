//! CLIP ONNX Provider - Run CLIP models via ONNX Runtime
//! 
//! Uses pre-exported ONNX models for cross-platform CLIP inference.
//! Models can be downloaded from Hugging Face or exported from PyTorch.

use crate::visual::{ClipError, ClipProvider};
use std::path::Path;

#[cfg(feature = "clip")]
use ort::{Environment, Session, SessionBuilder, Value};

#[cfg(feature = "clip")]
use image::{DynamicImage, GenericImageView};

/// ONNX-based CLIP provider
pub struct ClipOnnx {
    #[cfg(feature = "clip")]
    image_session: Session,
    #[cfg(feature = "clip")]
    text_session: Session,
    #[allow(dead_code)]
    embedding_dim: usize,
}

#[cfg(feature = "clip")]
impl ClipOnnx {
    /// Create a new CLIP ONNX provider
    /// 
    /// # Arguments
    /// * `model_dir` - Directory containing clip_image.onnx and clip_text.onnx
    pub fn new(model_dir: &Path) -> Result<Self, ClipError> {
        let env = Environment::builder()
            .with_name("clip")
            .build()
            .map_err(|e| ClipError::ModelError(format!("Failed to create ONNX environment: {}", e)))?;
        
        let image_model_path = model_dir.join("clip_image.onnx");
        let text_model_path = model_dir.join("clip_text.onnx");
        
        if !image_model_path.exists() {
            return Err(ClipError::ModelError(format!(
                "Image model not found: {:?}", image_model_path
            )));
        }
        
        if !text_model_path.exists() {
            return Err(ClipError::ModelError(format!(
                "Text model not found: {:?}", text_model_path
            )));
        }
        
        let image_session = SessionBuilder::new(&env)
            .map_err(|e| ClipError::ModelError(e.to_string()))?
            .with_model_from_file(&image_model_path)
            .map_err(|e| ClipError::ModelError(e.to_string()))?;
        
        let text_session = SessionBuilder::new(&env)
            .map_err(|e| ClipError::ModelError(e.to_string()))?
            .with_model_from_file(&text_model_path)
            .map_err(|e| ClipError::ModelError(e.to_string()))?;
        
        Ok(Self {
            image_session,
            text_session,
            embedding_dim: 512, // CLIP ViT-B/32 default
        })
    }
    
    /// Preprocess image for CLIP (resize to 224x224, normalize)
    fn preprocess_image(&self, image_path: &Path) -> Result<Vec<f32>, ClipError> {
        let img = image::open(image_path)
            .map_err(|e| ClipError::ImageError(format!("Failed to open image: {}", e)))?;
        
        // Resize to 224x224 (CLIP input size)
        let resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
        
        // Convert to RGB and normalize
        let rgb = resized.to_rgb8();
        let mut pixels: Vec<f32> = Vec::with_capacity(3 * 224 * 224);
        
        // CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std = [0.26862954f32, 0.26130258, 0.27577711];
        
        // CHW format (channels first)
        for c in 0..3 {
            for y in 0..224 {
                for x in 0..224 {
                    let pixel = rgb.get_pixel(x, y);
                    let value = pixel[c] as f32 / 255.0;
                    let normalized = (value - mean[c]) / std[c];
                    pixels.push(normalized);
                }
            }
        }
        
        Ok(pixels)
    }
}

#[cfg(feature = "clip")]
impl ClipProvider for ClipOnnx {
    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>, ClipError> {
        let pixels = self.preprocess_image(image_path)?;
        
        // Create input tensor [1, 3, 224, 224]
        let input = Value::from_array(([1usize, 3, 224, 224], pixels.as_slice()))
            .map_err(|e| ClipError::EncodingError(e.to_string()))?;
        
        let outputs = self.image_session
            .run(vec![input])
            .map_err(|e| ClipError::ModelError(format!("Inference failed: {}", e)))?;
        
        let output = outputs[0]
            .try_extract::<f32>()
            .map_err(|e| ClipError::EncodingError(e.to_string()))?;
        
        let embedding: Vec<f32> = output.view().iter().copied().collect();
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }
    
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, ClipError> {
        // Simple tokenization (in production, use proper CLIP tokenizer)
        // This is a placeholder - real implementation needs BPE tokenizer
        let tokens = simple_tokenize(text, 77);
        
        let input = Value::from_array(([1usize, 77], tokens.as_slice()))
            .map_err(|e| ClipError::EncodingError(e.to_string()))?;
        
        let outputs = self.text_session
            .run(vec![input])
            .map_err(|e| ClipError::ModelError(format!("Text inference failed: {}", e)))?;
        
        let output = outputs[0]
            .try_extract::<f32>()
            .map_err(|e| ClipError::EncodingError(e.to_string()))?;
        
        let embedding: Vec<f32> = output.view().iter().copied().collect();
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }
    
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Placeholder tokenizer (real CLIP uses BPE)
#[cfg(feature = "clip")]
fn simple_tokenize(text: &str, max_len: usize) -> Vec<i64> {
    let mut tokens = vec![49406i64]; // <|startoftext|>
    
    // Very simple word-to-id mapping (placeholder)
    for word in text.split_whitespace().take(max_len - 2) {
        // Simple hash-based token id (NOT real CLIP tokenization!)
        let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        tokens.push((hash % 49405 + 1) as i64);
    }
    
    tokens.push(49407); // <|endoftext|>
    
    // Pad to max_len
    tokens.resize(max_len, 0);
    tokens
}

// Fallback implementation when CLIP feature is disabled
#[cfg(not(feature = "clip"))]
impl ClipOnnx {
    pub fn new(_model_dir: &Path) -> Result<Self, ClipError> {
        Err(ClipError::ModelError("CLIP feature not enabled. Rebuild with --features clip".to_string()))
    }
}

#[cfg(not(feature = "clip"))]
impl ClipProvider for ClipOnnx {
    fn embed_image(&self, _image_path: &Path) -> Result<Vec<f32>, ClipError> {
        Err(ClipError::ModelError("CLIP not enabled".to_string()))
    }
    
    fn embed_text(&self, _text: &str) -> Result<Vec<f32>, ClipError> {
        Err(ClipError::ModelError("CLIP not enabled".to_string()))
    }
    
    fn embedding_dim(&self) -> usize {
        512
    }
}

/// Mock CLIP provider for testing (generates random embeddings)
pub struct MockClipProvider {
    dim: usize,
}

impl MockClipProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl ClipProvider for MockClipProvider {
    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>, ClipError> {
        // Generate deterministic "embedding" based on file path
        let hash = image_path.to_string_lossy().bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        
        let mut embedding = Vec::with_capacity(self.dim);
        let mut seed = hash;
        for _ in 0..self.dim {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            embedding.push((seed as f32 / u64::MAX as f32) * 2.0 - 1.0);
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }
    
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, ClipError> {
        // Generate deterministic "embedding" based on text
        let hash = text.bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        
        let mut embedding = Vec::with_capacity(self.dim);
        let mut seed = hash;
        for _ in 0..self.dim {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            embedding.push((seed as f32 / u64::MAX as f32) * 2.0 - 1.0);
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }
    
    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// HTTP-based CLIP provider that connects to clip_server.py
pub struct ClipServerProvider {
    server_url: String,
    dim: usize,
}

impl ClipServerProvider {
    /// Create a new CLIP server provider
    /// 
    /// # Arguments
    /// * `server_url` - Base URL of the CLIP server (e.g., "http://localhost:5050")
    pub fn new(server_url: &str) -> Result<Self, ClipError> {
        let provider = Self {
            server_url: server_url.trim_end_matches('/').to_string(),
            dim: 512,
        };
        
        // Check health
        provider.check_health()?;
        
        Ok(provider)
    }
    
    /// Check if the CLIP server is healthy
    pub fn check_health(&self) -> Result<(), ClipError> {
        let url = format!("{}/health", self.server_url);
        
        let response: serde_json::Value = ureq::get(&url)
            .call()
            .map_err(|e| ClipError::ModelError(format!("Server connection failed: {}", e)))?
            .into_json()
            .map_err(|e| ClipError::ModelError(format!("Invalid response: {}", e)))?;
        
        if response.get("status").and_then(|s| s.as_str()) != Some("ok") {
            return Err(ClipError::ModelError("Server not healthy".to_string()));
        }
        
        Ok(())
    }
}

impl ClipProvider for ClipServerProvider {
    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>, ClipError> {
        let url = format!("{}/embed/image", self.server_url);
        let path_str = image_path.to_string_lossy().to_string();
        
        let body = serde_json::json!({
            "path": path_str
        });
        
        let response: serde_json::Value = ureq::post(&url)
            .send_json(body)
            .map_err(|e| ClipError::ModelError(format!("Request failed: {}", e)))?
            .into_json()
            .map_err(|e| ClipError::EncodingError(format!("Invalid response: {}", e)))?;
        
        if let Some(error) = response.get("error") {
            return Err(ClipError::ImageError(error.to_string()));
        }
        
        let embedding: Vec<f32> = response
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| ClipError::EncodingError("No embedding in response".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        
        if embedding.len() != self.dim {
            return Err(ClipError::EncodingError(format!(
                "Expected {} dims, got {}", self.dim, embedding.len()
            )));
        }
        
        Ok(embedding)
    }
    
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, ClipError> {
        let url = format!("{}/embed/text", self.server_url);
        
        let body = serde_json::json!({
            "text": text
        });
        
        let response: serde_json::Value = ureq::post(&url)
            .send_json(body)
            .map_err(|e| ClipError::ModelError(format!("Request failed: {}", e)))?
            .into_json()
            .map_err(|e| ClipError::EncodingError(format!("Invalid response: {}", e)))?;
        
        if let Some(error) = response.get("error") {
            return Err(ClipError::ModelError(error.to_string()));
        }
        
        let embedding: Vec<f32> = response
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| ClipError::EncodingError("No embedding in response".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        
        if embedding.len() != self.dim {
            return Err(ClipError::EncodingError(format!(
                "Expected {} dims, got {}", self.dim, embedding.len()
            )));
        }
        
        Ok(embedding)
    }
    
    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_provider() {
        let provider = MockClipProvider::new(512);
        
        let emb1 = provider.embed_image(Path::new("/test/a.jpg")).unwrap();
        let emb2 = provider.embed_image(Path::new("/test/a.jpg")).unwrap();
        let emb3 = provider.embed_image(Path::new("/test/b.jpg")).unwrap();
        
        assert_eq!(emb1.len(), 512);
        assert_eq!(emb1, emb2); // Same path = same embedding
        assert_ne!(emb1, emb3); // Different path = different embedding
    }
}
