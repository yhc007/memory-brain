//! VLM (Vision Language Model) Module
//! 
//! Provides image-to-text description generation using local VLM models via Ollama.
//! Used to automatically generate descriptions for visual memories.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// VLM Provider trait for image description generation
pub trait VlmProvider: Send + Sync {
    /// Generate a description for an image
    fn describe_image(&self, image_path: &Path, prompt: Option<&str>) -> Result<String, VlmError>;
    
    /// Get the model name
    fn model_name(&self) -> &str;
}

/// VLM errors
#[derive(Debug)]
pub enum VlmError {
    /// Image file not found or unreadable
    ImageError(String),
    /// Model not available
    ModelError(String),
    /// API/connection error
    ApiError(String),
    /// Timeout
    Timeout,
}

impl std::fmt::Display for VlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VlmError::ImageError(msg) => write!(f, "Image error: {}", msg),
            VlmError::ModelError(msg) => write!(f, "Model error: {}", msg),
            VlmError::ApiError(msg) => write!(f, "API error: {}", msg),
            VlmError::Timeout => write!(f, "VLM request timed out"),
        }
    }
}

impl std::error::Error for VlmError {}

/// Ollama VLM request
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    images: Vec<String>,  // base64 encoded images
    stream: bool,
}

/// Ollama VLM response
#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

/// Ollama-based VLM provider
pub struct OllamaVlm {
    /// Ollama API URL (default: http://localhost:11434)
    api_url: String,
    /// Model name (e.g., "llava:7b", "llava:13b")
    model: String,
    /// Request timeout
    timeout: Duration,
}

impl OllamaVlm {
    /// Create a new Ollama VLM provider
    pub fn new(model: &str) -> Self {
        Self {
            api_url: "http://localhost:11434".to_string(),
            model: model.to_string(),
            timeout: Duration::from_secs(120),
        }
    }
    
    /// Create with custom API URL
    pub fn with_url(model: &str, api_url: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            model: model.to_string(),
            timeout: Duration::from_secs(120),
        }
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Read and encode image as base64
    fn encode_image(&self, image_path: &Path) -> Result<String, VlmError> {
        use std::fs;
        use std::io::Read;
        
        let mut file = fs::File::open(image_path)
            .map_err(|e| VlmError::ImageError(format!("Failed to open image: {}", e)))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| VlmError::ImageError(format!("Failed to read image: {}", e)))?;
        
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        Ok(STANDARD.encode(&buffer))
    }
}

impl VlmProvider for OllamaVlm {
    fn describe_image(&self, image_path: &Path, prompt: Option<&str>) -> Result<String, VlmError> {
        // Check if image exists
        if !image_path.exists() {
            return Err(VlmError::ImageError(format!(
                "Image not found: {:?}", image_path
            )));
        }
        
        // Encode image
        let image_base64 = self.encode_image(image_path)?;
        
        // Default prompt for memory-brain
        let default_prompt = "Describe this image in detail. Include: \
            1) Main subjects and objects \
            2) Setting/location if visible \
            3) Actions or activities happening \
            4) Notable colors, textures, or visual elements \
            5) Any text visible in the image \
            Be concise but comprehensive.";
        
        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.unwrap_or(default_prompt).to_string(),
            images: vec![image_base64],
            stream: false,
        };
        
        // Make request to Ollama
        let url = format!("{}/api/generate", self.api_url);
        
        let response = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&request)
            .map_err(|e| VlmError::ApiError(format!("Request failed: {}", e)))?;
        
        let ollama_response: OllamaResponse = response
            .into_json()
            .map_err(|e| VlmError::ApiError(format!("Failed to parse response: {}", e)))?;
        
        Ok(ollama_response.response.trim().to_string())
    }
    
    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Check if Ollama is running and has the specified model
pub fn check_ollama_model(model: &str) -> Result<bool, VlmError> {
    let url = "http://localhost:11434/api/tags";
    
    #[derive(Deserialize)]
    struct TagsResponse {
        models: Vec<ModelInfo>,
    }
    
    #[derive(Deserialize)]
    struct ModelInfo {
        name: String,
    }
    
    let response = ureq::get(url)
        .timeout(Duration::from_secs(5))
        .call()
        .map_err(|e| VlmError::ApiError(format!("Failed to connect to Ollama: {}", e)))?;
    
    let tags: TagsResponse = response
        .into_json()
        .map_err(|e| VlmError::ApiError(format!("Failed to parse response: {}", e)))?;
    
    // Check if model exists (handle both "llava:7b" and "llava" formats)
    let model_base = model.split(':').next().unwrap_or(model);
    
    Ok(tags.models.iter().any(|m| {
        m.name == model || m.name.starts_with(&format!("{}:", model_base))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vlm_error_display() {
        let err = VlmError::ImageError("test".to_string());
        assert!(err.to_string().contains("Image error"));
    }
}
