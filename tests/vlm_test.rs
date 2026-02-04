//! VLM Integration Test

use memory_brain::vlm::{OllamaVlm, VlmProvider, check_ollama_model};
use std::path::Path;

#[test]
fn test_check_ollama_model() {
    // Should have llava:7b installed
    match check_ollama_model("llava:7b") {
        Ok(has_model) => {
            println!("LLaVA model available: {}", has_model);
            assert!(has_model, "LLaVA model should be installed");
        }
        Err(e) => {
            eprintln!("Ollama connection error: {}", e);
            // Skip test if Ollama not running
        }
    }
}

#[test]
fn test_vlm_describe_image() {
    let vlm = OllamaVlm::new("llava:7b");
    let test_image = Path::new("/tmp/test_fox.jpg");
    
    if !test_image.exists() {
        eprintln!("Test image not found, skipping");
        return;
    }
    
    match vlm.describe_image(test_image, Some("What animal is in this image? Answer in one word.")) {
        Ok(description) => {
            println!("VLM description: {}", description);
            let lower = description.to_lowercase();
            assert!(
                lower.contains("fox") || lower.contains("animal") || lower.contains("canine"),
                "Description should mention the fox"
            );
        }
        Err(e) => {
            eprintln!("VLM error: {}", e);
            // Don't fail if Ollama not running
        }
    }
}
