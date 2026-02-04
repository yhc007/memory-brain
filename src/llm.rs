//! LLM Integration Module
//! 
//! Supports multiple backends:
//! - Local MLX models (via mlx-lm CLI)
//! - Ollama (local API)
//! - OpenAI-compatible APIs

use std::process::Command;

/// LLM provider trait
pub trait LlmProvider: Send + Sync {
    /// Generate a response for a prompt
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>>;
    
    /// Get provider name
    fn name(&self) -> &str;
}

/// MLX-LM provider (uses Python mlx-lm CLI)
pub struct MlxLmProvider {
    model: String,
}

impl MlxLmProvider {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }

    /// Get the Python path (uses venv if available)
    fn python_path() -> String {
        let venv_python = dirs::home_dir()
            .map(|h| h.join(".venvs/mlx-lm/bin/python3"))
            .filter(|p| p.exists())
            .map(|p| p.to_string_lossy().to_string());
        
        venv_python.unwrap_or_else(|| "python3".to_string())
    }

    /// Check if mlx-lm is available
    pub fn is_available() -> bool {
        Command::new(Self::python_path())
            .args(["-c", "import mlx_lm"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

impl LlmProvider for MlxLmProvider {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        // Escape the prompt for Python
        let escaped_prompt = prompt
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n");

        let python_code = format!(r#"
from mlx_lm import load, generate

model, tokenizer = load("{model}")
response = generate(
    model, 
    tokenizer, 
    prompt="{prompt}",
    max_tokens={max_tokens},
    verbose=False
)
print(response)
"#, model=self.model, prompt=escaped_prompt, max_tokens=max_tokens);

        let output = Command::new(Self::python_path())
            .args(["-c", &python_code])
            .output()?;

        if output.status.success() {
            let response = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // Remove the prompt from response if it's included
            let response = if response.starts_with(&prompt[..prompt.len().min(20)]) {
                response[prompt.len()..].trim().to_string()
            } else {
                response
            };
            Ok(response)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("MLX-LM error: {}", stderr).into())
        }
    }

    fn name(&self) -> &str {
        "mlx-lm"
    }
}

/// Ollama provider (local API)
pub struct OllamaProvider {
    model: String,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            base_url: "http://localhost:11434".to_string(),
        }
    }

    pub fn with_url(model: &str, base_url: &str) -> Self {
        Self {
            model: model.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Check if Ollama is running
    pub fn is_available() -> bool {
        reqwest_sync_get("http://localhost:11434/api/tags").is_ok()
    }
}

impl LlmProvider for OllamaProvider {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "num_predict": max_tokens
            }
        });

        // Use curl for simplicity (avoid adding reqwest dependency)
        let output = Command::new("curl")
            .args([
                "-s",
                "-X", "POST",
                &url,
                "-H", "Content-Type: application/json",
                "-d", &body.to_string(),
            ])
            .output()?;

        if output.status.success() {
            let response: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            Ok(response["response"].as_str().unwrap_or("").to_string())
        } else {
            Err("Ollama request failed".into())
        }
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

/// OpenAI-compatible API provider
pub struct OpenAIProvider {
    model: String,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(model: &str, api_key: &str) -> Self {
        Self {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_base_url(model: &str, api_key: &str, base_url: &str) -> Self {
        Self {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable
    pub fn from_env(model: &str) -> Option<Self> {
        std::env::var("OPENAI_API_KEY").ok().map(|key| Self::new(model, &key))
    }
}

impl LlmProvider for OpenAIProvider {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        });

        let output = Command::new("curl")
            .args([
                "-s",
                "-X", "POST",
                &url,
                "-H", "Content-Type: application/json",
                "-H", &format!("Authorization: Bearer {}", self.api_key),
                "-d", &body.to_string(),
            ])
            .output()?;

        if output.status.success() {
            let response: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            let content = response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("");
            Ok(content.to_string())
        } else {
            Err("OpenAI request failed".into())
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

/// Simple command-based LLM (for testing)
pub struct EchoProvider;

impl LlmProvider for EchoProvider {
    fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("[Echo] {}", prompt.chars().take(100).collect::<String>()))
    }

    fn name(&self) -> &str {
        "echo"
    }
}

// ============ Memory-Aware Chat ============

use crate::Brain;

/// Memory-augmented LLM chat
pub struct MemoryChat {
    brain: Brain,
    llm: Box<dyn LlmProvider>,
    system_prompt: String,
    memory_limit: usize,
}

impl MemoryChat {
    pub fn new(brain: Brain, llm: Box<dyn LlmProvider>) -> Self {
        Self {
            brain,
            llm,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            memory_limit: 5,
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Chat with memory-augmented context
    pub fn chat(&mut self, user_input: &str) -> Result<String, Box<dyn std::error::Error>> {
        // 1. Recall relevant memories
        let memories = self.brain.recall(user_input, self.memory_limit);
        
        // Debug
        if std::env::var("DEBUG").is_ok() {
            eprintln!("=== RECALL for '{}' ===", user_input);
            eprintln!("Found {} memories", memories.len());
            for m in &memories {
                eprintln!("  - {}", m.content);
            }
        }
        
        // 2. Build context from memories
        let memory_context = if memories.is_empty() {
            "No relevant memories found.".to_string()
        } else {
            let mem_texts: Vec<String> = memories
                .iter()
                .map(|m| format!("- {}", m.content))
                .collect();
            format!("Relevant memories about the user:\n{}", mem_texts.join("\n"))
        };

        // 3. Build full prompt (Llama 3 format)
        let full_prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            self.system_prompt,
            memory_context,
            user_input
        );

        // Debug: print prompt if DEBUG env var is set
        if std::env::var("DEBUG").is_ok() {
            eprintln!("\n=== PROMPT ===\n{}\n=== END PROMPT ===\n", full_prompt);
        }

        // 4. Generate response (shorter for better results)
        let response = self.llm.generate(&full_prompt, 200)?;
        
        // 5. Clean up response (remove any continuation markers)
        let response = response
            .split("<|eot_id|>")
            .next()
            .unwrap_or(&response)
            .split("<|start_header_id|>")
            .next()
            .unwrap_or(&response)
            .trim()
            .to_string();

        // 5. Store the interaction as episodic memory
        let interaction = format!("User asked: {} | Response: {}", 
            truncate(user_input, 50), 
            truncate(&response, 100)
        );
        self.brain.process(&interaction, Some("chat"))?;

        Ok(response)
    }

    /// Ask the LLM to summarize memories on a topic
    pub fn summarize_memories(&mut self, topic: &str) -> Result<String, Box<dyn std::error::Error>> {
        let memories = self.brain.recall(topic, 10);
        
        if memories.is_empty() {
            return Ok(format!("No memories found about: {}", topic));
        }

        let mem_texts: Vec<String> = memories
            .iter()
            .map(|m| format!("- {}", m.content))
            .collect();

        let prompt = format!(
            "Summarize these memories about '{}':\n{}\n\nSummary:",
            topic,
            mem_texts.join("\n")
        );

        self.llm.generate(&prompt, 200)
    }

    /// Extract and store key facts from text
    pub fn extract_and_store(&mut self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExtract 1-5 key facts from the text. Each fact should be a short, complete sentence. Output ONLY the facts, one per line, starting with '- '. Stop after listing the facts.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            text
        );

        let response = self.llm.generate(&prompt, 150)?;
        
        // Clean up response
        let response = response
            .split("<|eot_id|>")
            .next()
            .unwrap_or(&response)
            .trim();
        
        let facts: Vec<String> = response
            .lines()
            .filter(|l| l.trim().starts_with('-') || l.trim().starts_with('•') || l.trim().starts_with('*'))
            .map(|l| l.trim().trim_start_matches(|c| c == '-' || c == '•' || c == '*').trim().to_string())
            .filter(|l| !l.is_empty() && l.len() > 5)
            .take(5) // Max 5 facts
            .collect();

        // Store each fact
        for fact in &facts {
            self.brain.process(fact, Some("extracted"))?;
        }

        Ok(facts)
    }

    /// Get brain reference
    pub fn brain(&self) -> &Brain {
        &self.brain
    }

    /// Get mutable brain reference
    pub fn brain_mut(&mut self) -> &mut Brain {
        &mut self.brain
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a helpful AI assistant with a memory system.
When relevant memories are provided, use them to personalize your response.
Be concise and directly answer the question. Do not continue the conversation or ask follow-up questions."#;

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

/// Simple sync HTTP GET (for checking Ollama)
fn reqwest_sync_get(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", url])
        .output()?;
    
    let code = String::from_utf8_lossy(&output.stdout);
    if code.starts_with('2') {
        Ok(())
    } else {
        Err("Request failed".into())
    }
}

// ============ Auto-detect best provider ============

/// Auto-detect the best available LLM provider
pub fn auto_detect_provider() -> Box<dyn LlmProvider> {
    // 1. Check for Ollama (most common local option)
    if OllamaProvider::is_available() {
        println!("🦙 Using Ollama");
        return Box::new(OllamaProvider::new("llama3.2"));
    }

    // 2. Check for MLX-LM
    if MlxLmProvider::is_available() {
        println!("🍎 Using MLX-LM");
        return Box::new(MlxLmProvider::new("mlx-community/Llama-3.2-1B-Instruct-4bit"));
    }

    // 3. Check for OpenAI API key
    if let Some(provider) = OpenAIProvider::from_env("gpt-4o-mini") {
        println!("🤖 Using OpenAI API");
        return Box::new(provider);
    }

    // 4. Fallback to echo
    println!("⚠️ No LLM found, using echo mode");
    Box::new(EchoProvider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_provider() {
        let provider = EchoProvider;
        let response = provider.generate("Hello world", 100).unwrap();
        assert!(response.contains("Hello world"));
    }
}
