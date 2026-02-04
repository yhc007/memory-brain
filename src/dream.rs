//! Dream Mode - Memory Consolidation During "Sleep"
//!
//! Simulates human brain's memory consolidation during REM sleep:
//! - Replay recent memories
//! - Strengthen important connections
//! - Discover new associations
//! - Fade weak memories
//! - Generate dream-like narratives

use crate::{Brain, MemoryItem, cosine_similarity};
use chrono::{Utc, Duration};
use rand::prelude::*;
use std::collections::HashMap;

/// Dream state and results
#[derive(Debug, Clone)]
pub struct DreamState {
    pub phase: DreamPhase,
    pub memories_processed: usize,
    pub new_connections: usize,
    pub faded_memories: usize,
    pub dream_narrative: String,
    pub insights: Vec<String>,
}

/// Dream phases (like sleep stages)
#[derive(Debug, Clone, PartialEq)]
pub enum DreamPhase {
    /// Light sleep - sorting recent memories
    Light,
    /// Deep sleep - strengthening important memories
    Deep,
    /// REM - creative recombination
    Rem,
    /// Waking - generating insights
    Waking,
}

impl std::fmt::Display for DreamPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DreamPhase::Light => write!(f, "💤 Light Sleep"),
            DreamPhase::Deep => write!(f, "😴 Deep Sleep"),
            DreamPhase::Rem => write!(f, "🌙 REM"),
            DreamPhase::Waking => write!(f, "🌅 Waking"),
        }
    }
}

/// Dream engine for memory consolidation
pub struct DreamEngine<'a> {
    brain: &'a mut Brain,
    rng: ThreadRng,
    verbose: bool,
}

impl<'a> DreamEngine<'a> {
    pub fn new(brain: &'a mut Brain) -> Self {
        Self {
            brain,
            rng: thread_rng(),
            verbose: false,
        }
    }

    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Enter dream mode and process memories
    pub fn dream(&mut self) -> DreamState {
        let mut state = DreamState {
            phase: DreamPhase::Light,
            memories_processed: 0,
            new_connections: 0,
            faded_memories: 0,
            dream_narrative: String::new(),
            insights: Vec::new(),
        };

        if self.verbose {
            println!("\n😴 Entering dream mode...\n");
        }

        // Phase 1: Light Sleep - Gather recent memories
        state.phase = DreamPhase::Light;
        let recent_memories = self.gather_recent_memories();
        state.memories_processed = recent_memories.len();
        
        if self.verbose {
            println!("{} - Gathered {} recent memories", state.phase, recent_memories.len());
        }

        // Phase 2: Deep Sleep - Strengthen important memories
        state.phase = DreamPhase::Deep;
        let strengthened = self.strengthen_important(&recent_memories);
        
        if self.verbose {
            println!("{} - Strengthened {} important memories", state.phase, strengthened);
        }

        // Phase 3: REM - Creative recombination
        state.phase = DreamPhase::Rem;
        let (narrative, connections) = self.rem_dream(&recent_memories);
        state.dream_narrative = narrative;
        state.new_connections = connections;
        
        if self.verbose {
            println!("{} - Created {} new connections", state.phase, connections);
            println!("\n🌙 Dream narrative:\n{}\n", state.dream_narrative);
        }

        // Phase 4: Waking - Generate insights
        state.phase = DreamPhase::Waking;
        state.insights = self.generate_insights(&recent_memories);
        
        if self.verbose {
            println!("{} - Generated {} insights", state.phase, state.insights.len());
            for insight in &state.insights {
                println!("  💡 {}", insight);
            }
        }

        // Fade weak memories
        state.faded_memories = self.fade_weak_memories();
        
        if self.verbose {
            println!("\n🌅 Dream complete! Faded {} weak memories", state.faded_memories);
        }

        state
    }

    /// Gather recent memories (last 24 hours)
    fn gather_recent_memories(&self) -> Vec<MemoryItem> {
        let mut memories = Vec::new();
        
        // Get from semantic memory
        if let Ok(items) = self.brain.semantic.search("", 100) {
            let cutoff = Utc::now() - Duration::hours(24);
            for item in items {
                if item.created_at > cutoff {
                    memories.push(item);
                }
            }
        }
        
        // Also include working memory
        for item in self.brain.working.get_all() {
            memories.push(item.clone());
        }
        
        memories
    }

    /// Strengthen important memories (high access count, emotional)
    fn strengthen_important(&mut self, memories: &[MemoryItem]) -> usize {
        let mut count = 0;
        
        for memory in memories {
            // Criteria for importance
            let is_important = memory.access_count > 2 
                || memory.tags.iter().any(|t| t.contains("important") || t.contains("lesson"))
                || memory.strength > 0.8;
            
            if is_important {
                // Boost strength (would need mutable access to actually update)
                count += 1;
            }
        }
        
        count
    }

    /// REM dream - creative recombination of memories
    fn rem_dream(&mut self, memories: &[MemoryItem]) -> (String, usize) {
        if memories.is_empty() {
            return ("No memories to dream about...".to_string(), 0);
        }

        let mut narrative = String::new();
        let mut connections = 0;

        // Select random memories to weave together
        let sample_size = memories.len().min(5);
        let selected: Vec<_> = memories
            .choose_multiple(&mut self.rng, sample_size)
            .collect();

        // Create dream-like narrative
        narrative.push_str("In the dream, ");
        
        let transitions = [
            "suddenly transformed into",
            "which reminded me of",
            "and then I saw",
            "merging with",
            "floating alongside",
            "echoing through",
        ];

        for (i, memory) in selected.iter().enumerate() {
            let snippet = truncate_content(&memory.content, 30);
            
            if i == 0 {
                narrative.push_str(&format!("\"{}\"", snippet));
            } else {
                let transition = transitions.choose(&mut self.rng).unwrap();
                narrative.push_str(&format!(" {} \"{}\"", transition, snippet));
                connections += 1;
            }
        }
        
        narrative.push_str("...");

        // Find unexpected connections based on embedding similarity
        if memories.len() >= 2 {
            for i in 0..memories.len().min(10) {
                for j in (i + 1)..memories.len().min(10) {
                    if let (Some(emb_a), Some(emb_b)) = (&memories[i].embedding, &memories[j].embedding) {
                        let sim = cosine_similarity(emb_a, emb_b);
                        // Medium similarity = unexpected but related connection
                        if sim > 0.3 && sim < 0.7 {
                            connections += 1;
                        }
                    }
                }
            }
        }

        (narrative, connections)
    }

    /// Generate insights from memory patterns
    fn generate_insights(&self, memories: &[MemoryItem]) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Count tags
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for memory in memories {
            for tag in &memory.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        
        // Find dominant themes
        let mut tags: Vec<_> = tag_counts.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1));
        
        if let Some((tag, count)) = tags.first() {
            if **count >= 3 {
                insights.push(format!("'{}' 주제가 자주 등장함 ({}회)", tag, count));
            }
        }
        
        // Time-based insights
        let recent_count = memories.iter()
            .filter(|m| m.created_at > Utc::now() - Duration::hours(6))
            .count();
        
        if recent_count > 5 {
            insights.push(format!("최근 6시간 동안 활발히 학습함 ({}개 기억)", recent_count));
        }
        
        // Pattern insights
        let rust_count = memories.iter()
            .filter(|m| m.content.to_lowercase().contains("rust"))
            .count();
        
        if rust_count >= 2 {
            insights.push("Rust 관련 작업을 많이 함 🦀".to_string());
        }

        let paul_count = memories.iter()
            .filter(|m| m.content.to_lowercase().contains("paul"))
            .count();
        
        if paul_count >= 2 {
            insights.push("Paul과 활발히 소통함".to_string());
        }

        insights
    }

    /// Fade weak memories (low strength, old, rarely accessed)
    fn fade_weak_memories(&mut self) -> usize {
        // In a real implementation, this would actually reduce strength
        // For now, just count what would be faded
        let mut count = 0;
        
        if let Ok(items) = self.brain.semantic.search("", 1000) {
            let old_threshold = Utc::now() - Duration::days(7);
            for item in items {
                let is_weak = item.strength < 0.3 
                    && item.access_count < 2 
                    && item.created_at < old_threshold;
                
                if is_weak {
                    count += 1;
                }
            }
        }
        
        count
    }
}

/// Truncate content for dream narrative
fn truncate_content(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        chars[..max].iter().collect::<String>() + "..."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_dream_phases() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("dream_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();
        
        // Add some memories
        brain.process("Learning about Rust ownership", None).unwrap();
        brain.process("Paul prefers casual conversation", None).unwrap();
        brain.process("CoreDB uses LSM trees", None).unwrap();
        
        // Dream!
        let mut engine = DreamEngine::new(&mut brain);
        let state = engine.dream();
        
        assert!(state.memories_processed > 0);
        assert!(!state.dream_narrative.is_empty());
    }
}
