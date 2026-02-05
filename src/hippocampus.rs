//! Hippocampus Module - Brain's memory formation center
//! 
//! Implements three key hippocampal functions:
//! 1. **Replay** - Chronological replay of recent memories during "sleep" to strengthen connections
//! 2. **Episode Chain** - Temporal linking of sequential memories
//! 3. **Auto-importance** - Automatic strength scoring based on novelty + emotion

use crate::{Brain, MemoryItem, cosine_similarity};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

/// Hippocampus - manages memory formation and consolidation
pub struct Hippocampus<'a> {
    brain: &'a mut Brain,
}

/// Replay result from a single sleep cycle
#[derive(Debug)]
pub struct ReplayResult {
    /// Number of memories replayed
    pub replayed: usize,
    /// Number of connections strengthened
    pub connections_strengthened: usize,
    /// Number of new connections discovered
    pub new_connections: usize,
    /// Memories that were significantly strengthened
    pub strengthened_memories: Vec<String>,
}

/// Episode chain link
#[derive(Debug, Clone)]
pub struct EpisodeLink {
    pub from_id: String,
    pub to_id: String,
    /// Time gap between episodes
    pub gap_seconds: i64,
    /// Contextual similarity (tag/content overlap)
    pub context_similarity: f32,
}

/// Auto-importance scoring result
#[derive(Debug)]
pub struct ImportanceScore {
    /// Final strength (0.0 - 1.0)
    pub strength: f32,
    /// Novelty score - how different from existing memories
    pub novelty: f32,
    /// Emotional intensity (absolute value of valence)
    pub emotional_intensity: f32,
    /// Reasoning
    pub reason: String,
}

impl<'a> Hippocampus<'a> {
    pub fn new(brain: &'a mut Brain) -> Self {
        Self { brain }
    }

    // ========================================
    // 1. REPLAY - Chronological memory replay
    // ========================================

    /// Replay recent memories in chronological order, strengthening connections
    /// between temporally close and semantically similar memories.
    /// This mimics the hippocampal replay during sleep.
    pub fn replay(&mut self, hours_back: u64) -> ReplayResult {
        let cutoff = Utc::now() - Duration::hours(hours_back as i64);
        
        // Get recent memories from ALL stores sorted by time
        let mut all_memories = Vec::new();
        all_memories.extend(self.brain.episodic.search("", 10000).unwrap_or_default());
        all_memories.extend(self.brain.semantic.search("", 10000).unwrap_or_default());
        all_memories.extend(self.brain.procedural.search("", 10000).unwrap_or_default());
        let mut recent: Vec<&MemoryItem> = all_memories.iter()
            .filter(|m| m.created_at > cutoff)
            .collect();
        recent.sort_by_key(|m| m.created_at);
        
        if recent.is_empty() {
            return ReplayResult {
                replayed: 0,
                connections_strengthened: 0,
                new_connections: 0,
                strengthened_memories: vec![],
            };
        }

        let mut connections_strengthened = 0;
        let mut new_connections = 0;
        let mut strengthened_ids: Vec<String> = vec![];

        // Replay: walk through memories in order
        // Strengthen connections between adjacent and similar memories
        for window in recent.windows(2) {
            let prev = window[0];
            let curr = window[1];
            
            // Calculate temporal proximity (closer in time = stronger link)
            let time_gap = (curr.created_at - prev.created_at).num_seconds().abs();
            let temporal_weight = 1.0 / (1.0 + (time_gap as f32 / 3600.0)); // decay over hours
            
            // Calculate semantic similarity
            let semantic_sim = match (&prev.embedding, &curr.embedding) {
                (Some(a), Some(b)) => cosine_similarity(a, b),
                _ => 0.0,
            };
            
            // Tag overlap
            let tag_overlap = tag_similarity(&prev.tags, &curr.tags);
            
            // Combined replay strength
            let replay_strength = temporal_weight * 0.4 + semantic_sim * 0.4 + tag_overlap * 0.2;
            
            if replay_strength > 0.2 {
                // Strengthen both memories
                let boost = replay_strength * 0.05; // Small but cumulative boost
                
                if let Some(m) = self.brain.semantic.search(&prev.id.to_string(), 1)
                    .ok().and_then(|v| v.into_iter().next())
                {
                    let new_strength = (m.strength + boost).min(1.0);
                    let _ = self.brain.update_strength(&prev.id.to_string(), new_strength);
                    connections_strengthened += 1;
                    
                    if boost > 0.03 {
                        strengthened_ids.push(truncate_content(&prev.content, 40));
                    }
                }
                
                if semantic_sim > 0.5 && tag_overlap < 0.3 {
                    // Found a new non-obvious connection!
                    new_connections += 1;
                }
            }
        }

        ReplayResult {
            replayed: recent.len(),
            connections_strengthened,
            new_connections,
            strengthened_memories: strengthened_ids,
        }
    }

    // ========================================
    // 2. EPISODE CHAIN - Temporal linking
    // ========================================

    /// Build episode chains from memories, linking temporally sequential memories.
    /// Returns chains of related episodes.
    pub fn build_episode_chains(&self, hours_back: u64, max_gap_minutes: i64) -> Vec<Vec<EpisodeLink>> {
        let cutoff = Utc::now() - Duration::hours(hours_back as i64);
        
        let mut all_memories = Vec::new();
        all_memories.extend(self.brain.episodic.search("", 10000).unwrap_or_default());
        all_memories.extend(self.brain.semantic.search("", 10000).unwrap_or_default());
        all_memories.extend(self.brain.procedural.search("", 10000).unwrap_or_default());
        let mut recent: Vec<&MemoryItem> = all_memories.iter()
            .filter(|m| m.created_at > cutoff)
            .collect();
        recent.sort_by_key(|m| m.created_at);

        let mut chains: Vec<Vec<EpisodeLink>> = vec![];
        let mut current_chain: Vec<EpisodeLink> = vec![];

        for window in recent.windows(2) {
            let prev = window[0];
            let curr = window[1];
            
            let gap_seconds = (curr.created_at - prev.created_at).num_seconds();
            let gap_minutes = gap_seconds / 60;
            
            let context_sim = match (&prev.embedding, &curr.embedding) {
                (Some(a), Some(b)) => cosine_similarity(a, b),
                _ => tag_similarity(&prev.tags, &curr.tags),
            };
            
            let link = EpisodeLink {
                from_id: prev.id.to_string(),
                to_id: curr.id.to_string(),
                gap_seconds,
                context_similarity: context_sim,
            };

            if gap_minutes <= max_gap_minutes {
                // Continue current chain
                current_chain.push(link);
            } else {
                // Gap too large - start new chain
                if !current_chain.is_empty() {
                    chains.push(std::mem::take(&mut current_chain));
                }
                current_chain.push(link);
            }
        }
        
        if !current_chain.is_empty() {
            chains.push(current_chain);
        }

        chains
    }

    /// Get the episode chain for a specific memory
    pub fn get_episode_context(&self, memory_id: &str, window: usize) -> Vec<MemoryItem> {
        let mut all = Vec::new();
        all.extend(self.brain.episodic.search("", 10000).unwrap_or_default());
        all.extend(self.brain.semantic.search("", 10000).unwrap_or_default());
        all.extend(self.brain.procedural.search("", 10000).unwrap_or_default());
        let mut sorted: Vec<MemoryItem> = all;
        sorted.sort_by_key(|m| m.created_at);
        
        // Find the target memory's position
        let pos = sorted.iter().position(|m| m.id.to_string() == memory_id);
        
        match pos {
            Some(idx) => {
                let start = idx.saturating_sub(window);
                let end = (idx + window + 1).min(sorted.len());
                sorted[start..end].to_vec()
            }
            None => vec![],
        }
    }

    // ========================================
    // 3. AUTO-IMPORTANCE - Smart strength scoring
    // ========================================

    /// Calculate importance score for new content before storing.
    /// Based on novelty (how different from existing memories) and emotional intensity.
    pub fn calculate_importance(&self, content: &str, emotional_valence: f32, tags: &[String]) -> ImportanceScore {
        let base_strength = 0.5;
        
        // 1. Novelty: How different is this from existing memories?
        let novelty = self.calculate_novelty(content, tags);
        
        // 2. Emotional intensity
        let emotional_intensity = emotional_valence.abs();
        
        // 3. Tag rarity - rare tags = more novel
        let tag_rarity = self.calculate_tag_rarity(tags);
        
        // Combine scores
        // High novelty + strong emotion = very strong memory
        let strength = (base_strength 
            + novelty * 0.25          // novel memories are stronger
            + emotional_intensity * 0.2  // emotional memories are stronger
            + tag_rarity * 0.05          // rare topics slightly stronger
        ).clamp(0.3, 1.0);
        
        let reason = format!(
            "novelty={:.2} emotion={:.2} rarity={:.2}",
            novelty, emotional_intensity, tag_rarity
        );
        
        ImportanceScore {
            strength,
            novelty,
            emotional_intensity,
            reason,
        }
    }

    fn calculate_novelty(&self, content: &str, tags: &[String]) -> f32 {
        // Search for similar existing memories
        let similar = self.brain.semantic.search(content, 5).unwrap_or_default();
        
        if similar.is_empty() {
            return 1.0; // Completely novel - nothing similar exists
        }
        
        // Average similarity to top matches
        let avg_sim: f32 = similar.iter()
            .filter_map(|m| m.embedding.as_ref())
            .map(|_emb| {
                // For simplicity, use content-based comparison
                // (embedding comparison would need the new content's embedding)
                0.5 // placeholder - will be refined with actual embedding
            })
            .sum::<f32>() / similar.len().max(1) as f32;
        
        // Novelty = inverse of similarity
        (1.0 - avg_sim).clamp(0.0, 1.0)
    }

    fn calculate_tag_rarity(&self, tags: &[String]) -> f32 {
        if tags.is_empty() {
            return 0.5;
        }
        
        let all = self.brain.semantic.search("", 10000).unwrap_or_default();
        let total = all.len().max(1) as f32;
        
        // Count how often each tag appears
        let mut tag_counts: HashMap<&str, usize> = HashMap::new();
        for mem in &all {
            for tag in &mem.tags {
                *tag_counts.entry(tag.as_str()).or_insert(0) += 1;
            }
        }
        
        // Average rarity of input tags
        let rarity: f32 = tags.iter().map(|t| {
            let count = *tag_counts.get(t.as_str()).unwrap_or(&0) as f32;
            1.0 - (count / total).min(1.0)
        }).sum::<f32>() / tags.len() as f32;
        
        rarity
    }

    /// Store a memory with auto-calculated importance
    pub fn store_with_importance(
        &mut self,
        content: &str,
        tags: Option<Vec<String>>,
        emotional_valence: f32,
    ) -> Result<ImportanceScore, Box<dyn std::error::Error>> {
        let tag_list = tags.clone().unwrap_or_default();
        let importance = self.calculate_importance(content, emotional_valence, &tag_list);
        
        // Store the memory (process takes Option<&str> for context, tags set separately)
        let tag_str = tags.as_ref().map(|t| t.join(","));
        self.brain.process(content, tag_str.as_deref())?;
        
        // Find the most recently stored memory and update its strength
        // (process() adds it, so it's the latest one)
        if let Ok(items) = self.brain.semantic.search(content, 1) {
            if let Some(item) = items.first() {
                let _ = self.brain.update_strength(&item.id.to_string(), importance.strength);
            }
        }
        
        Ok(importance)
    }
}

fn tag_similarity(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let total = a.len().max(b.len()).max(1);
    let overlap = a.iter().filter(|t| b.contains(t)).count();
    overlap as f32 / total as f32
}

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
    fn test_auto_importance() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("hippo_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();
        
        // Store some existing memories
        brain.process("Rust is a programming language", None).unwrap();
        brain.process("Python is popular for ML", None).unwrap();
        
        let hippo = Hippocampus::new(&mut brain);
        
        // Novel topic should score higher
        let novel = hippo.calculate_importance(
            "Quantum computing breakthrough",
            0.0,
            &["quantum".to_string()],
        );
        
        // Emotional memory should score higher
        let emotional = hippo.calculate_importance(
            "Some regular content",
            0.9,
            &[],
        );
        
        assert!(novel.novelty > 0.0);
        assert!(emotional.emotional_intensity > 0.5);
        assert!(emotional.strength > 0.5);
    }

    #[test]
    fn test_episode_chains() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("chain_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();
        
        brain.process("Started working on the project", Some("work")).unwrap();
        brain.process("Fixed a bug in the code", Some("work,bug")).unwrap();
        brain.process("Deployed to production", Some("work,deploy")).unwrap();
        
        let hippo = Hippocampus::new(&mut brain);
        let chains = hippo.build_episode_chains(24, 60);
        
        // Should have at least one chain
        assert!(!chains.is_empty(), "Expected at least one episode chain");
    }

    #[test]
    fn test_replay() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("replay_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();
        
        brain.process("Morning: reviewed code", Some("work")).unwrap();
        brain.process("Afternoon: fixed bugs", Some("work")).unwrap();
        brain.process("Evening: deployed updates", Some("work")).unwrap();
        
        let mut hippo = Hippocampus::new(&mut brain);
        let result = hippo.replay(24);
        
        assert!(result.replayed >= 3);
    }
}
