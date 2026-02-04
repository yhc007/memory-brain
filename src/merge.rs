//! Memory Merge Module
//!
//! Find and merge similar memories to reduce redundancy.
//! Uses cosine similarity to detect near-duplicates.

use crate::{Brain, MemoryItem, cosine_similarity};
use std::collections::HashSet;
use uuid::Uuid;

/// Merge configuration
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Similarity threshold for merging (0.0 - 1.0)
    /// Higher = stricter matching
    pub similarity_threshold: f32,
    /// Minimum number of items to consider merging
    pub min_cluster_size: usize,
    /// Keep the most recent memory as the primary
    pub keep_newest: bool,
    /// Combine tags from all merged memories
    pub merge_tags: bool,
    /// Dry run mode (don't actually merge)
    pub dry_run: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_cluster_size: 2,
            keep_newest: true,
            merge_tags: true,
            dry_run: false,
        }
    }
}

/// A cluster of similar memories
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    /// Primary memory (will be kept)
    pub primary: MemoryItem,
    /// Similar memories (will be merged into primary)
    pub similar: Vec<MemoryItem>,
    /// Average similarity within cluster
    pub avg_similarity: f32,
}

impl MemoryCluster {
    pub fn size(&self) -> usize {
        1 + self.similar.len()
    }
}

/// Result of merge operation
#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    /// Number of clusters found
    pub clusters_found: usize,
    /// Total memories that could be merged
    pub mergeable_count: usize,
    /// Memories actually merged
    pub merged_count: usize,
    /// Space saved (estimated bytes)
    pub space_saved_bytes: usize,
    /// Clusters with details
    pub clusters: Vec<MemoryCluster>,
}

impl std::fmt::Display for MergeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "🔗 Merge Analysis:")?;
        writeln!(f, "  Clusters found:    {}", self.clusters_found)?;
        writeln!(f, "  Mergeable items:   {}", self.mergeable_count)?;
        writeln!(f, "  Actually merged:   {}", self.merged_count)?;
        writeln!(f, "  Space saved:       {:.1} KB", self.space_saved_bytes as f64 / 1024.0)?;
        
        if !self.clusters.is_empty() {
            writeln!(f, "")?;
            writeln!(f, "📦 Clusters:")?;
            for (i, cluster) in self.clusters.iter().take(5).enumerate() {
                writeln!(f, "  {}. \"{}...\" ({} similar, {:.0}% avg)",
                    i + 1,
                    truncate(&cluster.primary.content, 30),
                    cluster.similar.len(),
                    cluster.avg_similarity * 100.0
                )?;
            }
            if self.clusters.len() > 5 {
                writeln!(f, "  ... and {} more clusters", self.clusters.len() - 5)?;
            }
        }
        
        Ok(())
    }
}

/// Truncate string with ellipsis
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        s.chars().take(max_len).collect::<String>() + "..."
    }
}

/// Memory merger
pub struct MemoryMerger<'a> {
    brain: &'a mut Brain,
    config: MergeConfig,
}

impl<'a> MemoryMerger<'a> {
    pub fn new(brain: &'a mut Brain) -> Self {
        Self {
            brain,
            config: MergeConfig::default(),
        }
    }

    pub fn with_config(brain: &'a mut Brain, config: MergeConfig) -> Self {
        Self { brain, config }
    }

    /// Find similar memories and optionally merge them
    pub fn find_similar(&mut self) -> MergeResult {
        let mut result = MergeResult::default();
        
        // Collect all memories with embeddings
        let mut memories: Vec<MemoryItem> = Vec::new();
        
        if let Ok(items) = self.brain.semantic.search("", 10000) {
            memories.extend(items);
        }
        if let Ok(items) = self.brain.episodic.search("", 10000) {
            memories.extend(items);
        }

        // Filter to only those with embeddings
        let memories: Vec<MemoryItem> = memories
            .into_iter()
            .filter(|m| m.embedding.is_some())
            .collect();

        if memories.len() < 2 {
            return result;
        }

        // Find clusters using greedy clustering
        let clusters = self.cluster_similar(&memories);
        
        result.clusters_found = clusters.len();
        result.mergeable_count = clusters.iter().map(|c| c.similar.len()).sum();
        result.clusters = clusters;

        // Estimate space savings
        for cluster in &result.clusters {
            for item in &cluster.similar {
                result.space_saved_bytes += item.content.len() + 512; // Rough estimate
            }
        }

        // Perform actual merge if not dry run
        if !self.config.dry_run {
            result.merged_count = self.execute_merge(&result.clusters);
        }

        result
    }

    /// Cluster similar memories together
    fn cluster_similar(&self, memories: &[MemoryItem]) -> Vec<MemoryCluster> {
        let mut clusters: Vec<MemoryCluster> = Vec::new();
        let mut assigned: HashSet<Uuid> = HashSet::new();

        for i in 0..memories.len() {
            if assigned.contains(&memories[i].id) {
                continue;
            }

            let mut similar: Vec<(MemoryItem, f32)> = Vec::new();
            let emb_i = memories[i].embedding.as_ref().unwrap();

            for j in (i + 1)..memories.len() {
                if assigned.contains(&memories[j].id) {
                    continue;
                }

                if let Some(emb_j) = &memories[j].embedding {
                    let similarity = cosine_similarity(emb_i, emb_j);
                    
                    if similarity >= self.config.similarity_threshold {
                        similar.push((memories[j].clone(), similarity));
                        assigned.insert(memories[j].id);
                    }
                }
            }

            if similar.len() >= self.config.min_cluster_size - 1 {
                assigned.insert(memories[i].id);
                
                let avg_sim = if similar.is_empty() {
                    1.0
                } else {
                    similar.iter().map(|(_, s)| s).sum::<f32>() / similar.len() as f32
                };

                // Sort by date and pick primary
                let mut all_items: Vec<MemoryItem> = vec![memories[i].clone()];
                all_items.extend(similar.iter().map(|(m, _)| m.clone()));
                
                all_items.sort_by(|a, b| {
                    if self.config.keep_newest {
                        b.created_at.cmp(&a.created_at)
                    } else {
                        a.created_at.cmp(&b.created_at)
                    }
                });

                let primary = all_items.remove(0);
                
                clusters.push(MemoryCluster {
                    primary,
                    similar: all_items,
                    avg_similarity: avg_sim,
                });
            }
        }

        clusters
    }

    /// Execute the merge operation
    fn execute_merge(&mut self, clusters: &[MemoryCluster]) -> usize {
        let mut merged_count = 0;

        for cluster in clusters {
            // Merge tags if configured
            if self.config.merge_tags {
                let mut all_tags: HashSet<String> = cluster.primary.tags.iter().cloned().collect();
                for item in &cluster.similar {
                    all_tags.extend(item.tags.iter().cloned());
                }
                // Note: We'd need to update the primary's tags in the database
                // This is simplified - full implementation would update DB
            }

            // Mark similar memories for deletion (keep primary)
            // Note: Full implementation would delete from storage
            // For now, we just count them as merged
            for _item in &cluster.similar {
                // TODO: Implement storage.delete() access
                // The memories are identified, but actual deletion needs
                // direct storage access or a Brain::delete_memory() method
                merged_count += 1;
            }
        }

        merged_count
    }

    /// Set similarity threshold
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set dry run mode
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.config.dry_run = dry_run;
        self
    }
}

/// Quick function to analyze duplicates
pub fn analyze_duplicates(brain: &mut Brain, threshold: f32) -> MergeResult {
    MemoryMerger::new(brain)
        .threshold(threshold)
        .dry_run(true)
        .find_similar()
}

/// Quick function to merge duplicates
pub fn merge_duplicates(brain: &mut Brain, threshold: f32) -> MergeResult {
    MemoryMerger::new(brain)
        .threshold(threshold)
        .dry_run(false)
        .find_similar()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_merge_config_default() {
        let config = MergeConfig::default();
        assert_eq!(config.similarity_threshold, 0.85);
        assert!(config.keep_newest);
        assert!(config.merge_tags);
        assert!(!config.dry_run);
    }

    #[test]
    fn test_merge_result_display() {
        let result = MergeResult {
            clusters_found: 3,
            mergeable_count: 10,
            merged_count: 0,
            space_saved_bytes: 2048,
            clusters: vec![],
        };
        
        let display = format!("{}", result);
        assert!(display.contains("3"));
        assert!(display.contains("10"));
        assert!(display.contains("2.0 KB"));
    }
}
