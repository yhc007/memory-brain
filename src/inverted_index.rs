//! Inverted Index for Fast Keyword Search
//!
//! Maps keywords to document IDs for O(1) lookup instead of O(n) LIKE search.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use uuid::Uuid;

/// Simple tokenizer - splits text into lowercase words
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2) // Skip single chars
        .map(String::from)
        .collect()
}

/// Inverted Index for fast keyword search
pub struct InvertedIndex {
    /// keyword -> set of document IDs
    index: RwLock<HashMap<String, HashSet<Uuid>>>,
    /// document ID -> set of keywords (for deletion)
    doc_keywords: RwLock<HashMap<Uuid, HashSet<String>>>,
}

impl InvertedIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            index: RwLock::new(HashMap::new()),
            doc_keywords: RwLock::new(HashMap::new()),
        }
    }

    /// Add a document to the index
    pub fn add(&self, id: Uuid, content: &str) {
        let tokens = tokenize(content);
        if tokens.is_empty() {
            return;
        }

        let mut index = self.index.write().unwrap();
        let mut doc_keywords = self.doc_keywords.write().unwrap();

        let mut keywords = HashSet::new();
        for token in tokens {
            index
                .entry(token.clone())
                .or_insert_with(HashSet::new)
                .insert(id);
            keywords.insert(token);
        }
        doc_keywords.insert(id, keywords);
    }

    /// Add multiple documents in batch
    pub fn add_batch(&self, items: &[(Uuid, String)]) {
        let mut index = self.index.write().unwrap();
        let mut doc_keywords = self.doc_keywords.write().unwrap();

        for (id, content) in items {
            let tokens = tokenize(content);
            let mut keywords = HashSet::new();
            
            for token in tokens {
                index
                    .entry(token.clone())
                    .or_insert_with(HashSet::new)
                    .insert(*id);
                keywords.insert(token);
            }
            doc_keywords.insert(*id, keywords);
        }
    }

    /// Search for documents containing ALL keywords (AND search)
    pub fn search_and(&self, query: &str) -> Vec<Uuid> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let index = self.index.read().unwrap();
        
        let mut result: Option<HashSet<Uuid>> = None;
        
        for token in tokens {
            if let Some(docs) = index.get(&token) {
                match result {
                    None => result = Some(docs.clone()),
                    Some(ref mut set) => {
                        *set = set.intersection(docs).cloned().collect();
                    }
                }
            } else {
                // Token not found, no results
                return Vec::new();
            }
        }

        result.map(|s| s.into_iter().collect()).unwrap_or_default()
    }

    /// Search for documents containing ANY keyword (OR search)
    pub fn search_or(&self, query: &str) -> Vec<Uuid> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let index = self.index.read().unwrap();
        let mut result = HashSet::new();

        for token in tokens {
            if let Some(docs) = index.get(&token) {
                result.extend(docs.iter().cloned());
            }
        }

        result.into_iter().collect()
    }

    /// Search with relevance scoring (count matching keywords)
    pub fn search_ranked(&self, query: &str, limit: usize) -> Vec<(Uuid, usize)> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let index = self.index.read().unwrap();
        let mut scores: HashMap<Uuid, usize> = HashMap::new();

        for token in &tokens {
            if let Some(docs) = index.get(token) {
                for doc_id in docs {
                    *scores.entry(*doc_id).or_insert(0) += 1;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(limit);
        results
    }

    /// Remove a document from the index
    pub fn remove(&self, id: &Uuid) -> bool {
        let mut index = self.index.write().unwrap();
        let mut doc_keywords = self.doc_keywords.write().unwrap();

        if let Some(keywords) = doc_keywords.remove(id) {
            for keyword in keywords {
                if let Some(docs) = index.get_mut(&keyword) {
                    docs.remove(id);
                    // Clean up empty entries
                    if docs.is_empty() {
                        index.remove(&keyword);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Update a document (remove old, add new)
    pub fn update(&self, id: Uuid, new_content: &str) {
        self.remove(&id);
        self.add(id, new_content);
    }

    /// Get statistics
    pub fn stats(&self) -> IndexStats {
        let index = self.index.read().unwrap();
        let doc_keywords = self.doc_keywords.read().unwrap();

        IndexStats {
            unique_keywords: index.len(),
            documents: doc_keywords.len(),
            avg_keywords_per_doc: if doc_keywords.is_empty() {
                0.0
            } else {
                doc_keywords.values().map(|k| k.len()).sum::<usize>() as f64
                    / doc_keywords.len() as f64
            },
        }
    }

    /// Check if a keyword exists
    pub fn contains_keyword(&self, keyword: &str) -> bool {
        let index = self.index.read().unwrap();
        index.contains_key(&keyword.to_lowercase())
    }

    /// Get all keywords (for debugging)
    pub fn keywords(&self) -> Vec<String> {
        let index = self.index.read().unwrap();
        index.keys().cloned().collect()
    }

    /// Clear the index
    pub fn clear(&self) {
        let mut index = self.index.write().unwrap();
        let mut doc_keywords = self.doc_keywords.write().unwrap();
        index.clear();
        doc_keywords.clear();
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Index statistics
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub unique_keywords: usize,
    pub documents: usize,
    pub avg_keywords_per_doc: f64,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Inverted Index: {} keywords, {} docs, {:.1} avg keywords/doc",
            self.unique_keywords, self.documents, self.avg_keywords_per_doc
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // Too short
    }

    #[test]
    fn test_add_and_search() {
        let index = InvertedIndex::new();
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        index.add(id1, "Rust programming language");
        index.add(id2, "Python programming tutorial");
        index.add(id3, "Rust and Python comparison");

        // AND search
        let results = index.search_and("rust programming");
        assert_eq!(results.len(), 1);
        assert!(results.contains(&id1));

        // OR search
        let results = index.search_or("rust python");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_ranked_search() {
        let index = InvertedIndex::new();
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.add(id1, "rust rust rust programming");
        index.add(id2, "rust programming");

        let results = index.search_ranked("rust programming", 10);
        // Both have "rust" and "programming", but id1 has more "rust"
        // Actually our tokenizer dedupes, so both will have same score
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove() {
        let index = InvertedIndex::new();
        
        let id1 = Uuid::new_v4();
        index.add(id1, "test document");

        assert!(!index.search_and("test").is_empty());
        
        index.remove(&id1);
        
        assert!(index.search_and("test").is_empty());
    }

    #[test]
    fn test_stats() {
        let index = InvertedIndex::new();
        
        index.add(Uuid::new_v4(), "hello world");
        index.add(Uuid::new_v4(), "hello rust");

        let stats = index.stats();
        assert_eq!(stats.documents, 2);
        assert_eq!(stats.unique_keywords, 3); // hello, world, rust
    }
}
