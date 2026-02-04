//! Prediction Module - Pattern Analysis & Forecasting
//!
//! Analyzes memory patterns to:
//! - Predict what you might do next
//! - Alert about memories that might be forgotten
//! - Discover recurring patterns and habits

use crate::{Brain, MemoryItem};
use chrono::{Utc, Datelike, Timelike, Weekday};
use std::collections::HashMap;

/// Prediction result
#[derive(Debug, Clone)]
pub struct Prediction {
    pub content: String,
    pub confidence: f32,
    pub reason: String,
}

/// Forgetting alert
#[derive(Debug, Clone)]
pub struct ForgettingAlert {
    pub memory: MemoryItem,
    pub days_since_access: i64,
    pub strength: f32,
    pub urgency: AlertUrgency,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertUrgency {
    Low,      // Might forget soon
    Medium,   // Probably forgetting
    High,     // Almost forgotten!
}

impl std::fmt::Display for AlertUrgency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertUrgency::Low => write!(f, "🟡"),
            AlertUrgency::Medium => write!(f, "🟠"),
            AlertUrgency::High => write!(f, "🔴"),
        }
    }
}

/// Pattern type discovered
#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub frequency: usize,
    pub examples: Vec<String>,
}

/// Prediction engine
pub struct Predictor<'a> {
    brain: &'a Brain,
}

impl<'a> Predictor<'a> {
    pub fn new(brain: &'a Brain) -> Self {
        Self { brain }
    }

    /// Predict what might happen next based on patterns
    pub fn predict_next(&self, limit: usize) -> Vec<Prediction> {
        let mut predictions = Vec::new();
        
        // Get all memories for analysis
        let memories = self.get_all_memories();
        if memories.is_empty() {
            return predictions;
        }

        // Analyze time patterns
        let time_predictions = self.analyze_time_patterns(&memories);
        predictions.extend(time_predictions);

        // Analyze tag patterns
        let tag_predictions = self.analyze_tag_patterns(&memories);
        predictions.extend(tag_predictions);

        // Analyze content patterns
        let content_predictions = self.analyze_content_patterns(&memories);
        predictions.extend(content_predictions);

        // Sort by confidence
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        predictions.truncate(limit);

        predictions
    }

    /// Get memories that might be forgotten soon
    pub fn forgetting_alerts(&self, limit: usize) -> Vec<ForgettingAlert> {
        let mut alerts = Vec::new();
        let now = Utc::now();

        let memories = self.get_all_memories();

        for memory in memories {
            let days_since = (now - memory.last_accessed).num_days();
            
            // Calculate forgetting risk based on:
            // - Days since last access
            // - Current strength
            // - Access count
            let base_risk = days_since as f32 / 30.0; // Risk increases over 30 days
            let strength_factor = 1.0 - memory.strength;
            let access_factor = 1.0 / (memory.access_count as f32 + 1.0);
            
            let risk = (base_risk * 0.4 + strength_factor * 0.4 + access_factor * 0.2).min(1.0);

            if risk > 0.3 {
                let urgency = if risk > 0.7 {
                    AlertUrgency::High
                } else if risk > 0.5 {
                    AlertUrgency::Medium
                } else {
                    AlertUrgency::Low
                };

                alerts.push(ForgettingAlert {
                    memory: memory.clone(),
                    days_since_access: days_since,
                    strength: memory.strength,
                    urgency,
                });
            }
        }

        // Sort by urgency (high first)
        alerts.sort_by(|a, b| {
            let urgency_order = |u: &AlertUrgency| match u {
                AlertUrgency::High => 0,
                AlertUrgency::Medium => 1,
                AlertUrgency::Low => 2,
            };
            urgency_order(&a.urgency).cmp(&urgency_order(&b.urgency))
        });

        alerts.truncate(limit);
        alerts
    }

    /// Discover recurring patterns
    pub fn discover_patterns(&self) -> Vec<Pattern> {
        let mut patterns = Vec::new();
        let memories = self.get_all_memories();

        // Time-based patterns
        if let Some(pattern) = self.find_time_pattern(&memories) {
            patterns.push(pattern);
        }

        // Tag-based patterns
        patterns.extend(self.find_tag_patterns(&memories));

        // Word frequency patterns
        if let Some(pattern) = self.find_word_pattern(&memories) {
            patterns.push(pattern);
        }

        // Day of week patterns
        if let Some(pattern) = self.find_weekday_pattern(&memories) {
            patterns.push(pattern);
        }

        patterns
    }

    // ============ Internal Analysis Methods ============

    fn get_all_memories(&self) -> Vec<MemoryItem> {
        let mut memories = Vec::new();
        if let Ok(items) = self.brain.semantic.search("", 1000) {
            memories.extend(items);
        }
        memories
    }

    fn analyze_time_patterns(&self, memories: &[MemoryItem]) -> Vec<Prediction> {
        let mut predictions = Vec::new();
        let now = Utc::now();
        let current_hour = now.hour();

        // Count activities by hour
        let mut hour_counts: HashMap<u32, Vec<&MemoryItem>> = HashMap::new();
        for memory in memories {
            let hour = memory.created_at.hour();
            hour_counts.entry(hour).or_default().push(memory);
        }

        // Find what usually happens at this hour
        if let Some(items) = hour_counts.get(&current_hour) {
            if items.len() >= 2 {
                // Find common tags at this hour
                let mut tag_counts: HashMap<&str, usize> = HashMap::new();
                for item in items {
                    for tag in &item.tags {
                        *tag_counts.entry(tag.as_str()).or_insert(0) += 1;
                    }
                }

                if let Some((tag, count)) = tag_counts.iter().max_by_key(|(_, c)| *c) {
                    if *count >= 2 {
                        predictions.push(Prediction {
                            content: format!("{}시에는 보통 '{}' 관련 작업을 해", current_hour, tag),
                            confidence: (*count as f32 / items.len() as f32).min(0.9),
                            reason: format!("{}번 중 {}번 이 시간에 '{}'", items.len(), count, tag),
                        });
                    }
                }
            }
        }

        predictions
    }

    fn analyze_tag_patterns(&self, memories: &[MemoryItem]) -> Vec<Prediction> {
        let mut predictions = Vec::new();

        // Find tag sequences (what follows what)
        let mut tag_sequences: HashMap<String, HashMap<String, usize>> = HashMap::new();
        
        let sorted_memories: Vec<_> = {
            let mut m = memories.to_vec();
            m.sort_by_key(|m| m.created_at);
            m
        };

        for i in 0..sorted_memories.len().saturating_sub(1) {
            for prev_tag in &sorted_memories[i].tags {
                for next_tag in &sorted_memories[i + 1].tags {
                    if prev_tag != next_tag {
                        *tag_sequences
                            .entry(prev_tag.clone())
                            .or_default()
                            .entry(next_tag.clone())
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        // Find most recent tag
        if let Some(recent) = sorted_memories.last() {
            for tag in &recent.tags {
                if let Some(follows) = tag_sequences.get(tag) {
                    if let Some((next_tag, count)) = follows.iter().max_by_key(|(_, c)| *c) {
                        if *count >= 2 {
                            predictions.push(Prediction {
                                content: format!("'{}' 다음에는 보통 '{}'를 해", tag, next_tag),
                                confidence: (*count as f32 / 10.0).min(0.8),
                                reason: format!("{}번의 패턴 발견", count),
                            });
                        }
                    }
                }
            }
        }

        predictions
    }

    fn analyze_content_patterns(&self, memories: &[MemoryItem]) -> Vec<Prediction> {
        let mut predictions = Vec::new();

        // Find frequently mentioned topics
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for memory in memories {
            for word in memory.content.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 3 {
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Top topics
        let mut words: Vec<_> = word_counts.iter().collect();
        words.sort_by(|a, b| b.1.cmp(a.1));

        if let Some((word, count)) = words.first() {
            if **count >= 5 {
                predictions.push(Prediction {
                    content: format!("'{}'에 대해 계속 작업할 것 같아", word),
                    confidence: (**count as f32 / memories.len() as f32).min(0.7),
                    reason: format!("{}번 언급됨", count),
                });
            }
        }

        predictions
    }

    fn find_time_pattern(&self, memories: &[MemoryItem]) -> Option<Pattern> {
        let mut hour_counts: HashMap<u32, usize> = HashMap::new();
        for memory in memories {
            *hour_counts.entry(memory.created_at.hour()).or_insert(0) += 1;
        }

        let mut hours: Vec<_> = hour_counts.iter().collect();
        hours.sort_by(|a, b| b.1.cmp(a.1));

        if let Some((hour, count)) = hours.first() {
            if **count >= 5 {
                return Some(Pattern {
                    name: "활동 시간대".to_string(),
                    description: format!("{}시에 가장 활발 ({}개 기억)", hour, count),
                    frequency: **count,
                    examples: vec![],
                });
            }
        }

        None
    }

    fn find_tag_patterns(&self, memories: &[MemoryItem]) -> Vec<Pattern> {
        let mut patterns = Vec::new();
        let mut tag_counts: HashMap<String, usize> = HashMap::new();

        for memory in memories {
            for tag in &memory.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        let mut tags: Vec<_> = tag_counts.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1));

        for (tag, count) in tags.iter().take(3) {
            if **count >= 3 {
                patterns.push(Pattern {
                    name: format!("'{}' 주제", tag),
                    description: format!("{}번 등장", count),
                    frequency: **count,
                    examples: memories.iter()
                        .filter(|m| m.tags.contains(tag))
                        .take(3)
                        .map(|m| truncate(&m.content, 30))
                        .collect(),
                });
            }
        }

        patterns
    }

    fn find_word_pattern(&self, memories: &[MemoryItem]) -> Option<Pattern> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        let stop_words = ["the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
                          "in", "with", "to", "for", "of", "테스트", "데이터", "메모리"];

        for memory in memories {
            for word in memory.content.split_whitespace() {
                let word = word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                if word.len() > 2 && !stop_words.contains(&word.as_str()) {
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut words: Vec<_> = word_counts.iter().collect();
        words.sort_by(|a, b| b.1.cmp(a.1));

        if let Some((word, count)) = words.first() {
            if **count >= 5 {
                return Some(Pattern {
                    name: "핵심 키워드".to_string(),
                    description: format!("'{}' - {}번 언급", word, count),
                    frequency: **count,
                    examples: vec![],
                });
            }
        }

        None
    }

    fn find_weekday_pattern(&self, memories: &[MemoryItem]) -> Option<Pattern> {
        let mut day_counts: HashMap<Weekday, usize> = HashMap::new();
        
        for memory in memories {
            let day = memory.created_at.weekday();
            *day_counts.entry(day).or_insert(0) += 1;
        }

        let mut days: Vec<_> = day_counts.iter().collect();
        days.sort_by(|a, b| b.1.cmp(a.1));

        if let Some((day, count)) = days.first() {
            if **count >= 5 {
                let day_name = match day {
                    Weekday::Mon => "월요일",
                    Weekday::Tue => "화요일",
                    Weekday::Wed => "수요일",
                    Weekday::Thu => "목요일",
                    Weekday::Fri => "금요일",
                    Weekday::Sat => "토요일",
                    Weekday::Sun => "일요일",
                };
                return Some(Pattern {
                    name: "활동 요일".to_string(),
                    description: format!("{}에 가장 활발 ({}개)", day_name, count),
                    frequency: **count,
                    examples: vec![],
                });
            }
        }

        None
    }
}

fn truncate(s: &str, max: usize) -> String {
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
    fn test_predictor() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("predict_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();

        // Add some test memories
        for i in 0..10 {
            brain.process(&format!("Rust programming test {}", i), None).unwrap();
        }

        let predictor = Predictor::new(&brain);
        
        let predictions = predictor.predict_next(5);
        // Should have some predictions based on patterns
        
        let alerts = predictor.forgetting_alerts(5);
        // New memories shouldn't trigger alerts
        
        let patterns = predictor.discover_patterns();
        // Should find some patterns
    }
}
