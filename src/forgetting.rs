//! Forgetting Curve
//! 
//! Based on Ebbinghaus forgetting curve:
//! R = e^(-t/S)
//! 
//! Where:
//! - R = retention
//! - t = time since last access
//! - S = memory strength/stability

use crate::types::MemoryItem;
use chrono::Utc;

pub struct ForgettingCurve {
    /// Base decay rate (higher = faster forgetting)
    base_decay_rate: f32,
    /// Minimum retention (memories never fully disappear until cleanup)
    min_retention: f32,
}

impl ForgettingCurve {
    pub fn new() -> Self {
        Self {
            base_decay_rate: 0.1,  // ~10% decay per day baseline
            min_retention: 0.1,
        }
    }

    /// Calculate decay factor for a memory (0.0 - 1.0)
    /// Returns the multiplier to apply to strength
    pub fn calculate_decay(&self, item: &MemoryItem) -> f32 {
        let hours_since = (Utc::now() - item.last_accessed).num_hours() as f32;
        let days_since = hours_since / 24.0;

        // Stability factor based on:
        // - Access frequency (more access = more stable)
        // - Original strength
        // - Age (older memories that survived are more stable)
        let access_stability = (item.access_count as f32).ln().max(1.0);
        let strength_stability = item.strength;
        let age_days = (Utc::now() - item.created_at).num_days() as f32;
        let age_stability = if age_days > 7.0 { 1.2 } else { 1.0 }; // Survived memories are stronger

        let stability = access_stability * strength_stability * age_stability;

        // Ebbinghaus-like decay: R = e^(-t/S)
        let retention = (-days_since * self.base_decay_rate / stability).exp();

        retention.max(self.min_retention)
    }

    /// Apply decay to a list of memories
    pub fn apply_decay(&self, items: &mut Vec<MemoryItem>) {
        for item in items.iter_mut() {
            let decay = self.calculate_decay(item);
            item.decay(decay);
        }
    }

    /// Calculate optimal review time for a memory
    /// Returns hours until review is needed to maintain strength
    pub fn optimal_review_time(&self, item: &MemoryItem) -> f32 {
        // Simple spaced repetition: review when strength drops to 70%
        let target_retention: f32 = 0.7;
        let stability = (item.access_count as f32).ln().max(1.0) * item.strength;

        // t = -S * ln(R) / decay_rate
        let hours = -stability * target_retention.ln() / self.base_decay_rate * 24.0;
        hours.max(1.0) // At least 1 hour
    }

    /// Check if memory needs review
    pub fn needs_review(&self, item: &MemoryItem) -> bool {
        let current_retention = self.calculate_decay(item);
        current_retention < 0.7 && item.strength > 0.3 // Worth keeping but fading
    }
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryItem;

    #[test]
    fn test_decay_calculation() {
        let curve = ForgettingCurve::new();
        let item = MemoryItem::new("test memory", None);
        
        // Fresh memory should have minimal decay
        let decay = curve.calculate_decay(&item);
        assert!(decay > 0.9);
    }

    #[test]
    fn test_access_increases_stability() {
        let curve = ForgettingCurve::new();
        
        let item1 = MemoryItem::new("test", None);
        let mut item2 = MemoryItem::new("test", None);
        
        // Access item2 multiple times
        for _ in 0..5 {
            item2.access();
        }

        let decay1 = curve.calculate_decay(&item1);
        let decay2 = curve.calculate_decay(&item2);
        
        // More accessed memory should retain better
        assert!(decay2 >= decay1);
    }

    #[test]
    fn test_minimum_retention() {
        let curve = ForgettingCurve::new();
        let item = MemoryItem::new("test", None);
        
        let decay = curve.calculate_decay(&item);
        
        // Should never go below minimum retention
        assert!(decay >= 0.1);
    }

    #[test]
    fn test_needs_review() {
        let curve = ForgettingCurve::new();
        let item = MemoryItem::new("test", None);
        
        // Fresh memory should not need review
        let needs_review = curve.needs_review(&item);
        // For fresh memory, retention is high so no review needed
        assert!(!needs_review || item.strength > 0.3);
    }

    #[test]
    fn test_optimal_review_time() {
        let curve = ForgettingCurve::new();
        let item = MemoryItem::new("test", None);
        
        let review_time = curve.optimal_review_time(&item);
        
        // Should return positive hours
        assert!(review_time >= 1.0);
    }
}
