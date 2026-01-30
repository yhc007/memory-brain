//! SIMD-Optimized Vector Operations
//!
//! Fast cosine similarity and dot product using SIMD instructions.
//! Falls back to scalar on unsupported platforms.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized cosine similarity
/// 
/// Uses NEON on ARM64 (Apple Silicon), AVX on x86_64, scalar fallback otherwise.
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { cosine_similarity_neon(a, b) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { cosine_similarity_avx(a, b) }
        } else {
            cosine_similarity_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        cosine_similarity_scalar(a, b)
    }
}

/// SIMD-optimized dot product
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { dot_product_avx(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_product_scalar(a, b)
    }
}

/// SIMD-optimized L2 norm (magnitude)
#[inline]
pub fn l2_norm_simd(v: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { l2_norm_neon(v) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { l2_norm_avx(v) }
        } else {
            l2_norm_scalar(v)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        l2_norm_scalar(v)
    }
}

// ============ ARM64 NEON Implementation (Apple Silicon) ============

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_neon(a, b);
    let norm_a = l2_norm_neon(a);
    let norm_b = l2_norm_neon(b);
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    
    let mut sum = vdupq_n_f32(0.0);
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    // Process 4 floats at a time
    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        sum = vfmaq_f32(sum, va, vb); // fused multiply-add
    }
    
    // Horizontal sum
    let mut result = vaddvq_f32(sum);
    
    // Handle remainder
    for i in (chunks * 4)..len {
        result += *a_ptr.add(i) * *b_ptr.add(i);
    }
    
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_norm_neon(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 4;
    
    let mut sum = vdupq_n_f32(0.0);
    let v_ptr = v.as_ptr();
    
    for i in 0..chunks {
        let va = vld1q_f32(v_ptr.add(i * 4));
        sum = vfmaq_f32(sum, va, va);
    }
    
    let mut result = vaddvq_f32(sum);
    
    for i in (chunks * 4)..len {
        let x = *v_ptr.add(i);
        result += x * x;
    }
    
    result.sqrt()
}

// ============ x86_64 AVX Implementation ============

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn cosine_similarity_avx(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_avx(a, b);
    let norm_a = l2_norm_avx(a);
    let norm_b = l2_norm_avx(b);
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    
    let mut sum = _mm256_setzero_ps();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    // Process 8 floats at a time
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }
    
    // Horizontal sum (256-bit -> 128-bit -> scalar)
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(low, high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    
    // Handle remainder
    for i in (chunks * 8)..len {
        result += *a_ptr.add(i) * *b_ptr.add(i);
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn l2_norm_avx(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 8;
    
    let mut sum = _mm256_setzero_ps();
    let v_ptr = v.as_ptr();
    
    for i in 0..chunks {
        let va = _mm256_loadu_ps(v_ptr.add(i * 8));
        let sq = _mm256_mul_ps(va, va);
        sum = _mm256_add_ps(sum, sq);
    }
    
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(low, high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    
    for i in (chunks * 8)..len {
        let x = *v_ptr.add(i);
        result += x * x;
    }
    
    result.sqrt()
}

// ============ Scalar Fallback ============

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_scalar(a, b);
    let norm_a = l2_norm_scalar(a);
    let norm_b = l2_norm_scalar(b);
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2_norm_scalar(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============ Batch Operations ============

/// Compute cosine similarities between a query and multiple vectors
pub fn batch_cosine_similarity(query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| cosine_similarity_simd(query, v))
        .collect()
}

/// Find top-k most similar vectors
pub fn top_k_similar(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut similarities: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity_simd(query, v)))
        .collect();
    
    // Partial sort for top-k
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    similarities.truncate(k);
    similarities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        
        let sim = cosine_similarity_simd(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        
        let sim = cosine_similarity_simd(&a, &b);
        assert!(sim.abs() < 0.0001, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];
        
        let sim = cosine_similarity_simd(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.0001, "Opposite vectors should have similarity -1.0");
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let dot = dot_product_simd(&a, &b);
        assert!((dot - 10.0).abs() < 0.0001);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        
        let norm = l2_norm_simd(&v);
        assert!((norm - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_large_vector() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (255 - i) as f32).collect();
        
        let sim = cosine_similarity_simd(&a, &b);
        // Just check it doesn't crash and returns reasonable value
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_batch_similarity() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // identical
            vec![0.0, 1.0, 0.0, 0.0],  // orthogonal
            vec![0.5, 0.5, 0.0, 0.0],  // partial
        ];
        
        let sims = batch_cosine_similarity(&query, &vectors);
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 0.0001);
        assert!(sims[1].abs() < 0.0001);
    }

    #[test]
    fn test_top_k() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let vectors = vec![
            vec![0.0, 1.0, 0.0, 0.0],  // 0.0
            vec![1.0, 0.0, 0.0, 0.0],  // 1.0
            vec![0.5, 0.5, 0.0, 0.0],  // ~0.7
        ];
        
        let top = top_k_similar(&query, &vectors, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // index 1 should be first (highest similarity)
    }
}
