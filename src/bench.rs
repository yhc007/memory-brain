//! Benchmark Module
//!
//! Measure performance of various operations:
//! - SIMD vs scalar cosine similarity
//! - Index lookup speed
//! - Embedding generation
//! - Search performance

use std::time::{Duration, Instant};
use crate::simd_ops::cosine_similarity_simd;

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub ops_per_sec: f64,
    pub avg_time_us: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<30} {:>10} ops  {:>10.2} ops/sec  {:>10.2} µs/op",
            self.name,
            self.iterations,
            self.ops_per_sec,
            self.avg_time_us
        )
    }
}

/// Benchmark runner
pub struct Benchmarker {
    results: Vec<BenchResult>,
}

impl Benchmarker {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Run a benchmark
    pub fn bench<F>(&mut self, name: &str, iterations: usize, mut f: F)
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..10 {
            f();
        }

        // Actual benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed = start.elapsed();

        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;

        self.results.push(BenchResult {
            name: name.to_string(),
            iterations,
            total_time: elapsed,
            ops_per_sec,
            avg_time_us,
        });
    }

    /// Get results
    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }

    /// Print results
    pub fn print_results(&self) {
        println!("\n📊 Benchmark Results\n");
        println!("{:-<70}", "");
        for result in &self.results {
            println!("{}", result);
        }
        println!("{:-<70}", "");
    }
}

impl Default for Benchmarker {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar cosine similarity (for comparison)
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Run all benchmarks
pub fn run_benchmarks(verbose: bool) -> Vec<BenchResult> {
    let mut bench = Benchmarker::new();

    // Generate test vectors
    let dim_128: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let dim_128_b: Vec<f32> = (0..128).map(|i| ((i + 50) as f32 / 64.0) - 1.0).collect();
    
    let dim_256: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
    let dim_256_b: Vec<f32> = (0..256).map(|i| ((i + 100) as f32 / 128.0) - 1.0).collect();

    let dim_512: Vec<f32> = (0..512).map(|i| (i as f32 / 256.0) - 1.0).collect();
    let dim_512_b: Vec<f32> = (0..512).map(|i| ((i + 200) as f32 / 256.0) - 1.0).collect();

    if verbose {
        println!("🔧 Running benchmarks...\n");
    }

    // Use black_box to prevent optimization
    fn black_box<T>(x: T) -> T {
        unsafe { std::ptr::read_volatile(&x) }
    }

    // SIMD vs Scalar benchmarks
    bench.bench("scalar_cosine_128d", 100_000, || {
        black_box(cosine_similarity_scalar(
            black_box(&dim_128), 
            black_box(&dim_128_b)
        ));
    });

    bench.bench("simd_cosine_128d", 100_000, || {
        black_box(cosine_similarity_simd(
            black_box(&dim_128), 
            black_box(&dim_128_b)
        ));
    });

    bench.bench("scalar_cosine_256d", 100_000, || {
        black_box(cosine_similarity_scalar(
            black_box(&dim_256), 
            black_box(&dim_256_b)
        ));
    });

    bench.bench("simd_cosine_256d", 100_000, || {
        black_box(cosine_similarity_simd(
            black_box(&dim_256), 
            black_box(&dim_256_b)
        ));
    });

    bench.bench("scalar_cosine_512d", 50_000, || {
        black_box(cosine_similarity_scalar(
            black_box(&dim_512), 
            black_box(&dim_512_b)
        ));
    });

    bench.bench("simd_cosine_512d", 50_000, || {
        black_box(cosine_similarity_simd(
            black_box(&dim_512), 
            black_box(&dim_512_b)
        ));
    });

    // Batch operations
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..128).map(|j| ((i * j) as f32 / 1000.0) - 0.5).collect())
        .collect();

    bench.bench("batch_similarity_100x128d", 1_000, || {
        let mut sum = 0.0f32;
        for v in &vectors {
            sum += cosine_similarity_simd(&dim_128, v);
        }
        black_box(sum);
    });

    // Index operations (bloom filter style)
    use std::collections::HashSet;
    let mut set: HashSet<u64> = HashSet::new();
    for i in 0..10000 {
        set.insert(i);
    }

    bench.bench("hashset_lookup", 1_000_000, || {
        let _ = set.contains(&5000);
    });

    // Print results
    if verbose {
        bench.print_results();

        // Calculate speedups
        println!("\n⚡ SIMD Speedup:\n");
        
        let results = bench.results();
        for i in (0..results.len()).step_by(2) {
            if i + 1 < results.len() && results[i].name.starts_with("scalar") {
                let scalar = &results[i];
                let simd = &results[i + 1];
                let speedup = scalar.avg_time_us / simd.avg_time_us;
                println!("  {} → {}: {:.2}x faster", 
                    scalar.name.replace("scalar_", ""),
                    simd.name.replace("simd_", ""),
                    speedup
                );
            }
        }
    }

    bench.results().to_vec()
}

/// Quick SIMD test
pub fn test_simd_correctness() -> bool {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: Vec<f32> = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let scalar = cosine_similarity_scalar(&a, &b);
    let simd = cosine_similarity_simd(&a, &b);

    let diff = (scalar - simd).abs();
    diff < 0.0001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matches_scalar() {
        assert!(test_simd_correctness());
    }

    #[test]
    fn test_benchmarker() {
        let mut bench = Benchmarker::new();
        
        bench.bench("test_op", 1000, || {
            let _ = 1 + 1;
        });

        assert_eq!(bench.results().len(), 1);
        assert!(bench.results()[0].ops_per_sec > 0.0);
    }
}
