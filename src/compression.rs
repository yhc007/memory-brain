//! Memory Compression Module
//!
//! Compress old memories to save storage space.
//! Uses custom compression optimized for text and embeddings:
//! - Embedding quantization: f32 → i8 (75% size reduction)
//! - Simple RLE for repeated patterns
//! - Delta encoding for time series

use std::io::{Read, Write};

/// Quantized embedding (i8 instead of f32)
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// Quantized values (-128 to 127)
    pub values: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point offset
    pub zero_point: f32,
}

impl QuantizedEmbedding {
    /// Quantize f32 embedding to i8
    /// Reduces size by 75% (4 bytes → 1 byte per value)
    pub fn from_f32(embedding: &[f32]) -> Self {
        if embedding.is_empty() {
            return Self {
                values: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
            };
        }

        // Find min/max for scaling
        let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let range = max_val - min_val;
        // Map to 0-255 range (using full i8 range: -128 to 127)
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };

        let values: Vec<i8> = embedding
            .iter()
            .map(|&v| {
                // Normalize to 0-255, then shift to -128 to 127
                let normalized = ((v - min_val) / scale).round() as i32;
                (normalized - 128).clamp(-128, 127) as i8
            })
            .collect();

        Self {
            values,
            scale,
            zero_point: min_val,
        }
    }

    /// Dequantize back to f32
    pub fn to_f32(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&v| {
                // Shift from -128..127 back to 0..255, then scale
                let normalized = (v as i32 + 128) as f32;
                normalized * self.scale + self.zero_point
            })
            .collect()
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.values.len() + 8 // values + scale + zero_point
    }

    /// Original size if stored as f32
    pub fn original_size_bytes(&self) -> usize {
        self.values.len() * 4
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size_bytes() == 0 {
            return 1.0;
        }
        self.original_size_bytes() as f64 / self.size_bytes() as f64
    }
}

/// Simple Run-Length Encoding for text
pub struct RleEncoder;

impl RleEncoder {
    /// Encode bytes using RLE
    /// Good for text with repeated characters
    pub fn encode(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut count = 1u8;

            while i + (count as usize) < data.len() 
                && data[i + count as usize] == byte 
                && count < 255 
            {
                count += 1;
            }

            if count >= 4 {
                // Use RLE: marker (0xFF) + count + byte
                result.push(0xFF);
                result.push(count);
                result.push(byte);
            } else {
                // Store literal bytes
                for _ in 0..count {
                    if byte == 0xFF {
                        // Escape 0xFF
                        result.push(0xFF);
                        result.push(1);
                        result.push(0xFF);
                    } else {
                        result.push(byte);
                    }
                }
            }

            i += count as usize;
        }

        result
    }

    /// Decode RLE-encoded bytes
    pub fn decode(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFF && i + 2 < data.len() {
                let count = data[i + 1] as usize;
                let byte = data[i + 2];
                result.extend(std::iter::repeat(byte).take(count));
                i += 3;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        result
    }
}

/// Delta encoding for sequences of similar values
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encode using delta compression
    pub fn encode_i8(values: &[i8]) -> Vec<i8> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]); // First value stored as-is

        for i in 1..values.len() {
            // Store difference from previous value
            let delta = values[i].wrapping_sub(values[i - 1]);
            result.push(delta);
        }

        result
    }

    /// Decode delta-encoded values
    pub fn decode_i8(encoded: &[i8]) -> Vec<i8> {
        if encoded.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(encoded.len());
        result.push(encoded[0]);

        for i in 1..encoded.len() {
            let value = result[i - 1].wrapping_add(encoded[i]);
            result.push(value);
        }

        result
    }

    /// Encode f32 values using delta + quantization
    pub fn encode_f32(values: &[f32]) -> CompressedF32 {
        let quantized = QuantizedEmbedding::from_f32(values);
        let delta_encoded = Self::encode_i8(&quantized.values);
        
        CompressedF32 {
            data: delta_encoded,
            scale: quantized.scale,
            zero_point: quantized.zero_point,
        }
    }

    /// Decode back to f32
    pub fn decode_f32(compressed: &CompressedF32) -> Vec<f32> {
        let delta_decoded = Self::decode_i8(&compressed.data);
        let quantized = QuantizedEmbedding {
            values: delta_decoded,
            scale: compressed.scale,
            zero_point: compressed.zero_point,
        };
        quantized.to_f32()
    }
}

/// Compressed f32 vector
#[derive(Debug, Clone)]
pub struct CompressedF32 {
    pub data: Vec<i8>,
    pub scale: f32,
    pub zero_point: f32,
}

impl CompressedF32 {
    pub fn size_bytes(&self) -> usize {
        self.data.len() + 8
    }

    pub fn original_size_bytes(&self) -> usize {
        self.data.len() * 4
    }

    pub fn compression_ratio(&self) -> f64 {
        self.original_size_bytes() as f64 / self.size_bytes() as f64
    }
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub items_compressed: usize,
}

impl CompressionStats {
    pub fn ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            return 1.0;
        }
        self.original_bytes as f64 / self.compressed_bytes as f64
    }

    pub fn savings_percent(&self) -> f64 {
        if self.original_bytes == 0 {
            return 0.0;
        }
        (1.0 - (self.compressed_bytes as f64 / self.original_bytes as f64)) * 100.0
    }
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compression: {} items, {:.1} KB → {:.1} KB ({:.1}x, {:.1}% saved)",
            self.items_compressed,
            self.original_bytes as f64 / 1024.0,
            self.compressed_bytes as f64 / 1024.0,
            self.ratio(),
            self.savings_percent()
        )
    }
}

/// Compress a batch of embeddings
pub fn compress_embeddings(embeddings: &[Vec<f32>]) -> (Vec<CompressedF32>, CompressionStats) {
    let mut compressed = Vec::with_capacity(embeddings.len());
    let mut stats = CompressionStats::default();

    for embedding in embeddings {
        let original_size = embedding.len() * 4;
        let comp = DeltaEncoder::encode_f32(embedding);
        
        stats.original_bytes += original_size;
        stats.compressed_bytes += comp.size_bytes();
        stats.items_compressed += 1;
        
        compressed.push(comp);
    }

    (compressed, stats)
}

/// Decompress a batch of embeddings
pub fn decompress_embeddings(compressed: &[CompressedF32]) -> Vec<Vec<f32>> {
    compressed.iter().map(|c| DeltaEncoder::decode_f32(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() {
        let original: Vec<f32> = vec![0.1, 0.5, -0.3, 0.8, -0.9];
        let quantized = QuantizedEmbedding::from_f32(&original);
        let restored = quantized.to_f32();

        // Check approximate equality (quantization has some loss)
        let max_error: f32 = original
            .iter()
            .zip(restored.iter())
            .map(|(o, r)| (o - r).abs())
            .fold(0.0f32, f32::max);
        
        // Error should be within 1/255 of the range
        let range = 0.8 - (-0.9); // 1.7
        let expected_max_error = range / 255.0 * 2.0; // Allow 2x margin
        assert!(max_error < expected_max_error, "Too much quantization error: {} (expected < {})", max_error, expected_max_error);

        // Check compression ratio (small vectors have overhead from scale/zero_point)
        // 5 values: original 20 bytes, compressed 13 bytes = 1.5x
        assert!(quantized.compression_ratio() > 1.4, "Should have some compression: {}", quantized.compression_ratio());
    }

    #[test]
    fn test_quantization_large() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
        let quantized = QuantizedEmbedding::from_f32(&original);
        let restored = quantized.to_f32();

        // Check restoration accuracy
        let max_error: f32 = original
            .iter()
            .zip(restored.iter())
            .map(|(o, r)| (o - r).abs())
            .fold(0.0f32, f32::max);

        // Range is 2.0 (-1 to ~1), so max error should be about 2/255 ≈ 0.008
        assert!(max_error < 0.02, "Max error too high: {}", max_error);
    }

    #[test]
    fn test_rle_encode_decode() {
        let data = b"aaaaaabbbbccdddddddd";
        let encoded = RleEncoder::encode(data);
        let decoded = RleEncoder::decode(&encoded);

        assert_eq!(data.as_slice(), decoded.as_slice());
        assert!(encoded.len() < data.len(), "RLE should compress");
    }

    #[test]
    fn test_rle_no_repetition() {
        let data = b"abcdefgh";
        let encoded = RleEncoder::encode(data);
        let decoded = RleEncoder::decode(&encoded);

        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_delta_encoding() {
        let values: Vec<i8> = vec![10, 12, 11, 15, 14, 16];
        let encoded = DeltaEncoder::encode_i8(&values);
        let decoded = DeltaEncoder::decode_i8(&encoded);

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_compress_decompress_f32() {
        let original: Vec<f32> = vec![0.1, 0.15, 0.12, 0.18, 0.16, 0.2];
        let compressed = DeltaEncoder::encode_f32(&original);
        let restored = DeltaEncoder::decode_f32(&compressed);

        // Range is 0.1 (0.1 to 0.2), so max error ≈ 0.1/255 ≈ 0.0004
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!((o - r).abs() < 0.01, "Compression error too high: {} vs {}", o, r);
        }

        // Compression ratio: original 24 bytes (6 * 4), compressed ~14 bytes (6 + 8)
        // Ratio should be around 1.7
        println!("Compression ratio: {}", compressed.compression_ratio());
        assert!(compressed.compression_ratio() > 1.5, "Should have some compression: {}", compressed.compression_ratio());
    }

    #[test]
    fn test_batch_compression() {
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect())
            .collect();

        let (compressed, stats) = compress_embeddings(&embeddings);
        let restored = decompress_embeddings(&compressed);

        assert_eq!(embeddings.len(), restored.len());
        assert!(stats.ratio() > 3.0, "Batch compression ratio: {}", stats.ratio());
        
        println!("{}", stats);
    }
}
