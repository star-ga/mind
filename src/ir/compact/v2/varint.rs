// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! ULEB128 variable-length integer encoding for MIC-B v2.
//!
//! ULEB128 (Unsigned Little Endian Base 128) encodes unsigned integers
//! using 7 bits per byte, with the MSB as a continuation flag.
//!
//! Benefits:
//! - Small values (0-127) use just 1 byte
//! - No fixed-width overhead for typical IR indices

use std::io::{Read, Write};

/// Maximum bytes for u64 ULEB128 encoding (ceil(64/7) = 10).
const MAX_ULEB128_BYTES: usize = 10;

/// Write an unsigned integer as ULEB128.
///
/// # Examples
///
/// ```ignore
/// let mut buf = Vec::new();
/// uleb128_write(&mut buf, 127).unwrap();  // [0x7F]
/// uleb128_write(&mut buf, 128).unwrap();  // [0x80, 0x01]
/// ```
pub fn uleb128_write<W: Write>(w: &mut W, mut value: u64) -> std::io::Result<usize> {
    let mut bytes_written = 0;

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }

        w.write_all(&[byte])?;
        bytes_written += 1;

        if value == 0 {
            break;
        }
    }

    Ok(bytes_written)
}

/// Read an unsigned integer from ULEB128 encoding.
///
/// Returns error on:
/// - EOF before complete value
/// - Overflow (> 10 bytes or value > u64::MAX)
pub fn uleb128_read<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8; 1];

    for _ in 0..MAX_ULEB128_BYTES {
        r.read_exact(&mut buf)?;
        let byte = buf[0];

        // Add lower 7 bits to result
        let value = (byte & 0x7F) as u64;

        // Check for overflow
        if shift >= 64 || (shift == 63 && value > 1) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ULEB128 overflow",
            ));
        }

        result |= value << shift;
        shift += 7;

        // Check continuation bit
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "ULEB128 too long (> 10 bytes)",
    ))
}

/// Encode a signed integer using zigzag encoding.
///
/// Zigzag maps signed integers to unsigned integers:
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
///
/// This makes small negative values small unsigned values,
/// which encode efficiently in ULEB128.
#[inline]
pub fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Decode a zigzag-encoded unsigned integer to signed.
#[inline]
pub fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

/// Write a signed integer as zigzag-encoded ULEB128.
pub fn sleb128_write<W: Write>(w: &mut W, value: i64) -> std::io::Result<usize> {
    uleb128_write(w, zigzag_encode(value))
}

/// Read a signed integer from zigzag-encoded ULEB128.
pub fn sleb128_read<R: Read>(r: &mut R) -> std::io::Result<i64> {
    Ok(zigzag_decode(uleb128_read(r)?))
}

/// Calculate the encoded size of a value in ULEB128.
#[allow(dead_code)]
pub fn uleb128_size(mut value: u64) -> usize {
    let mut size = 0;
    loop {
        size += 1;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_uleb128_single_byte() {
        // Values 0-127 should be single byte
        for v in 0..=127u64 {
            let mut buf = Vec::new();
            uleb128_write(&mut buf, v).unwrap();
            assert_eq!(buf.len(), 1);
            assert_eq!(buf[0], v as u8);

            let mut cursor = Cursor::new(&buf);
            assert_eq!(uleb128_read(&mut cursor).unwrap(), v);
        }
    }

    #[test]
    fn test_uleb128_two_bytes() {
        // 128 should be [0x80, 0x01]
        let mut buf = Vec::new();
        uleb128_write(&mut buf, 128).unwrap();
        assert_eq!(buf, vec![0x80, 0x01]);

        let mut cursor = Cursor::new(&buf);
        assert_eq!(uleb128_read(&mut cursor).unwrap(), 128);
    }

    #[test]
    fn test_uleb128_large_values() {
        let test_values = [255u64, 256, 1000, 10000, 100000, u32::MAX as u64, u64::MAX];

        for v in test_values {
            let mut buf = Vec::new();
            uleb128_write(&mut buf, v).unwrap();

            let mut cursor = Cursor::new(&buf);
            assert_eq!(uleb128_read(&mut cursor).unwrap(), v, "Failed for {}", v);
        }
    }

    #[test]
    fn test_uleb128_size() {
        assert_eq!(uleb128_size(0), 1);
        assert_eq!(uleb128_size(127), 1);
        assert_eq!(uleb128_size(128), 2);
        assert_eq!(uleb128_size(16383), 2);
        assert_eq!(uleb128_size(16384), 3);
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);

        // Roundtrip
        for v in -1000..=1000i64 {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }

        // Edge cases
        assert_eq!(zigzag_decode(zigzag_encode(i64::MIN)), i64::MIN);
        assert_eq!(zigzag_decode(zigzag_encode(i64::MAX)), i64::MAX);
    }

    #[test]
    fn test_sleb128_roundtrip() {
        let test_values = [0i64, 1, -1, 127, -128, 1000, -1000, i64::MIN, i64::MAX];

        for v in test_values {
            let mut buf = Vec::new();
            sleb128_write(&mut buf, v).unwrap();

            let mut cursor = Cursor::new(&buf);
            assert_eq!(sleb128_read(&mut cursor).unwrap(), v, "Failed for {}", v);
        }
    }

    #[test]
    fn test_uleb128_eof() {
        let buf: Vec<u8> = vec![];
        let mut cursor = Cursor::new(&buf);
        assert!(uleb128_read(&mut cursor).is_err());
    }

    #[test]
    fn test_uleb128_incomplete() {
        // Continuation bit set but no more bytes
        let buf = vec![0x80];
        let mut cursor = Cursor::new(&buf);
        assert!(uleb128_read(&mut cursor).is_err());
    }

    #[test]
    fn test_uleb128_max_bytes() {
        // u64::MAX should use 10 bytes
        let mut buf = Vec::new();
        uleb128_write(&mut buf, u64::MAX).unwrap();
        assert_eq!(buf.len(), 10);
    }
}
