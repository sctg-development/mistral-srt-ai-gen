/*
 * MIT License
 * 
 * Copyright (c) 2026 Ronan Le Meillat - SCTG Development
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use std::sync::{Arc, Mutex};
use rand::Rng;

/// KeyRotator handles round-robin rotation of API keys
#[derive(Debug)]
pub struct KeyRotator {
    keys: Vec<String>,
    current_index: Arc<Mutex<usize>>,
}

impl KeyRotator {
    /// Create a new KeyRotator from a comma-separated string of API keys
    ///
    /// # Arguments
    /// * `api_keys` - Comma-separated string of API keys
    ///
    /// # Returns
    /// KeyRotator instance
    pub fn new(api_keys: &str) -> Self {
        let keys: Vec<String> = api_keys
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Start at a random index to distribute load
        let start_index = if keys.len() > 1 {
            let mut rng = rand::thread_rng();
            rng.gen_range(0..keys.len())
        } else {
            0
        };

        KeyRotator {
            keys,
            current_index: Arc::new(Mutex::new(start_index)),
        }
    }

    /// Get the next API key in round-robin fashion
    ///
    /// # Returns
    /// Option containing the next API key, or None if no keys are available
    pub fn get_next_key(&self) -> Option<String> {
        if self.keys.is_empty() {
            return None;
        }

        let mut index = self.current_index.lock().unwrap();

        // Get current key
        let current_key = self.keys[*index].clone();

        // Move to next key for round-robin
        *index = (*index + 1) % self.keys.len();

        Some(current_key)
    }

    /// Get the number of available API keys
    ///
    /// # Returns
    /// Number of API keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_key() {
        let rotator = KeyRotator::new("single_key");
        assert_eq!(rotator.key_count(), 1);
        assert_eq!(rotator.get_next_key(), Some("single_key".to_string()));
        assert_eq!(rotator.get_next_key(), Some("single_key".to_string()));
    }

    #[test]
    fn test_multiple_keys() {
        let rotator = KeyRotator::new("key1, key2, key3");
        assert_eq!(rotator.key_count(), 3);

        // Should rotate through keys
        let key1 = rotator.get_next_key();
        let key2 = rotator.get_next_key();
        let key3 = rotator.get_next_key();
        let key4 = rotator.get_next_key(); // Should wrap around

        assert_ne!(key1, key2);
        assert_ne!(key2, key3);
        assert_eq!(key1, key4);
    }

    #[test]
    fn test_empty_keys() {
        let rotator = KeyRotator::new("");
        assert_eq!(rotator.key_count(), 0);
        assert_eq!(rotator.get_next_key(), None);
    }

    #[test]
    fn test_whitespace_handling() {
        let rotator = KeyRotator::new("  key1  , key2 ,  key3  ");
        assert_eq!(rotator.key_count(), 3);

        let keys: Vec<String> = rotator.keys.iter().map(|k| k.clone()).collect();
        assert_eq!(keys[0], "key1");
        assert_eq!(keys[1], "key2");
        assert_eq!(keys[2], "key3");
    }

    #[test]
    fn test_random_start() {
        // This test just verifies it doesn't panic and returns valid keys
        let rotator = KeyRotator::new("key1,key2,key3,key4,key5");
        assert!(rotator.key_count() > 1);

        // Should be able to get keys
        for _ in 0..10 {
            let key = rotator.get_next_key();
            assert!(key.is_some());
            assert!(["key1", "key2", "key3", "key4", "key5"].contains(&key.unwrap().as_str()));
        }
    }

    #[test]
    fn test_round_robin_behavior() {
        let rotator = KeyRotator::new("key1,key2,key3");

        // Get keys in sequence and verify round-robin behavior
        let key1 = rotator.get_next_key().unwrap();
        let key2 = rotator.get_next_key().unwrap();
        let key3 = rotator.get_next_key().unwrap();
        let key4 = rotator.get_next_key().unwrap(); // Should wrap around to key1

        // All keys should be different in the first cycle
        assert_ne!(key1, key2);
        assert_ne!(key2, key3);
        assert_ne!(key3, key1);

        // Should wrap around correctly
        assert_eq!(key1, key4);
    }
}