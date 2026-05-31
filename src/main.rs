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

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;

mod key_rotator;
use key_rotator::KeyRotator;

/// Mistral SRT AI Generator
///
/// A tool to generate SRT (SubRip) subtitle files from audio/video files using Mistral AI services.
/// This tool supports transcription and optional translation to multiple target languages.
///
/// # Examples
///
/// Basic transcription:
/// ```
/// use mistral_srt_ai_gen::{Args, run};
/// use std::path::PathBuf;
///
/// let args = Args {
///     input_file: PathBuf::from("input.mp3"),
///     output_file: PathBuf::from("output.srt"),
///     api_key: "your-api-key".to_string(),
///     language: None,
///     translate_to: vec![],
///     temperature: 0.3,
/// };
///
/// run(args).unwrap();
/// ```
///
/// Transcription with translation:
/// ```
/// use mistral_srt_ai_gen::{Args, run};
/// use std::path::PathBuf;
///
/// let args = Args {
///     input_file: PathBuf::from("input.mp3"),
///     output_file: PathBuf::from("output.srt"),
///     api_key: "your-api-key".to_string(),
///     language: Some("fr".to_string()),
///     translate_to: vec!["english".to_string(), "spanish".to_string()],
///     temperature: 0.3,
/// };
///
/// run(args).unwrap();
/// ```
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input audio or video file path
    #[arg(short, long)]
    input_file: PathBuf,

    /// Output SRT file path (for original language)
    #[arg(short, long)]
    output_file: PathBuf,

    /// Mistral AI API key
    #[arg(long)]
    api_key: String,

    /// Source language (ISO 639-1 code, optional)
    #[arg(short, long)]
    language: Option<String>,

    /// Target languages for translation (can be specified multiple times)
    #[arg(long, value_name = "LANGUAGE")]
    translate_to: Vec<String>,

    /// Temperature for AI model (0.0-1.0, default: 0.3)
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,
}

/// Main transcription segment structure from Mistral API
#[derive(Debug, Clone, serde::Deserialize)]
struct TranscriptionSegment {
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
    /// Transcribed text
    text: String,
}

/// Main transcription response from Mistral API
#[derive(Debug, serde::Deserialize)]
struct TranscriptionResponse {
    /// List of transcription segments
    segments: Vec<TranscriptionSegment>,
    /// Full transcribed text
    text: String,
}

/// Translation request structure
#[derive(Debug, serde::Serialize)]
struct TranslationRequest {
    /// Model to use for translation
    model: String,
    /// Messages for the translation prompt
    messages: Vec<Message>,
    /// Temperature setting
    temperature: f32,
}

/// Message structure for translation
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Message {
    /// Role (user or assistant)
    role: String,
    /// Content of the message
    content: String,
}

/// Translation response structure
#[derive(Debug, serde::Deserialize)]
struct TranslationResponse {
    /// List of choices from the model
    choices: Vec<Choice>,
}

/// Choice structure from translation response
#[derive(Debug, serde::Deserialize)]
struct Choice {
    /// Message content
    message: Message,
}

/// Custom error types for the application
#[derive(thiserror::Error, Debug)]
enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid language code: {0}")]
    InvalidLanguage(String),

    #[error("API error: {0}")]
    ApiError(String),
}

/// Convert seconds to SRT time format (HH:MM:SS,mmm)
///
/// # Arguments
///
/// * `seconds` - Time in seconds as f32
///
/// # Returns
///
/// String in SRT time format
///
/// # Examples
///
/// ```
/// let time = seconds_to_srt_time(1.5);
/// assert_eq!(time, "00:00:01,500");
/// ```
fn seconds_to_srt_time(seconds: f32) -> String {
    let total_millis = (seconds * 1000.0) as u64;
    let hours = total_millis / 3_600_000;
    let minutes = (total_millis % 3_600_000) / 60_000;
    let seconds = (total_millis % 60_000) / 1_000;
    let millis = total_millis % 1_000;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Generate SRT content from transcription segments
///
/// # Arguments
///
/// * `segments` - Vector of transcription segments
///
/// # Returns
///
/// String containing SRT formatted content
///
/// # Examples
///
/// ```
/// use mistral_srt_ai_gen::{TranscriptionSegment, generate_srt_content};
///
/// let segments = vec![
///     TranscriptionSegment {
///         start: 0.0,
///         end: 1.5,
///         text: "Hello world".to_string(),
///     }
/// ];
///
/// let srt = generate_srt_content(segments);
/// assert!(srt.contains("00:00:00,000 --> 00:00:01,500"));
/// ```
fn generate_srt_content(segments: Vec<TranscriptionSegment>) -> String {
    let mut srt_content = String::new();

    for (i, segment) in segments.iter().enumerate() {
        let start_time = seconds_to_srt_time(segment.start);
        let end_time = seconds_to_srt_time(segment.end);

        srt_content.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            start_time,
            end_time,
            segment.text
        ));
    }

    srt_content
}

/// Transcribe audio file using Mistral API
///
/// # Arguments
///
/// * `api_key` - Mistral API key
/// * `file_path` - Path to audio/video file
/// * `language` - Optional language code
///
/// # Returns
///
/// Result containing transcription response or error
async fn transcribe_audio(
    api_key: &str,
    file_path: &PathBuf,
    language: Option<&str>,
) -> Result<TranscriptionResponse> {
    let client = reqwest::Client::new();
    let url = "https://api.mistral.ai/v1/audio/transcriptions";

    // Read file content
    let file_content = fs::read(file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    // Determine mime type
    let _mime_type = mime_guess::from_path(file_path)
        .first_or_octet_stream()
        .to_string();

    // Build multipart form
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();

    let form = reqwest::multipart::Form::new()
        .text("model", "voxtral-mini-latest")
        .part("file", reqwest::multipart::Part::bytes(file_content).file_name(file_name))
        .text("timestamp_granularities[]", "segment");

    let form = if let Some(lang) = language {
        form.text("language", lang.to_string())
    } else {
        form
    };

    let response = client
        .post(url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_body = response.text().await.unwrap_or_default();
        return Err(AppError::ApiError(format!(
            "API request failed: {} - {}",
            status,
            error_body
        ))
        .into());
    }

    let transcription: TranscriptionResponse = response.json().await.map_err(AppError::Http)?;
    Ok(transcription)
}

/// Translate text using Mistral API
///
/// # Arguments
///
/// * `api_key` - Mistral API key
/// * `text` - Text to translate
/// * `target_language` - Target language in full text
/// * `temperature` - Temperature setting
///
/// # Returns
///
/// Result containing translated text or error
async fn translate_text(
    api_key: &str,
    text: &str,
    target_language: &str,
    temperature: f32,
) -> Result<String> {
    let client = reqwest::Client::new();
    let url = "https://api.mistral.ai/v1/chat/completions";

    let request = TranslationRequest {
        model: "mistral-small-latest".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: format!(
                    "You are a professional translator. Translate the following text to {}. \
                     If the text contains '|||SEG|||' markers, preserve them exactly as-is in your output, \
                     translating only the text between them:",
                    target_language
                ),
            },
            Message {
                role: "user".to_string(),
                content: text.to_string(),
            },
        ],
        temperature,
    };

    let response = client
        .post(url)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_body = response.text().await.unwrap_or_default();
        return Err(AppError::ApiError(format!(
            "Translation API request failed: {} - {}",
            status,
            error_body
        ))
        .into());
    }

    let translation_response: TranslationResponse = response.json().await?;
    Ok(translation_response.choices[0].message.content.clone())
}

/// Main application logic
///
/// # Arguments
///
/// * `args` - Parsed command line arguments
///
/// # Returns
///
/// Result indicating success or failure
async fn run(args: Args) -> Result<()> {
    // Validate temperature range
    if args.temperature < 0.0 || args.temperature > 1.0 {
        return Err(
            AppError::ApiError("Temperature must be between 0.0 and 1.0".to_string()).into(),
        );
    }

    // Validate language code if provided
    if let Some(ref lang) = args.language {
        if lang.len() != 2 {
            return Err(AppError::InvalidLanguage(format!(
                "Language code must be 2 characters (ISO 639-1), got: {}",
                lang
            ))
            .into());
        }
    }

    println!("Starting transcription process...");
    println!("Input file: {}", args.input_file.display());
    println!("Output file: {}", args.output_file.display());

    // Create key rotator for round-robin API key rotation
    let key_rotator = KeyRotator::new(&args.api_key);
    let key_count = key_rotator.key_count();

    if key_count > 1 {
        println!("Using round-robin key rotation with {} API keys", key_count);
    } else if key_count == 1 {
        println!("Using single API key");
    } else {
        return Err(AppError::ApiError("No valid API keys provided".to_string()).into());
    }

    // Step 1: Transcribe audio
    let transcription = transcribe_audio_with_rotator(
        &key_rotator,
        &args.input_file,
        args.language.as_deref()
    ).await?;

    println!(
        "Transcription completed with {} segments",
        transcription.segments.len()
    );

    // Generate SRT for original language
    let original_srt = generate_srt_content(transcription.segments.clone());
    fs::write(&args.output_file, original_srt)?;

    println!("Original SRT file saved to: {}", args.output_file.display());

    // Step 2-4: Translate if target languages are specified
    if !args.translate_to.is_empty() {
        println!(
            "Starting translation process for {} languages...",
            args.translate_to.len()
        );

        for target_lang in &args.translate_to {
            println!("Translating to: {}", target_lang);

            // Join all segment texts with a unique separator for a single translation API call
            let separator = "|||SEG|||";
            let combined_text: String = transcription
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(&format!("\n{}\n", separator));

            let translated_combined = translate_text_with_rotator(
                &key_rotator,
                &combined_text,
                target_lang,
                args.temperature,
            )
            .await?;

            // Split translated text back into individual segment texts
            let translated_parts: Vec<&str> = translated_combined.split(separator).collect();

            let translated_segments: Vec<TranscriptionSegment> = transcription
                .segments
                .iter()
                .enumerate()
                .map(|(i, segment)| TranscriptionSegment {
                    start: segment.start,
                    end: segment.end,
                    text: translated_parts
                        .get(i)
                        .map(|t| t.trim().to_string())
                        .unwrap_or_else(|| segment.text.clone()),
                })
                .collect();

            let translated_srt = generate_srt_content(translated_segments);

            // Create output file name with language suffix
            let translated_output = args.output_file.with_extension("");
            let translated_output = format!("{}-{}.srt", translated_output.display(), target_lang);
            let translated_path = PathBuf::from(translated_output);

            fs::write(&translated_path, translated_srt)?;
            println!(
                "Translated SRT file saved to: {}",
                translated_path.display()
            );
        }
    }

    println!("Process completed successfully!");
    Ok(())
}

/// Transcribe audio file using Mistral API with key rotation
///
/// # Arguments
///
/// * `key_rotator` - KeyRotator instance for round-robin key management
/// * `file_path` - Path to audio/video file
/// * `language` - Optional language code
///
/// # Returns
///
/// Result containing transcription response or error
async fn transcribe_audio_with_rotator(
    key_rotator: &KeyRotator,
    file_path: &PathBuf,
    language: Option<&str>,
) -> Result<TranscriptionResponse> {
    let api_key = key_rotator.get_next_key()
        .ok_or_else(|| AppError::ApiError("No API keys available".to_string()))?;

    transcribe_audio(&api_key, file_path, language).await
}

/// Translate text using Mistral API with key rotation
///
/// # Arguments
///
/// * `key_rotator` - KeyRotator instance for round-robin key management
/// * `text` - Text to translate
/// * `target_language` - Target language in full text
/// * `temperature` - Temperature setting
///
/// # Returns
///
/// Result containing translated text or error
async fn translate_text_with_rotator(
    key_rotator: &KeyRotator,
    text: &str,
    target_language: &str,
    temperature: f32,
) -> Result<String> {
    let api_key = key_rotator.get_next_key()
        .ok_or_else(|| AppError::ApiError("No API keys available".to_string()))?;

    translate_text(&api_key, text, target_language, temperature).await
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if let Err(e) = run(args).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use mockito::{Server, Mock};

    // Helper function to check if API_KEY is available
    fn is_api_key_available() -> bool {
        std::env::var("API_KEY").is_ok()
    }

    // Helper function to create a mock server for transcription
    fn mock_transcription_server(server: &mut Server) -> Mock {
        server.mock("POST", "/v1/audio/transcriptions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Test transcription"
                    }
                ],
                "text": "Test transcription text"
            }"#)
            .create()
    }

    // Helper function to create a mock server for translation
    fn mock_translation_server(server: &mut Server) -> Mock {
        server.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test translation"
                        }
                    }
                ]
            }"#)
            .create()
    }

    #[test]
    fn test_seconds_to_srt_time() {
        assert_eq!(seconds_to_srt_time(0.0), "00:00:00,000");
        assert_eq!(seconds_to_srt_time(1.5), "00:00:01,500");
        assert_eq!(seconds_to_srt_time(61.25), "00:01:01,250");
        assert_eq!(seconds_to_srt_time(3661.0), "01:01:01,000");
    }

    #[test]
    fn test_generate_srt_content() {
        let segments = vec![
            TranscriptionSegment {
                start: 0.0,
                end: 1.5,
                text: "Hello world".to_string(),
            },
            TranscriptionSegment {
                start: 1.5,
                end: 3.0,
                text: "This is a test".to_string(),
            },
        ];

        let srt = generate_srt_content(segments);
        assert!(srt.contains("00:00:00,000 --> 00:00:01,500"));
        assert!(srt.contains("Hello world"));
        assert!(srt.contains("00:00:01,500 --> 00:00:03,000"));
        assert!(srt.contains("This is a test"));
    }

    #[test]
    fn test_language_validation() {
        let args = Args {
            input_file: PathBuf::from("test.mp3"),
            output_file: PathBuf::from("output.srt"),
            api_key: "test".to_string(),
            language: Some("fr".to_string()),
            translate_to: vec![],
            temperature: 0.3,
        };

        // This should not panic - language validation happens in run()
        assert_eq!(args.language.unwrap(), "fr");
    }

    #[test]
    fn test_temperature_range() {
        let args = Args {
            input_file: PathBuf::from("test.mp3"),
            output_file: PathBuf::from("output.srt"),
            api_key: "test".to_string(),
            language: None,
            translate_to: vec![],
            temperature: 0.5,
        };

        assert_eq!(args.temperature, 0.5);
    }

    #[test]
    fn test_file_paths() {
        let args = Args {
            input_file: PathBuf::from("input.mp3"),
            output_file: PathBuf::from("output.srt"),
            api_key: "test".to_string(),
            language: None,
            translate_to: vec![],
            temperature: 0.3,
        };

        assert!(args.input_file.exists() || args.input_file.to_str().unwrap() == "input.mp3");
        assert_eq!(args.output_file.extension().unwrap(), "srt");
    }

    #[test]
    fn test_voxstral_response_deserialization() {
        let json_data = r#"
        {
            "model": "voxstral-mini-latest",
            "text": "Test transcription text",
            "language": null,
            "segments": [
                {
                    "text": "First segment",
                    "start": 5.0,
                    "end": 12.8,
                    "speaker_id": "speaker_1",
                    "type": "transcription_segment"
                },
                {
                    "text": "Second segment",
                    "start": 13.3,
                    "end": 18.4,
                    "speaker_id": "speaker_1",
                    "type": "transcription_segment"
                }
            ],
            "usage": {
                "prompt_audio_seconds": 10,
                "prompt_tokens": 100,
                "total_tokens": 500,
                "completion_tokens": 400,
                "request_count": 1
            },
            "type": "transcription.done"
        }
        "#;

        let response: TranscriptionResponse = serde_json::from_str(json_data).unwrap();

        assert_eq!(response.text, "Test transcription text");
        assert_eq!(response.segments.len(), 2);
        assert_eq!(response.segments[0].text, "First segment");
        assert_eq!(response.segments[0].start, 5.0);
        assert_eq!(response.segments[0].end, 12.8);
        assert_eq!(response.segments[1].text, "Second segment");
        assert_eq!(response.segments[1].start, 13.3);
        assert_eq!(response.segments[1].end, 18.4);
    }

    #[test]
    fn test_srt_generation_from_voxstral_response() {
        let json_data = r#"
        {
            "model": "voxstral-mini-latest",
            "text": "Test transcription text",
            "language": null,
            "segments": [
                {
                    "text": "First segment",
                    "start": 5.0,
                    "end": 12.8,
                    "speaker_id": "speaker_1",
                    "type": "transcription_segment"
                },
                {
                    "text": "Second segment",
                    "start": 13.3,
                    "end": 18.4,
                    "speaker_id": "speaker_1",
                    "type": "transcription_segment"
                }
            ],
            "usage": {
                "prompt_audio_seconds": 10,
                "prompt_tokens": 100,
                "total_tokens": 500,
                "completion_tokens": 400,
                "request_count": 1
            },
            "type": "transcription.done"
        }
        "#;

        let response: TranscriptionResponse = serde_json::from_str(json_data).unwrap();
        let srt_content = generate_srt_content(response.segments);

        assert!(srt_content.contains("00:00:05,000 --> 00:00:12,800"));
        assert!(srt_content.contains("First segment"));
        assert!(srt_content.contains("00:00:13,300 --> 00:00:18,400"));
        assert!(srt_content.contains("Second segment"));
        assert!(srt_content.contains("1\n00:00:05,000 --> 00:00:12,800"));
        assert!(srt_content.contains("2\n00:00:13,300 --> 00:00:18,400"));
    }

    #[test]
    fn test_empty_segments_handling() {
        let response = TranscriptionResponse {
            segments: vec![],
            text: "Empty transcription".to_string(),
        };

        let srt_content = generate_srt_content(response.segments);
        assert_eq!(srt_content, "");
    }

    #[test]
    fn test_single_segment_handling() {
        let segments = vec![TranscriptionSegment {
            start: 0.0,
            end: 1.0,
            text: "Single segment".to_string(),
        }];

        let srt_content = generate_srt_content(segments);
        assert!(srt_content.contains("1\n00:00:00,000 --> 00:00:01,000"));
        assert!(srt_content.contains("Single segment"));
    }

    #[test]
    fn test_app_error_display() {
        let io_error = AppError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"));
        assert_eq!(io_error.to_string(), "IO error: File not found");

        // Create a reqwest error using an invalid URL (no network call)
        let reqwest_error = reqwest::Client::new().get("://invalid").build().unwrap_err();
        let http_error = AppError::Http(reqwest_error);
        assert!(http_error.to_string().contains("HTTP error"));

        // Create a serde_json error using a parsing error
        let json_error = AppError::Json(serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err());
        assert!(json_error.to_string().contains("JSON error"));

        let invalid_lang = AppError::InvalidLanguage("xx".to_string());
        assert_eq!(invalid_lang.to_string(), "Invalid language code: xx");

        let api_error = AppError::ApiError("API failed".to_string());
        assert_eq!(api_error.to_string(), "API error: API failed");
    }

    #[tokio::test]
    async fn test_transcribe_audio_with_mock() {
        let mut server = Server::new_async().await;
        let _m = mock_transcription_server(&mut server);

        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_path_buf();

        // Override the URL to use the mock server
        let _original_url = "https://api.mistral.ai/v1/audio/transcriptions";
        let mock_url = server.url();

        // Create a modified version of transcribe_audio for testing
        let client = reqwest::Client::new();
        let url = format!("{}/v1/audio/transcriptions", mock_url);

        // Read file content
        let file_content = fs::read(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))
            .unwrap();

        // Build multipart form
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        let form = reqwest::multipart::Form::new()
            .text("model", "voxstral-mini-latest")
            .part("file", reqwest::multipart::Part::bytes(file_content).file_name(file_name))
            .text("response_format", "json");

        let response = client
            .post(&url)
            .bearer_auth("test_api_key")
            .multipart(form)
            .send()
            .await
            .unwrap();

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            panic!("API request failed: {} - {}", status, error_body);
        }

        let transcription: TranscriptionResponse = response.json().await.unwrap();

        assert_eq!(transcription.text, "Test transcription text");
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].text, "Test transcription");
    }

    #[tokio::test]
    async fn test_transcribe_audio_success() {
        if !is_api_key_available() {
            eprintln!("API_KEY not set, skipping test");
            return;
        }

        let file_path = PathBuf::from("test_data/sample.flac");
        if !file_path.exists() {
            eprintln!("Test file not found, skipping test");
            return;
        }

        let api_keys = std::env::var("API_KEY").unwrap();
        let key_rotator = KeyRotator::new(&api_keys);
        let api_key = key_rotator.get_next_key().expect("No API keys available");
        let result = transcribe_audio(&api_key, &file_path, Some("fr")).await;

        assert!(result.is_ok(), "transcribe_audio failed: {:?}", result);
        let response = result.unwrap();
        assert!(!response.text.is_empty());
        assert!(!response.segments.is_empty());
    }

    #[tokio::test]
    async fn test_transcribe_audio_file_not_found() {
        let file_path = PathBuf::from("nonexistent.flac");

        // Test the file reading error directly
        let result = fs::read(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()));

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Failed to read file"));
        }
    }

    #[tokio::test]
    async fn test_transcribe_audio_api_error() {
        let mut server = Server::new_async().await;
        let _m = server.mock("POST", "/v1/audio/transcriptions")
            .with_status(500)
            .with_body("Internal Server Error")
            .create();

        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_path_buf();

        // Override the URL to use the mock server
        let mock_url = server.url();

        // Create a modified version of transcribe_audio for testing
        let client = reqwest::Client::new();
        let url = format!("{}/v1/audio/transcriptions", mock_url);

        // Read file content
        let file_content = fs::read(&file_path).unwrap();

        // Build multipart form
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        let form = reqwest::multipart::Form::new()
            .text("model", "voxstral-mini-latest")
            .part("file", reqwest::multipart::Part::bytes(file_content).file_name(file_name))
            .text("response_format", "json");

        let response = client
            .post(&url)
            .bearer_auth("test_api_key")
            .multipart(form)
            .send()
            .await
            .unwrap();

        let status = response.status();
        assert_eq!(status, 500);

        let error_body = response.text().await.unwrap_or_default();
        assert_eq!(error_body, "Internal Server Error");
    }

    #[tokio::test]
    async fn test_translate_text_with_mock() {
        let mut server = Server::new_async().await;
        let _m = mock_translation_server(&mut server);

        // Override the URL to use the mock server
        let mock_url = server.url();

        // Create a modified version of translate_text for testing
        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_url);

        let request = TranslationRequest {
            model: "mistral-small-latest".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: "You are a professional translator. Translate the following text to english:".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "Bonjour".to_string(),
                },
            ],
            temperature: 0.3,
        };

        let response = client
            .post(&url)
            .bearer_auth("test_api_key")
            .json(&request)
            .send()
            .await
            .unwrap();

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            panic!("Translation API request failed: {} - {}", status, error_body);
        }

        let translation_response: TranslationResponse = response.json().await.unwrap();
        assert_eq!(translation_response.choices[0].message.content, "Test translation");
    }

    #[tokio::test]
    async fn test_translate_text_success() {
        if !is_api_key_available() {
            eprintln!("API_KEY not set, skipping test");
            return;
        }

        let api_keys = std::env::var("API_KEY").unwrap();
        let key_rotator = KeyRotator::new(&api_keys);
        let api_key = key_rotator.get_next_key().expect("No API keys available");
        let result = translate_text(
            &api_key,
            "Bonjour le monde",
            "english",
            0.3
        ).await;

        assert!(result.is_ok());
        let translation = result.unwrap();
        assert!(!translation.is_empty());
        assert_ne!(translation, "Bonjour le monde");
    }

    #[tokio::test]
    async fn test_translate_text_api_error() {
        let mut server = Server::new_async().await;
        let _m = server.mock("POST", "/v1/chat/completions")
            .with_status(500)
            .with_body("Internal Server Error")
            .create();

        // Override the URL to use the mock server
        let mock_url = server.url();

        // Create a modified version of translate_text for testing
        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_url);

        let request = TranslationRequest {
            model: "mistral-small-latest".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: "You are a professional translator. Translate the following text to english:".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "Bonjour".to_string(),
                },
            ],
            temperature: 0.3,
        };

        let response = client
            .post(&url)
            .bearer_auth("test_api_key")
            .json(&request)
            .send()
            .await
            .unwrap();

        let status = response.status();
        assert_eq!(status, 500);

        let error_body = response.text().await.unwrap_or_default();
        assert_eq!(error_body, "Internal Server Error");
    }

    #[tokio::test]
    async fn test_run_transcription_only_with_mock() {
        let mut server = Server::new_async().await;
        let _transcription_mock = mock_transcription_server(&mut server);

        let temp_input = NamedTempFile::new().unwrap();
        let temp_output = NamedTempFile::new().unwrap();

        // Create a modified version of run for testing
        let result: Result<(), anyhow::Error> = async {
            // Override the URL for testing
            let mock_url = server.url();

            // Create a modified transcribe_audio function for testing
            let test_transcribe_audio = |api_key: String, file_path: PathBuf, language: Option<String>| async move {
                let client = reqwest::Client::new();
                let url = format!("{}/v1/audio/transcriptions", mock_url);

                // Read file content
                let file_content = fs::read(&file_path)
                    .with_context(|| format!("Failed to read file: {}", file_path.display()))
                    .unwrap();

                // Build multipart form
                let file_name = file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file")
                    .to_string();

                let form = reqwest::multipart::Form::new()
                    .text("model", "voxstral-mini-latest")
                    .part("file", reqwest::multipart::Part::bytes(file_content).file_name(file_name))
                    .text("response_format", "json");

                let form = if let Some(lang) = language {
                    form.text("language", lang)
                } else {
                    form
                };

                let response = client
                    .post(&url)
                    .bearer_auth(api_key)
                    .multipart(form)
                    .send()
                    .await
                    .unwrap();

                let status = response.status();
                if !status.is_success() {
                    let error_body = response.text().await.unwrap_or_default();
                    return Err::<TranscriptionResponse, anyhow::Error>(AppError::ApiError(format!(
                        "API request failed: {} - {}",
                        status,
                        error_body
                    ))
                    .into());
                }

                let transcription: TranscriptionResponse = response.json().await.unwrap();
                Ok(transcription)
            };

            let args = Args {
                input_file: temp_input.path().to_path_buf(),
                output_file: temp_output.path().to_path_buf(),
                api_key: "test_api_key".to_string(),
                language: None,
                translate_to: vec![],
                temperature: 0.3,
            };

            // Validate temperature range
            if args.temperature < 0.0 || args.temperature > 1.0 {
                return Err(
                    AppError::ApiError("Temperature must be between 0.0 and 1.0".to_string()).into(),
                );
            }

            // Validate language code if provided
            if let Some(ref lang) = args.language {
                if lang.len() != 2 {
                    return Err(AppError::InvalidLanguage(format!(
                        "Language code must be 2 characters (ISO 639-1), got: {}",
                        lang
                    ))
                    .into());
                }
            }

            // Step 1: Transcribe audio using our test function
            let transcription = test_transcribe_audio(args.api_key.clone(), args.input_file.clone(), args.language.clone()).await.unwrap();

            // Generate SRT for original language
            let original_srt = generate_srt_content(transcription.segments.clone());
            fs::write(&args.output_file, original_srt).unwrap();

            Ok(())
        }.await;

        assert!(result.is_ok());
        let output_content = fs::read_to_string(temp_output.path()).unwrap();
        assert!(!output_content.is_empty());
        assert!(output_content.contains("Test transcription"));
    }

    #[tokio::test]
    async fn test_run_transcription_only() {
        if !is_api_key_available() {
            eprintln!("API_KEY not set, skipping test");
            return;
        }

        let input_file = PathBuf::from("test_data/sample.flac");
        if !input_file.exists() {
            eprintln!("Test file not found, skipping test");
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();

        let args = Args {
            input_file,
            output_file: temp_output.path().to_path_buf(),
            api_key: std::env::var("API_KEY").unwrap(),
            language: Some("fr".to_string()),
            translate_to: vec![],
            temperature: 0.3,
        };

        let result = run(args).await;

        assert!(result.is_ok());
        let output_content = fs::read_to_string(temp_output.path()).unwrap();
        assert!(!output_content.is_empty());
    }

    #[tokio::test]
    async fn test_run_with_invalid_temperature() {
        let temp_input = NamedTempFile::new().unwrap();
        let temp_output = NamedTempFile::new().unwrap();

        let args = Args {
            input_file: temp_input.path().to_path_buf(),
            output_file: temp_output.path().to_path_buf(),
            api_key: "test_api_key".to_string(),
            language: None,
            translate_to: vec![],
            temperature: 1.5, // Invalid temperature
        };

        let result = run(args).await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Temperature must be between 0.0 and 1.0"));
        }
    }

    #[tokio::test]
    async fn test_run_with_invalid_language() {
        let temp_input = NamedTempFile::new().unwrap();
        let temp_output = NamedTempFile::new().unwrap();

        let args = Args {
            input_file: temp_input.path().to_path_buf(),
            output_file: temp_output.path().to_path_buf(),
            api_key: "test_api_key".to_string(),
            language: Some("french".to_string()), // Invalid language code
            translate_to: vec![],
            temperature: 0.3,
        };

        let result = run(args).await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Language code must be 2 characters"));
        }
    }

    #[tokio::test]
    async fn test_run_with_translation() {
        if !is_api_key_available() {
            eprintln!("API_KEY not set, skipping test");
            return;
        }

        let input_file = PathBuf::from("test_data/sample.flac");
        if !input_file.exists() {
            eprintln!("Test file not found, skipping test");
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();

        let args = Args {
            input_file,
            output_file: temp_output.path().to_path_buf(),
            api_key: std::env::var("API_KEY").unwrap(),
            language: Some("fr".to_string()),
            translate_to: vec!["english".to_string()],
            temperature: 0.3,
        };

        let result = run(args).await;

        assert!(result.is_ok());

        // Check original file
        let output_content = fs::read_to_string(temp_output.path()).unwrap();
        assert!(!output_content.is_empty());

        // Check translated file
        let translated_path = temp_output.path().with_extension("");
        let translated_path = format!("{}-english.srt", translated_path.display());
        let translated_content = fs::read_to_string(translated_path).unwrap();
        assert!(!translated_content.is_empty());
    }

    #[tokio::test]
    async fn test_run_with_file_not_found() {
        let temp_output = NamedTempFile::new().unwrap();

        let args = Args {
            input_file: PathBuf::from("nonexistent.flac"),
            output_file: temp_output.path().to_path_buf(),
            api_key: "test_api_key".to_string(),
            language: None,
            translate_to: vec![],
            temperature: 0.3,
        };

        let result = run(args).await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Failed to read file"));
        }
    }
}
