# Mistral SRT AI Generator

A Rust-based tool for generating subtitles (SRT files) using Mistral AI models.

## Features

- Generate subtitles from audio/video files
- AI-powered transcription and timing
- Support for multiple output formats
- Configurable through environment variables

## Installation

### Prerequisites

- Rust 1.60+ (install via [rustup](https://rustup.rs/))
- Cargo package manager

### Build from source

```bash
git clone https://github.com/yourusername/mistral-srt-ai-gen.git
cd mistral-srt-ai-gen
cargo build --release
```

## Usage

### Basic usage

```bash
# Generate subtitles from an audio file
./target/release/mistral-srt-ai-gen --input-file test_data/sample.flac --output-file sample.srt -l fr --translate-to english --translate-to castellano --api-key 3a…bWJA,cx6…gm6
```

### Using time ranges

```bash
# Generate subtitles for a specific time range (e.g., from 30 seconds to 90 seconds)
./target/release/mistral-srt-ai-gen --input-file test_data/sample.flac --output-file sample.srt --start 30s --end 90s --api-key 3a…bWJA

# Using hh:mm:ss,ms format (e.g., from 1 minute to 1 minute 30 seconds)
./target/release/mistral-srt-ai-gen --input-file test_data/sample.flac --output-file sample.srt --start 00:01:00,000 --end 00:01:30,500 --api-key 3a…bWJA
```

When `--start` and/or `--end` are provided, the tool extracts only that source segment before upload.
Supported formats for source segment extraction are: `mp3`, `wav`, `m4a`, `flac`, `ogg`, `mp4`, `mov`, `mkv`, `webm`.

### Time format options

The `--start` and `--end` parameters support two formats:
- `[number]s` - e.g., `30s`, `120s`, `0.5s`
- `hh:mm:ss,ms` - e.g., `00:01:30,500`, `01:00:00,000`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Mistral AI for their powerful language models
- The Rust community for excellent development tools