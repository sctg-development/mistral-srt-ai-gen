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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Mistral AI for their powerful language models
- The Rust community for excellent development tools