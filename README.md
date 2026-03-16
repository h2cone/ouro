# ouro

Minimal Rust agent example built around the OpenAI Responses API.

## Features

- Single-file agent implementation in `examples/agent/main.rs`
- Minimal dependencies with `serde_json` and `ureq`
- Built-in local tools for shell execution plus file reads and writes
- Test coverage for tool execution, response handling, and iteration limits

## Getting Started

### Requirements

- Rust and Cargo
- An `OPENAI_API_KEY` environment variable

### Run the example agent

```powershell
$env:OPENAI_API_KEY="your-api-key"
cargo run --example agent -- "Summarize the current directory"
```

Optional environment variables:

- `OPENAI_BASE_URL` to point at a compatible API base URL
- `OPENAI_MODEL` to override the default model (`gpt-5.4`)

The crate root binary prints a reminder for the example entrypoint:

```powershell
cargo run
```

## Test

```powershell
cargo test
```

## License

MIT. See [LICENSE](LICENSE).
