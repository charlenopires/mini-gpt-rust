# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Mini-GPT-Rust is an educational implementation of a Large Language Model (LLM) in Rust, focusing on teaching the fundamental components of modern transformer-based models. The project includes comprehensive demonstrations, interactive web interfaces, and optimized kernel implementations.

## Key Commands

### Development Commands

```bash
# Build the project
cargo build

# Build with optimizations for production
cargo build --release

# Run all tests
cargo test

# Run specific module tests
cargo test attention
cargo test tokenizer
cargo test transformer
cargo test model

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

### Core Application Commands

```bash
# Training a model
cargo run -- train --data data/corpus_pt_br.txt --epochs 100

# Generate text
cargo run -- generate --prompt "Era uma vez" --max-tokens 100

# Interactive chat
cargo run -- chat

# Load and use a saved model
cargo run -- load --checkpoint models/mini_gpt.safetensors --prompt "Hello world"

# List available model checkpoints
cargo run -- list --dir models
```

### Demonstration System

```bash
# Run all educational demonstrations
cargo run -- demo

# Run specific module demonstrations
cargo run -- demo --module attention
cargo run -- demo --module tokenizer
cargo run -- demo --module transformer

# Interactive mode with educational explanations
cargo run -- demo --interactive --educational --show-tensors

# Run benchmarks
cargo run -- demo --benchmark
```

### Web Interface

```bash
# Start basic web server for interactive demonstrations
cargo run -- web --host 127.0.0.1 --port 8080 --dir interativos

# Start advanced web server with WebSocket integration
cargo run -- web --integration --port 3001

# Web interface with educational interactives accessible at:
# - Tokenization: /tokenization
# - Attention: /attention
# - Embeddings: /embeddings
# - Transformer: /transformer
# - Training: /training
# - Inference: /inference
# - Chunking: /chunking
```

### Performance and Benchmarking

```bash
# Run kernel fusion benchmarks
cargo run -- benchmark --batch-size 4 --seq-len 128 --d-model 512

# Test text chunking strategies
cargo run -- chunk --input data/sample_text.txt --strategy semantic

# Benchmark chunking performance
cargo run -- chunk-bench --input data/large_text.txt --strategies "fixed,semantic,adaptive"
```

### Examples

```bash
# Run educational examples
cargo run --example transformer_architecture
cargo run --example tokenization_process
cargo run --example embeddings_explained
cargo run --example training_system
cargo run --example gradient_computation
cargo run --example optimization_techniques
cargo run --example inference_engine
cargo run --example memory_management
```

## Architecture Overview

### Core Components

1. **Tokenizer (`src/tokenizer.rs`)**
   - Implements Byte Pair Encoding (BPE) algorithm
   - Converts text to numerical tokens and back
   - Supports Portuguese and multilingual text
   - Educational demonstrations of tokenization process

2. **Attention Mechanism (`src/attention.rs`)**
   - Self-attention and multi-head attention implementation
   - Core component that allows tokens to "communicate"
   - Mathematical implementation of attention formula: `Attention(Q,K,V) = softmax(QK^T / √d_k)V`

3. **Transformer Blocks (`src/transformer.rs`)**
   - Complete transformer architecture with layer normalization
   - Feed-forward networks with GELU activation
   - Residual connections for gradient flow
   - Pre-LN (Pre-Layer Normalization) for training stability

4. **Model (`src/model.rs`)**
   - Main Mini-GPT model combining all components
   - Embedding layers for tokens and positions
   - Language modeling head for next-token prediction
   - Checkpoint saving/loading with SafeTensors format

5. **Training (`src/training.rs`)**
   - Complete training loop with Adam optimizer
   - Loss calculation and backpropagation
   - Checkpoint management and metadata tracking
   - Educational logging of training progress

### Advanced Features

1. **Educational Logger (`src/educational_logger.rs`)**
   - Detailed explanations of model operations
   - Visualization of attention maps and tensor operations
   - Interactive learning aids with step-by-step breakdowns

2. **Kernel Optimizations (`src/kernels.rs`)**
   - Fused attention and feed-forward operations
   - Memory-efficient implementations
   - Performance benchmarking tools
   - Adaptive execution strategies

3. **Web Server (`src/web_server.rs`)**
   - Axum-based async web server
   - WebSocket support for real-time communication
   - Interactive educational demonstrations
   - REST API for parameter control

4. **Text Chunking (`src/chunking.rs`)**
   - Multiple chunking strategies (fixed, semantic, adaptive, overlapping)
   - Preserves sentence and paragraph boundaries
   - Quality metrics and performance analysis

### Key Design Patterns

1. **Educational Focus**: Every component includes extensive documentation explaining the "why" behind implementations
2. **Performance Optimization**: GPU acceleration with Metal/CUDA fallbacks
3. **Interactive Learning**: Web-based demonstrations with real-time visualization
4. **Modular Design**: Each component can be studied and modified independently

## Development Guidelines

### Code Organization

- `src/` - Core implementation modules
- `examples/` - Educational examples demonstrating specific concepts
- `interativos/` - Web-based interactive demonstrations
- `models/` - Saved model checkpoints
- `data/` - Training data and corpora

### Testing Strategy

- Unit tests for each module focusing on mathematical correctness
- Integration tests for full model pipelines
- Benchmark tests for performance regression detection
- Educational validation tests ensuring explanations remain accurate

### Performance Considerations

- The project prioritizes Metal GPU acceleration on Apple Silicon
- CPU fallbacks ensure compatibility across platforms
- Kernel fusion provides 3-5x speedup for critical operations
- Memory optimization reduces usage by 30-50%

### Educational Features

- Use `--educational` flag to enable detailed logging
- Use `--interactive` mode for step-by-step explanations
- Web interface provides visual representations of concepts
- All mathematical formulas are explained with intuitive analogies

## Common Tasks

### Adding New Features

1. Create module in `src/` with extensive educational documentation
2. Add corresponding example in `examples/`
3. Include unit tests with clear explanations
4. Add demonstration command in main CLI
5. Consider web interface integration

### Debugging Model Issues

1. Use educational logger: `cargo run -- generate --educational --show-tensors`
2. Run specific module demos: `cargo run -- demo --module <component>`
3. Check attention visualizations in web interface
4. Use interactive chat for step-by-step analysis

### Performance Optimization

1. Run benchmarks: `cargo run -- benchmark --benchmark-type all`
2. Profile with educational logs enabled
3. Test kernel fusion effectiveness
4. Monitor memory usage patterns

### Training New Models

1. Prepare corpus in `data/` directory
2. Configure model parameters in training command
3. Monitor progress with educational logging
4. Save checkpoints with descriptive metadata
5. Validate with generation tests

This project serves as both a functional LLM implementation and a comprehensive educational resource for understanding modern transformer architectures.
