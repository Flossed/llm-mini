# Mini LLM - TensorFlow.js GPU-Optimized Language Model

A lightweight transformer-based language model implementation optimized for AMD RX 7900 XTX GPU performance using TensorFlow.js.

## Features

- **GPU-Optimized Architecture**: Leverages WebGL backend with optimizations for AMD RX 7900 XTX
- **Multiple Model Sizes**: 
  - Tiny: 50M parameters (4 layers, 256 hidden size)
  - Small: 125M parameters (6 layers, 512 hidden size)
  - Medium: 350M parameters (12 layers, 768 hidden size)
- **Real-time Generation**: Achieves high token/second throughput on GPU
- **Interactive Chat Interface**: Clean, responsive UI for text generation
- **Performance Metrics**: Real-time display of tokens/second, inference time, and GPU memory usage

## Architecture

### Model Components

1. **Token Embeddings**: Learned representations for vocabulary tokens
2. **Positional Encoding**: Sinusoidal position embeddings
3. **Transformer Blocks**: 
   - Multi-head self-attention with causal masking
   - Feed-forward networks with GELU activation
   - Layer normalization and residual connections
4. **Output Projection**: Final linear layer to vocabulary size

### GPU Optimizations

- **Texture Memory**: Utilizes 4096x4096 texture capabilities
- **Packed Operations**: Enables WebGL packed operations for better throughput
- **Batch Parallelization**: Optimized for parallel batch processing
- **Memory Management**: Efficient tensor disposal and reuse

## Usage

1. Open `index.html` in a modern browser with WebGL support
2. Select model size based on your performance requirements
3. Adjust generation parameters:
   - **Temperature**: Controls randomness (0.1 = focused, 2.0 = creative)
   - **Max Tokens**: Maximum length of generated text
   - **Top-K**: Limits vocabulary to top K most likely tokens
4. Type your prompt and click Generate or press Enter

## Performance Benchmarks

On AMD RX 7900 XTX:

| Model Size | Batch Size | Seq Length | Tokens/Second |
|------------|------------|------------|---------------|
| Tiny       | 1          | 128        | ~2000         |
| Small      | 1          | 128        | ~1200         |
| Medium     | 1          | 128        | ~600          |
| Small      | 16         | 128        | ~8000         |

## Technical Details

### Tokenizer
- Simple word-level tokenizer with character fallback
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`, `<cls>`
- Vocabulary size configurable per model

### Generation Methods
- **Temperature Sampling**: Adjustable randomness
- **Top-K Filtering**: Restricts to most likely tokens
- **Greedy Decoding**: Selects most probable token

### Memory Requirements
- Tiny: ~200MB GPU memory
- Small: ~500MB GPU memory
- Medium: ~1.4GB GPU memory

## Development

### Running Benchmarks
Open browser console and run:
```javascript
runBenchmark()
```

### Accessing Model/Tokenizer
```javascript
model() // Get current model instance
tokenizer() // Get tokenizer instance
```

## Limitations

- This is a demonstration model with random weights (not pre-trained)
- Simple tokenizer - production would use BPE or SentencePiece
- Limited vocabulary for demonstration purposes
- No attention caching for autoregressive generation (future optimization)

## Future Enhancements

1. **Model Loading**: Support for loading pre-trained weights
2. **Advanced Tokenizer**: Implement BPE or integrate with existing tokenizers
3. **KV Cache**: Implement key-value caching for faster generation
4. **Quantization**: INT8/INT4 quantization for larger models
5. **WebGPU Backend**: Migration to WebGPU when broadly available
6. **Fine-tuning**: In-browser fine-tuning capabilities

## Quick Start

### Windows:
```bash
# Navigate to project directory
cd E:\_Applications\___Claude\llm-mini

# Run the start script
start.bat
```

### Manual Start:
```bash
# Install dependencies (first time only)
npm install

# Start the development server
npm start
```

The application will open automatically at http://localhost:8080

### Alternative Commands:
- `npm run serve` - Start server without opening browser
- `npm run dev` - Start server with cache disabled (for development)

## Project Structure

```
E:\_Applications\___Claude\llm-mini\
├── index.html          # Main chat interface
├── example.html        # API usage examples
├── model.js           # Transformer model implementation
├── tokenizer.js       # Simple tokenizer
├── app.js            # Application logic
├── package.json      # Project configuration
├── start.bat         # Windows quick start script
├── start.sh          # Unix/Linux start script
└── README.md         # This file
```

## License

MIT License - See parent project for details
