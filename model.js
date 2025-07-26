// Mini Transformer LLM Model Implementation
// Optimized for AMD RX 7900 XTX GPU performance

class MiniLLM {
    constructor(config = {}) {
        this.config = {
            vocabSize: config.vocabSize || 50257,
            hiddenSize: config.hiddenSize || 768,
            numLayers: config.numLayers || 12,
            numHeads: config.numHeads || 12,
            maxSeqLength: config.maxSeqLength || 512,
            dropout: config.dropout || 0.1,
            ...config
        };
        
        this.model = null;
        this.embeddings = null;
        this.positionalEncoding = null;
        this.attentionMasks = new Map();
        
        // GPU optimization flags
        this.gpuConfig = {
            enablePackedOperations: true,
            enableParallelExecution: true,
            forceFP16: false, // RX 7900 XTX has good FP32 performance
            textureMemoryHint: 4096, // Leverage your 4096x4096 texture capability
            batchParallelization: true
        };
    }
    
    async initialize() {
        console.log('Initializing Mini LLM with config:', this.config);
        
        // Set GPU optimization flags
        tf.env().set('WEBGL_PACK', this.gpuConfig.enablePackedOperations);
        tf.env().set('WEBGL_PACK_BINARY_OPERATIONS', true);
        tf.env().set('WEBGL_PACK_ARRAY_OPERATIONS', true);
        tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', this.gpuConfig.forceFP16);
        
        await this.buildModel();
        await this.initializeWeights();
        
        // Warm up the GPU
        await this.warmupGPU();
        
        return this;
    }
    
    async buildModel() {
        const inputs = tf.input({
            shape: [null, this.config.maxSeqLength],
            dtype: 'int32',
            name: 'input_ids'
        });
        
        // Token embeddings
        const embedding = tf.layers.embedding({
            inputDim: this.config.vocabSize,
            outputDim: this.config.hiddenSize,
            maskZero: false,
            name: 'token_embedding'
        });
        
        let x = embedding.apply(inputs);
        
        // Add positional encoding
        x = this.addPositionalEncoding(x);
        
        // Transformer blocks
        for (let i = 0; i < this.config.numLayers; i++) {
            x = await this.transformerBlock(x, i);
        }
        
        // Final layer norm
        x = tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-6,
            name: 'final_layernorm'
        }).apply(x);
        
        // Output projection
        const outputProjection = tf.layers.dense({
            units: this.config.vocabSize,
            activation: 'linear',
            name: 'output_projection'
        });
        
        const outputs = outputProjection.apply(x);
        
        this.model = tf.model({
            inputs: inputs,
            outputs: outputs,
            name: 'mini_llm'
        });
        
        console.log('Model architecture built successfully');
        this.model.summary();
    }
    
    addPositionalEncoding(x) {
        // Create a custom layer instead of using Lambda which is not available in browser
        class PositionalEncodingLayer extends tf.layers.Layer {
            constructor(config) {
                super(config);
            }
            
            computeOutputShape(inputShape) {
                return inputShape;
            }
            
            call(inputs, kwargs) {
                return tf.tidy(() => {
                    const input = inputs[0];
                    const inputShape = input.shape;
                    const batchSize = inputShape[0];
                    const seqLength = inputShape[1];
                    const hiddenSize = inputShape[2];
                    
                    // Create positional encoding
                    const positionIndices = tf.range(0, seqLength, 1, 'float32');
                    const dimensionIndices = tf.range(0, hiddenSize, 1, 'float32');
                    
                    const rates = tf.div(
                        dimensionIndices,
                        tf.scalar(hiddenSize)
                    ).mul(tf.scalar(-Math.log(10000.0)));
                    
                    const angleRates = tf.exp(rates);
                    const angleRads = tf.outerProduct(positionIndices, angleRates);
                    
                    // Apply sin to even indices, cos to odd indices
                    const sines = tf.sin(angleRads);
                    const cosines = tf.cos(angleRads);
                    
                    // Interleave sines and cosines
                    const posEncoding = tf.stack([sines, cosines], -1);
                    const posEncodingFinal = tf.reshape(posEncoding, [seqLength, hiddenSize]);
                    
                    // Expand for batch dimension
                    const posEncodingBatch = tf.expandDims(posEncodingFinal, 0);
                    const posEncodingExpanded = tf.tile(posEncodingBatch, [batchSize, 1, 1]);
                    
                    // Add to input
                    return tf.add(input, posEncodingExpanded);
                });
            }
            
            static get className() {
                return 'PositionalEncodingLayer';
            }
        }
        
        const posEncodingLayer = new PositionalEncodingLayer({ name: 'positional_encoding' });
        return posEncodingLayer.apply(x);
    }
    
    async transformerBlock(x, blockIndex) {
        const residual = x;
        
        // Multi-head attention
        x = tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-6,
            name: `block_${blockIndex}_ln1`
        }).apply(x);
        
        x = await this.multiHeadAttention(x, blockIndex);
        x = tf.layers.dropout({
            rate: this.config.dropout,
            name: `block_${blockIndex}_attn_dropout`
        }).apply(x);
        x = tf.layers.add().apply([residual, x]);
        
        // Feed-forward network
        const ffnResidual = x;
        x = tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-6,
            name: `block_${blockIndex}_ln2`
        }).apply(x);
        
        // FFN with GPU-optimized dimensions
        x = tf.layers.dense({
            units: this.config.hiddenSize * 4,
            activation: 'gelu',
            name: `block_${blockIndex}_ffn1`
        }).apply(x);
        
        x = tf.layers.dense({
            units: this.config.hiddenSize,
            activation: 'linear',
            name: `block_${blockIndex}_ffn2`
        }).apply(x);
        
        x = tf.layers.dropout({
            rate: this.config.dropout,
            name: `block_${blockIndex}_ffn_dropout`
        }).apply(x);
        
        x = tf.layers.add().apply([ffnResidual, x]);
        
        return x;
    }
    
    async multiHeadAttention(x, blockIndex) {
        const headDim = Math.floor(this.config.hiddenSize / this.config.numHeads);
        
        // Q, K, V projections - optimized for GPU parallelism
        const qkvProjection = tf.layers.dense({
            units: this.config.hiddenSize * 3,
            useBias: false,
            name: `block_${blockIndex}_qkv`
        });
        
        const qkv = qkvProjection.apply(x);
        
        // Create custom attention layer
        const blockIndex_ = blockIndex;
        const config = this.config;
        const getCausalMask = this.getCausalMask.bind(this);
        
        class MultiHeadAttentionLayer extends tf.layers.Layer {
            constructor(layerConfig) {
                super(layerConfig);
                this.headDim = headDim;
                this.numHeads = config.numHeads;
                this.hiddenSize = config.hiddenSize;
            }
            
            computeOutputShape(inputShape) {
                return [inputShape[0], inputShape[1], this.hiddenSize];
            }
            
            call(inputs, kwargs) {
                return tf.tidy(() => {
                    const input = inputs[0];
                    const inputShape = input.shape;
                    const batchSize = inputShape[0];
                    const seqLength = inputShape[1];
                    
                    // Split QKV
                    const qkvSplit = tf.split(input, 3, -1);
                    let [q, k, v] = qkvSplit;
                    
                    // Reshape for multi-head attention
                    const reshapeHeads = (t) => {
                        t = tf.reshape(t, [batchSize, seqLength, this.numHeads, this.headDim]);
                        return tf.transpose(t, [0, 2, 1, 3]);
                    };
                    
                    q = reshapeHeads(q);
                    k = reshapeHeads(k);
                    v = reshapeHeads(v);
                    
                    // Scaled dot-product attention
                    let scores = tf.matMul(q, k, false, true);
                    scores = tf.div(scores, tf.scalar(Math.sqrt(this.headDim)));
                    
                    // Apply causal mask
                    const mask = getCausalMask(seqLength);
                    const expandedMask = tf.expandDims(tf.expandDims(mask, 0), 0);
                    const tiledMask = tf.tile(expandedMask, [batchSize, this.numHeads, 1, 1]);
                    scores = tf.add(scores, tf.mul(tiledMask, tf.scalar(-1e9)));
                    
                    // Softmax
                    const attentionWeights = tf.softmax(scores, -1);
                    
                    // Apply attention to values
                    let output = tf.matMul(attentionWeights, v);
                    
                    // Reshape back
                    output = tf.transpose(output, [0, 2, 1, 3]);
                    output = tf.reshape(output, [batchSize, seqLength, this.hiddenSize]);
                    
                    return output;
                });
            }
            
            static get className() {
                return 'MultiHeadAttentionLayer';
            }
        }
        
        const attentionLayer = new MultiHeadAttentionLayer({ name: `block_${blockIndex}_attention` });
        const attentionOutput = attentionLayer.apply(qkv);
        
        // Output projection to match residual connection dimensions
        const outputProjection = tf.layers.dense({
            units: this.config.hiddenSize,
            useBias: false,
            name: `block_${blockIndex}_attn_output`
        });
        
        return outputProjection.apply(attentionOutput);
    }
    
    getCausalMask(seqLength) {
        const key = `mask_${seqLength}`;
        if (this.attentionMasks.has(key)) {
            return this.attentionMasks.get(key);
        }
        
        // Create causal mask
        const mask = tf.linalg.bandPart(
            tf.ones([seqLength, seqLength]),
            -1, 0
        );
        
        const causalMask = tf.sub(tf.scalar(1), mask);
        this.attentionMasks.set(key, causalMask);
        
        return causalMask;
    }
    
    async initializeWeights() {
        // Initialize with small random weights
        // In a real implementation, you would load pretrained weights
        const layers = this.model.layers;
        
        for (const layer of layers) {
            if (layer.getWeights().length > 0) {
                const weights = layer.getWeights();
                const newWeights = weights.map(w => {
                    const shape = w.shape;
                    const fanIn = shape[0];
                    const fanOut = shape[shape.length - 1];
                    const scale = Math.sqrt(2.0 / (fanIn + fanOut));
                    
                    return tf.randomNormal(shape, 0, scale);
                });
                
                layer.setWeights(newWeights);
            }
        }
        
        console.log('Model weights initialized');
    }
    
    async warmupGPU() {
        console.log('Warming up GPU...');
        
        // Run a few inference passes to warm up the GPU
        const dummyInput = tf.zeros([1, 32], 'int32');
        
        for (let i = 0; i < 3; i++) {
            const start = performance.now();
            const output = this.model.predict(dummyInput);
            await output.data();
            output.dispose();
            const end = performance.now();
            
            console.log(`Warmup pass ${i + 1}: ${(end - start).toFixed(2)}ms`);
        }
        
        dummyInput.dispose();
        console.log('GPU warmup complete');
    }
    
    async generate(inputIds, options = {}) {
        const {
            maxLength = 100,
            temperature = 0.7,
            topK = 40,
            topP = 0.9,
            doSample = true,
            padTokenId = 0,
            eosTokenId = 50256
        } = options;
        
        let currentIds = tf.tensor2d([inputIds], [1, inputIds.length], 'int32');
        const generatedTokens = [];
        
        const startTime = performance.now();
        let totalTokens = 0;
        
        for (let i = 0; i < maxLength; i++) {
            // Pad sequence if necessary
            const currentLength = currentIds.shape[1];
            if (currentLength < this.config.maxSeqLength) {
                const padding = tf.zeros([1, this.config.maxSeqLength - currentLength], 'int32');
                currentIds = tf.concat([currentIds, padding], 1);
            }
            
            // Get model predictions
            const outputs = this.model.predict(currentIds);
            
            // Get logits for the last position
            const lastLogits = tf.slice(
                outputs,
                [0, currentLength - 1, 0],
                [1, 1, this.config.vocabSize]
            );
            
            // Apply temperature
            const scaledLogits = tf.div(lastLogits, tf.scalar(temperature));
            
            // Apply top-k filtering
            let filteredLogits = scaledLogits;
            if (topK > 0) {
                filteredLogits = await this.topKFiltering(scaledLogits, topK);
            }
            
            // Sample from the distribution
            const probs = tf.softmax(filteredLogits, -1);
            const nextToken = doSample 
                ? await this.sampleFromDistribution(probs)
                : await this.greedyDecode(probs);
            
            generatedTokens.push(nextToken);
            totalTokens++;
            
            // Check for EOS token
            if (nextToken === eosTokenId) {
                break;
            }
            
            // Update input sequence
            const nextTokenTensor = tf.tensor2d([[nextToken]], [1, 1], 'int32');
            const newIds = tf.slice(currentIds, [0, 0], [1, currentLength]);
            currentIds.dispose();
            currentIds = tf.concat([newIds, nextTokenTensor], 1);
            
            // Clean up tensors
            outputs.dispose();
            lastLogits.dispose();
            scaledLogits.dispose();
            filteredLogits.dispose();
            probs.dispose();
            nextTokenTensor.dispose();
            newIds.dispose();
        }
        
        currentIds.dispose();
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const tokensPerSecond = (totalTokens / totalTime) * 1000;
        
        return {
            tokens: generatedTokens,
            tokensPerSecond: tokensPerSecond,
            totalTime: totalTime
        };
    }
    
    async topKFiltering(logits, k) {
        const [topKValues, topKIndices] = tf.topk(logits, k);
        
        // Create a mask for top-k values
        const minValue = tf.min(topKValues);
        const mask = tf.greaterEqual(logits, minValue);
        
        // Apply mask
        const filtered = tf.where(
            mask,
            logits,
            tf.fill(logits.shape, -Infinity)
        );
        
        topKValues.dispose();
        topKIndices.dispose();
        minValue.dispose();
        mask.dispose();
        
        return filtered;
    }
    
    async sampleFromDistribution(probs) {
        const probsArray = await probs.data();
        const cumsum = [];
        let sum = 0;
        
        for (let i = 0; i < probsArray.length; i++) {
            sum += probsArray[i];
            cumsum.push(sum);
        }
        
        const random = Math.random();
        for (let i = 0; i < cumsum.length; i++) {
            if (random < cumsum[i]) {
                return i;
            }
        }
        
        return cumsum.length - 1;
    }
    
    async greedyDecode(probs) {
        const probsArray = await probs.data();
        let maxIndex = 0;
        let maxValue = probsArray[0];
        
        for (let i = 1; i < probsArray.length; i++) {
            if (probsArray[i] > maxValue) {
                maxValue = probsArray[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    getMemoryUsage() {
        const memInfo = tf.memory();
        return {
            numTensors: memInfo.numTensors,
            numBytes: memInfo.numBytes,
            numBytesMB: (memInfo.numBytes / 1024 / 1024).toFixed(2)
        };
    }
    
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        
        // Dispose cached masks
        for (const [key, mask] of this.attentionMasks) {
            mask.dispose();
        }
        this.attentionMasks.clear();
    }
}

// Model size configurations
const MODEL_CONFIGS = {
    tiny: {
        vocabSize: 10000,
        hiddenSize: 256,
        numLayers: 4,
        numHeads: 4,
        maxSeqLength: 256
    },
    small: {
        vocabSize: 30000,
        hiddenSize: 512,
        numLayers: 6,
        numHeads: 8,
        maxSeqLength: 512
    },
    medium: {
        vocabSize: 50257,
        hiddenSize: 768,
        numLayers: 12,
        numHeads: 12,
        maxSeqLength: 512
    }
};

// Export for use in other modules
window.MiniLLM = MiniLLM;
window.MODEL_CONFIGS = MODEL_CONFIGS;
