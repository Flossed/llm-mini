// Main Application Logic for Mini LLM
// Leverages AMD RX 7900 XTX GPU capabilities

let model = null;
let tokenizer = null;
let isGenerating = false;
let currentModelSize = 'small';

// Initialize TensorFlow.js with GPU optimizations
async function initializeTF() {
    try {
        // Set WebGL backend
        await tf.setBackend('webgl');
        
        // Get backend info
        const backend = tf.getBackend();
        document.getElementById('backend-info').innerHTML = `<strong>Backend:</strong> ${backend}`;
        document.getElementById('backend-info').className = 'status success';
        
        // Get WebGL info
        const gl = document.createElement('canvas').getContext('webgl2') || 
                   document.createElement('canvas').getContext('webgl');
        
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        
        document.getElementById('gpu-info').innerHTML = 
            `<strong>GPU:</strong> ${vendor} ${renderer}<br>
             <strong>WebGL:</strong> ${gl.getParameter(gl.VERSION)}`;
        document.getElementById('gpu-info').className = 'status success';
        
        // Set environment flags for RX 7900 XTX optimization
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false); // RX 7900 XTX performs well with FP32
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
        tf.env().set('WEBGL_CPU_FORWARD', false);
        tf.env().set('WEBGL_MAX_TEXTURE_SIZE', 16384); // RX 7900 XTX can handle large textures
        
        console.log('TensorFlow.js initialized successfully');
        return true;
    } catch (error) {
        console.error('Error initializing TensorFlow.js:', error);
        document.getElementById('backend-info').innerHTML = 
            `<strong>Error:</strong> ${error.message}`;
        document.getElementById('backend-info').className = 'status error';
        return false;
    }
}

// Initialize the model
async function initializeModel(size = 'small') {
    try {
        document.getElementById('model-status').innerHTML = `Loading ${size} model...`;
        document.getElementById('model-status').className = 'status loading';
        
        // Dispose of previous model if exists
        if (model) {
            model.dispose();
        }
        
        // Create new model with selected configuration
        const config = MODEL_CONFIGS[size];
        model = new MiniLLM(config);
        await model.initialize();
        
        // Update tokenizer with matching vocab size
        tokenizer = new SimpleTokenizer(config.vocabSize);
        
        // Update UI
        document.getElementById('model-status').innerHTML = 
            `<strong>Model:</strong> ${size} (${config.numLayers} layers, ${config.hiddenSize} hidden size)`;
        document.getElementById('model-status').className = 'status success';
        
        // Enable chat interface
        document.getElementById('userInput').disabled = false;
        document.getElementById('generateBtn').disabled = false;
        
        // Update memory usage
        updateMemoryUsage();
        
        currentModelSize = size;
        console.log(`Model initialized: ${size}`);
    } catch (error) {
        console.error('Error initializing model:', error);
        document.getElementById('model-status').innerHTML = 
            `<strong>Error:</strong> ${error.message}`;
        document.getElementById('model-status').className = 'status error';
    }
}

// Generate response
async function generateResponse() {
    if (isGenerating || !model || !tokenizer) return;
    
    const userInput = document.getElementById('userInput').value.trim();
    if (!userInput) return;
    
    isGenerating = true;
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('progressBar').style.display = 'block';
    
    // Add user message to chat
    addMessage(userInput, 'user');
    document.getElementById('userInput').value = '';
    
    try {
        // Tokenize input
        const inputTokens = tokenizer.encode(userInput, true);
        console.log('Input tokens:', inputTokens);
        
        // Get generation parameters
        const temperature = parseFloat(document.getElementById('temperature').value);
        const maxTokens = parseInt(document.getElementById('max-tokens').value);
        const topK = parseInt(document.getElementById('top-k').value);
        
        // Show generating indicator
        const messageId = addMessage('', 'assistant', true);
        
        // Generate response
        const startTime = performance.now();
        const result = await model.generate(inputTokens, {
            maxLength: maxTokens,
            temperature: temperature,
            topK: topK,
            doSample: temperature > 0
        });
        
        const endTime = performance.now();
        
        // Decode response
        const responseText = tokenizer.decode(result.tokens, true);
        
        // Update message with generated text
        updateMessage(messageId, responseText);
        
        // Update metrics
        document.getElementById('tokensPerSecond').textContent = 
            result.tokensPerSecond.toFixed(1);
        document.getElementById('inferenceTime').textContent = 
            (endTime - startTime).toFixed(0);
        
        updateMemoryUsage();
        
    } catch (error) {
        console.error('Error generating response:', error);
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        isGenerating = false;
        document.getElementById('generateBtn').disabled = false;
        document.getElementById('progressBar').style.display = 'none';
    }
}

// Add message to chat history
function addMessage(text, sender, isGenerating = false) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    if (isGenerating) {
        messageDiv.classList.add('generating');
    }
    
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;
    messageDiv.textContent = text || '▌';
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return messageId;
}

// Update message content
function updateMessage(messageId, text) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        messageDiv.textContent = text;
        messageDiv.classList.remove('generating');
    }
}

// Update memory usage display
function updateMemoryUsage() {
    if (!model) return;
    
    const memInfo = model.getMemoryUsage();
    document.getElementById('memoryUsage').textContent = memInfo.numBytesMB;
}

// Event listeners
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize TensorFlow.js
    const tfInitialized = await initializeTF();
    if (!tfInitialized) return;
    
    // Initialize model
    await initializeModel('small');
    
    // Model size change
    document.getElementById('model-size').addEventListener('change', async (e) => {
        await initializeModel(e.target.value);
    });
    
    // Temperature slider
    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temp-value').textContent = e.target.value;
    });
    
    // Enter key to generate
    document.getElementById('userInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateResponse();
        }
    });
    
    // Monitor memory usage
    setInterval(updateMemoryUsage, 5000);
});

// Benchmark function for testing GPU performance
async function runBenchmark() {
    if (!model) return;
    
    console.log('Running benchmark...');
    const batchSizes = [1, 2, 4, 8, 16];
    const sequenceLengths = [32, 64, 128, 256];
    
    for (const batchSize of batchSizes) {
        for (const seqLength of sequenceLengths) {
            // Create dummy input
            const dummyInput = tf.zeros([batchSize, seqLength], 'int32');
            
            // Warm up
            const warmupOutput = model.model.predict(dummyInput);
            await warmupOutput.data();
            warmupOutput.dispose();
            
            // Benchmark
            const iterations = 10;
            const start = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                const output = model.model.predict(dummyInput);
                await output.data();
                output.dispose();
            }
            
            const end = performance.now();
            const avgTime = (end - start) / iterations;
            const throughput = (batchSize * seqLength) / (avgTime / 1000);
            
            console.log(`Batch: ${batchSize}, Seq: ${seqLength}, Avg Time: ${avgTime.toFixed(2)}ms, Throughput: ${throughput.toFixed(0)} tokens/sec`);
            
            dummyInput.dispose();
        }
    }
    
    console.log('Benchmark complete');
}

// Export for debugging
window.runBenchmark = runBenchmark;
window.model = () => model;
window.tokenizer = () => tokenizer;
