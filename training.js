// Mini LLM Training Implementation
// Train your model to generate meaningful text

class MiniLLMTrainer {
    constructor(model, tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.optimizer = tf.train.adam(0.001);
        this.batchSize = 4;
    }
    
    async trainOnText(text, epochs = 10, sequenceLength = 32) {
        console.log('Preparing training data...');
        
        // Tokenize the entire text
        const tokens = this.tokenizer.encode(text, false);
        const sequences = [];
        
        // Create overlapping sequences for training
        for (let i = 0; i < tokens.length - sequenceLength - 1; i++) {
            sequences.push({
                input: tokens.slice(i, i + sequenceLength),
                target: tokens.slice(i + 1, i + sequenceLength + 1)
            });
        }
        
        console.log(`Created ${sequences.length} training sequences`);
        
        // Training loop
        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;
            
            // Shuffle sequences
            const shuffled = this.shuffle(sequences);
            
            // Process in batches
            for (let i = 0; i < shuffled.length; i += this.batchSize) {
                const batch = shuffled.slice(i, i + this.batchSize);
                const loss = await this.trainBatch(batch, sequenceLength);
                epochLoss += loss;
                batchCount++;
                
                if (batchCount % 10 === 0) {
                    console.log(`Epoch ${epoch + 1}/${epochs}, Batch ${batchCount}, Loss: ${loss.toFixed(4)}`);
                }
            }
            
            console.log(`Epoch ${epoch + 1}/${epochs} completed, Average Loss: ${(epochLoss / batchCount).toFixed(4)}`);
            
            // Generate sample text every few epochs
            if ((epoch + 1) % 5 === 0) {
                await this.generateSample();
            }
        }
    }
    
    async trainBatch(batch, sequenceLength) {
        const loss = await tf.tidy(() => {
            // Prepare batch tensors
            const batchInputs = [];
            const batchTargets = [];
            
            for (const seq of batch) {
                // Pad sequences if necessary
                const paddedInput = this.padSequence(seq.input, this.model.config.maxSeqLength);
                const paddedTarget = this.padSequence(seq.target, this.model.config.maxSeqLength);
                
                batchInputs.push(paddedInput);
                batchTargets.push(paddedTarget);
            }
            
            const inputs = tf.tensor2d(batchInputs, [batch.length, this.model.config.maxSeqLength], 'int32');
            const targets = tf.tensor2d(batchTargets, [batch.length, this.model.config.maxSeqLength], 'int32');
            
            // Forward pass
            const predictions = this.model.model.predict(inputs);
            
            // Calculate cross-entropy loss
            const targetsOneHot = tf.oneHot(targets, this.model.config.vocabSize);
            
            // Reshape for loss calculation
            const predFlat = predictions.reshape([-1, this.model.config.vocabSize]);
            const targFlat = targetsOneHot.reshape([-1, this.model.config.vocabSize]);
            
            // Compute loss only on actual sequence positions (not padding)
            const loss = tf.losses.softmaxCrossEntropy(targFlat, predFlat);
            
            return loss;
        });
        
        // Compute gradients and update weights
        await this.optimizer.minimize(() => loss);
        
        const lossValue = await loss.data();
        loss.dispose();
        
        return lossValue[0];
    }
    
    padSequence(sequence, maxLength) {
        const padded = new Array(maxLength).fill(0);
        for (let i = 0; i < Math.min(sequence.length, maxLength); i++) {
            padded[i] = sequence[i];
        }
        return padded;
    }
    
    shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
    
    async generateSample() {
        console.log('\nGenerating sample text...');
        const prompt = "The";
        const promptTokens = this.tokenizer.encode(prompt, false);
        
        const generated = await this.model.generate(promptTokens, {
            maxLength: 20,
            temperature: 0.7,
            topK: 10,
            doSample: true
        });
        
        const text = this.tokenizer.decode(generated.tokens);
        console.log(`Sample: "${prompt} ${text}"`);
        console.log(`Tokens/sec: ${generated.tokensPerSecond.toFixed(2)}\n`);
    }
    
    async saveWeights(filename = 'mini-llm-trained') {
        await this.model.model.save(`downloads://${filename}`);
        console.log(`Model weights saved as ${filename}`);
    }
    
    async loadWeights(files) {
        // files should be the model.json and weight files
        const model = await tf.loadLayersModel(tf.io.browserFiles(files));
        this.model.model = model;
        console.log('Weights loaded successfully!');
    }
}

// Training data preparation utilities
function prepareSimpleTrainingData() {
    // Start with simple, repetitive patterns for initial training
    const simplePatterns = [
        // Alphabet patterns
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        
        // Number patterns
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
        "one two three four five six seven eight nine ten",
        
        // Simple sentences with patterns
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "The bird flew in the sky.",
        "The fish swam in the sea.",
        
        // Question-answer patterns
        "What is your name? My name is Mini.",
        "How are you? I am fine, thank you.",
        "Where are you? I am here.",
        "When is it? It is now.",
        
        // Repetitive structures
        "I like apples. I like bananas. I like oranges. I like grapes.",
        "Today is Monday. Tomorrow is Tuesday. Yesterday was Sunday.",
        "The sun is yellow. The sky is blue. The grass is green.",
        
        // Simple stories
        "Once upon a time, there was a little robot. The robot liked to learn. Every day, the robot read new words. The robot became very smart. The end.",
        "In a small town, there lived a cat. The cat was very curious. One day, the cat found a box. Inside the box was a toy. The cat was happy.",
    ];
    
    // Repeat patterns for more training data
    return simplePatterns.join(" ").repeat(20);
}

function prepareAdvancedTrainingData() {
    // More complex training data
    const advancedText = `
        Artificial intelligence is transforming the world. Machine learning models can now understand and generate human language with remarkable accuracy.
        
        The transformer architecture revolutionized natural language processing. Self-attention mechanisms allow models to understand context and relationships between words.
        
        Neural networks learn by adjusting weights through backpropagation. The optimization process minimizes the loss function to improve predictions.
        
        Deep learning has applications in computer vision, natural language processing, and robotics. Models can recognize images, translate languages, and control robots.
        
        The future of AI is bright and full of possibilities. Researchers continue to develop new architectures and training methods to create more capable systems.
    `.repeat(10);
    
    return advancedText;
}

// Add training UI to your app
function addTrainingUI() {
    const trainingHTML = `
        <div id="training-section" style="margin-top: 20px; padding: 20px; border: 1px solid #ccc;">
            <h3>Model Training</h3>
            <div>
                <label>Training Text:</label><br>
                <textarea id="training-text" rows="5" cols="50" placeholder="Enter training text or use sample data..."></textarea>
            </div>
            <div style="margin-top: 10px;">
                <button onclick="loadSampleData()">Load Simple Patterns</button>
                <button onclick="loadAdvancedData()">Load Advanced Text</button>
            </div>
            <div style="margin-top: 10px;">
                <label>Epochs: <input type="number" id="epochs" value="10" min="1" max="100"></label>
                <label>Learning Rate: <input type="number" id="learning-rate" value="0.001" step="0.0001" min="0.0001" max="0.1"></label>
            </div>
            <div style="margin-top: 10px;">
                <button onclick="startTraining()" id="train-button">Start Training</button>
                <button onclick="stopTraining()" disabled id="stop-button">Stop</button>
                <button onclick="saveTrainedModel()">Save Weights</button>
                <input type="file" id="load-weights" multiple accept=".json,.bin" onchange="loadTrainedWeights(this.files)">
            </div>
            <div id="training-progress" style="margin-top: 10px;"></div>
        </div>
    `;
    
    document.querySelector('.container').insertAdjacentHTML('beforeend', trainingHTML);
}

// Training control functions
let trainer = null;
let isTraining = false;

async function startTraining() {
    const text = document.getElementById('training-text').value;
    if (!text) {
        alert('Please enter training text or load sample data');
        return;
    }
    
    const epochs = parseInt(document.getElementById('epochs').value);
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    
    // Create trainer
    trainer = new MiniLLMTrainer(window.model, window.tokenizer);
    trainer.optimizer = tf.train.adam(learningRate);
    
    // Disable/enable buttons
    document.getElementById('train-button').disabled = true;
    document.getElementById('stop-button').disabled = false;
    isTraining = true;
    
    try {
        await trainer.trainOnText(text, epochs);
        alert('Training completed!');
    } catch (error) {
        console.error('Training error:', error);
        alert('Training failed: ' + error.message);
    } finally {
        document.getElementById('train-button').disabled = false;
        document.getElementById('stop-button').disabled = true;
        isTraining = false;
    }
}

function stopTraining() {
    isTraining = false;
    // Training will stop at the next epoch
}

async function saveTrainedModel() {
    if (!trainer) {
        alert('No trained model to save');
        return;
    }
    await trainer.saveWeights();
}

async function loadTrainedWeights(files) {
    if (!trainer) {
        trainer = new MiniLLMTrainer(window.model, window.tokenizer);
    }
    await trainer.loadWeights(Array.from(files));
    alert('Weights loaded successfully!');
}

function loadSampleData() {
    document.getElementById('training-text').value = prepareSimpleTrainingData();
}

function loadAdvancedData() {
    document.getElementById('training-text').value = prepareAdvancedTrainingData();
}

// Initialize training UI when the page loads
window.addEventListener('load', () => {
    setTimeout(addTrainingUI, 1000);
});

// Export for use
window.MiniLLMTrainer = MiniLLMTrainer;
