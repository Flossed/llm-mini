# Getting Trained Weights for Mini LLM

## Option 1: Train the Model Yourself (Recommended for Learning)

### Create a Simple Training Script

```javascript
// training.js
class MiniLLMTrainer {
    constructor(model, tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.optimizer = tf.train.adam(0.001);
    }
    
    async trainOnText(text, epochs = 10, sequenceLength = 32) {
        // Prepare training data
        const tokens = this.tokenizer.encode(text);
        const sequences = [];
        
        // Create overlapping sequences
        for (let i = 0; i < tokens.length - sequenceLength; i++) {
            sequences.push({
                input: tokens.slice(i, i + sequenceLength),
                target: tokens.slice(i + 1, i + sequenceLength + 1)
            });
        }
        
        console.log(`Training on ${sequences.length} sequences...`);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            
            for (const seq of sequences) {
                const loss = await this.trainStep(seq.input, seq.target);
                totalLoss += loss;
            }
            
            console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${totalLoss / sequences.length}`);
        }
    }
    
    async trainStep(inputIds, targetIds) {
        const loss = tf.tidy(() => {
            // Forward pass
            const predictions = this.model.model.predict(
                tf.tensor2d([inputIds], [1, inputIds.length])
            );
            
            // Calculate loss (cross-entropy)
            const targets = tf.oneHot(targetIds, this.model.config.vocabSize);
            const loss = tf.losses.softmaxCrossEntropy(
                targets.reshape([-1, this.model.config.vocabSize]),
                predictions.reshape([-1, this.model.config.vocabSize])
            );
            
            return loss;
        });
        
        // Compute gradients and update weights
        const grads = tf.grads((x) => loss)(this.model.model.trainableWeights);
        this.optimizer.applyGradients(
            grads.map((g, i) => ({
                tensor: this.model.model.trainableWeights[i],
                gradient: g
            }))
        );
        
        const lossValue = await loss.data();
        loss.dispose();
        
        return lossValue[0];
    }
    
    async saveWeights(filename = 'model-weights') {
        await this.model.model.save(`downloads://${filename}`);
        console.log('Weights saved!');
    }
}

// Usage example:
async function trainModel() {
    // Load some training text
    const trainingText = `
        The quick brown fox jumps over the lazy dog.
        Machine learning is fascinating and powerful.
        Neural networks can learn complex patterns.
        Transformers revolutionized natural language processing.
        // Add more text here...
    `.repeat(100); // Repeat for more training data
    
    const trainer = new MiniLLMTrainer(window.model, window.tokenizer);
    await trainer.trainOnText(trainingText, epochs=50);
    await trainer.saveWeights();
}
```

## Option 2: Use Pre-trained Weights from Small Models

### Convert Existing Model Weights

1. **Use a small GPT-2 model** (TensorFlow.js compatible):

```javascript
// Option A: Load from TensorFlow Hub (if available)
async function loadPretrainedGPT2() {
    try {
        // This would load a small GPT-2 variant
        const model = await tf.loadLayersModel(
            'https://tfhub.dev/tensorflow/tfjs-model/gpt2-small/1/model.json'
        );
        return model;
    } catch (error) {
        console.error('Model not available:', error);
    }
}

// Option B: Convert from Hugging Face model
// Use Python script to convert and export:
```

```python
# convert_model.py
import tensorflow as tf
import transformers
import tensorflowjs as tfjs

# Load a small model
model = transformers.TFGPT2Model.from_pretrained('distilgpt2')
model.save_pretrained('./distilgpt2_tf', saved_model=True)

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(
    model,
    './tfjs_model'
)
```

## Option 3: Create a Tiny Pre-trained Model

### Train on Simple Patterns First

```javascript
// Create a simple pattern-learning dataset
function createSimpleDataset() {
    const patterns = [
        // Simple sequences
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
        "the cat sat on the mat",
        "the dog ran in the park",
        // Repetitive patterns
        "hello world hello world hello world",
        "yes no yes no yes no yes no",
        // Simple conversations
        "hello how are you i am fine thank you",
        "what is your name my name is mini",
    ];
    
    return patterns.join(" ").repeat(50);
}

// Train on simple patterns first
async function pretrainOnPatterns() {
    const simpleText = createSimpleDataset();
    const trainer = new MiniLLMTrainer(window.model, window.tokenizer);
    
    // Start with high learning rate for simple patterns
    trainer.optimizer = tf.train.adam(0.01);
    await trainer.trainOnText(simpleText, epochs=100, sequenceLength=16);
    
    // Then fine-tune on more complex text
    trainer.optimizer = tf.train.adam(0.001);
    const complexText = loadYourTextCorpus(); // Your text data
    await trainer.trainOnText(complexText, epochs=50);
}
```

## Option 4: Use Transfer Learning

### Load Partial Weights from Compatible Models

```javascript
async function transferLearnFromBERT() {
    // Load embeddings from a small BERT model
    const bertModel = await tf.loadLayersModel(
        'path/to/small-bert-embeddings/model.json'
    );
    
    // Transfer embedding weights
    const bertEmbeddings = bertModel.getLayer('embeddings').getWeights();
    window.model.model.getLayer('token_embedding').setWeights(bertEmbeddings);
    
    console.log('Transferred embedding weights from BERT!');
}
```

## Option 5: Quick Demo Weights

### Create Weights for Basic Demo

```javascript
// Create demo weights that at least produce common words
function createDemoWeights() {
    const vocabSize = window.model.config.vocabSize;
    const hiddenSize = window.model.config.hiddenSize;
    
    // Create embedding weights biased toward common tokens
    const embeddingWeights = tf.tidy(() => {
        const weights = tf.randomNormal([vocabSize, hiddenSize], 0, 0.1);
        
        // Bias common word indices
        const commonIndices = [
            window.tokenizer.vocab.get('the'),
            window.tokenizer.vocab.get('a'),
            window.tokenizer.vocab.get('is'),
            window.tokenizer.vocab.get('it'),
            // ... more common words
        ].filter(idx => idx !== undefined);
        
        // Make common words more likely
        return weights; // Modify as needed
    });
    
    window.model.model.getLayer('token_embedding').setWeights([embeddingWeights]);
}
```

## Recommended Approach for Quick Results

1. **Start with Simple Pattern Training** (Option 3)
   - Quick to implement
   - Shows immediate improvement
   - Good for understanding how training works

2. **Train on Small Corpus** (Option 1)
   - Use public domain texts (Project Gutenberg)
   - Start with 1-10MB of text
   - Train for several hours

3. **Save and Load Weights**

```javascript
// Save after training
await model.model.save('downloads://mini-llm-weights');

// Load in future sessions
async function loadSavedWeights() {
    const weightFiles = document.getElementById('weight-upload').files;
    const weights = await tf.loadLayersModel(
        tf.io.browserFiles([weightFiles[0], weightFiles[1]])
    );
    window.model.model = weights;
}
```

## Simple Training Data Sources

1. **Public Domain Books**: Project Gutenberg
2. **Wikipedia Dumps**: Simple English Wikipedia
3. **Common Crawl**: Filtered web text
4. **Your Own Text**: Stories, articles, chat logs

## Next Steps

1. Implement the training script
2. Prepare a small text corpus (start with 1MB)
3. Train for a few hours on your GPU
4. Save the weights
5. Test generation with trained model

The key is to start small and gradually increase complexity. Even training on a small corpus will produce much better results than random weights!
