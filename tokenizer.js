// Simple Tokenizer Implementation for Mini LLM
// This is a basic tokenizer - in production, you'd use a proper tokenizer like GPT-2's

class SimpleTokenizer {
    constructor(vocabSize = 50257) {
        this.vocabSize = vocabSize;
        this.vocab = new Map();
        this.reverseVocab = new Map();
        this.specialTokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4,
            '<cls>': 5
        };
        
        this.initializeVocab();
    }
    
    initializeVocab() {
        // Add special tokens
        let index = 0;
        for (const [token, id] of Object.entries(this.specialTokens)) {
            this.vocab.set(token, id);
            this.reverseVocab.set(id, token);
            index = Math.max(index, id + 1);
        }
        
        // Add ASCII printable characters
        for (let i = 32; i < 127; i++) {
            const char = String.fromCharCode(i);
            if (!this.vocab.has(char)) {
                this.vocab.set(char, index);
                this.reverseVocab.set(index, char);
                index++;
            }
        }
        
        // Add common words (simplified vocabulary)
        const commonWords = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
            'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'getting',
            'made', 'find', 'where', 'much', 'too', 'very', 'still', 'being', 'going', 'why',
            'before', 'never', 'here', 'more', 'always', 'those', 'tell', 'really', 'something', 'put',
            'thing', 'long', 'take', 'see', 'make', 'many', 'over', 'such', 'great', 'think',
            'say', 'help', 'low', 'line', 'differ', 'turn', 'cause', 'same', 'mean', 'part',
            'start', 'seem', 'next', 'sound', 'take', 'only', 'little', 'round', 'man', 'year',
            'came', 'show', 'every', 'good', 'me', 'give', 'our', 'under', 'name', 'very',
            'through', 'just', 'form', 'sentence', 'great', 'think', 'say', 'help', 'low', 'line'
        ];
        
        // Add common word variations
        for (const word of commonWords) {
            if (index >= this.vocabSize) break;
            
            // Lowercase
            if (!this.vocab.has(word)) {
                this.vocab.set(word, index++);
                this.reverseVocab.set(index - 1, word);
            }
            
            // Uppercase
            const upperWord = word.toUpperCase();
            if (!this.vocab.has(upperWord) && index < this.vocabSize) {
                this.vocab.set(upperWord, index++);
                this.reverseVocab.set(index - 1, upperWord);
            }
            
            // Capitalized
            const capitalizedWord = word.charAt(0).toUpperCase() + word.slice(1);
            if (!this.vocab.has(capitalizedWord) && index < this.vocabSize) {
                this.vocab.set(capitalizedWord, index++);
                this.reverseVocab.set(index - 1, capitalizedWord);
            }
        }
        
        // Add common punctuation combinations
        const punctuations = ['.', ',', '!', '?', ';', ':', '"', "'", '-', '(', ')', '[', ']', '{', '}'];
        for (const punct of punctuations) {
            if (!this.vocab.has(punct) && index < this.vocabSize) {
                this.vocab.set(punct, index++);
                this.reverseVocab.set(index - 1, punct);
            }
        }
        
        // Fill remaining vocabulary with number combinations
        for (let i = 0; i < 1000 && index < this.vocabSize; i++) {
            const numStr = i.toString();
            if (!this.vocab.has(numStr)) {
                this.vocab.set(numStr, index++);
                this.reverseVocab.set(index - 1, numStr);
            }
        }
        
        console.log(`Tokenizer initialized with ${this.vocab.size} tokens`);
    }
    
    encode(text, addSpecialTokens = true) {
        const tokens = [];
        
        if (addSpecialTokens) {
            tokens.push(this.specialTokens['<bos>']);
        }
        
        // Simple word-level tokenization with character fallback
        const words = text.match(/\w+|[^\w\s]/g) || [];
        
        for (const word of words) {
            if (this.vocab.has(word)) {
                tokens.push(this.vocab.get(word));
            } else if (this.vocab.has(word.toLowerCase())) {
                tokens.push(this.vocab.get(word.toLowerCase()));
            } else {
                // Character-level fallback for unknown words
                for (const char of word) {
                    if (this.vocab.has(char)) {
                        tokens.push(this.vocab.get(char));
                    } else {
                        tokens.push(this.specialTokens['<unk>']);
                    }
                }
            }
        }
        
        if (addSpecialTokens) {
            tokens.push(this.specialTokens['<eos>']);
        }
        
        return tokens;
    }
    
    decode(tokens, skipSpecialTokens = true) {
        const decoded = [];
        
        for (const token of tokens) {
            if (this.reverseVocab.has(token)) {
                const word = this.reverseVocab.get(token);
                
                if (skipSpecialTokens && this.isSpecialToken(word)) {
                    continue;
                }
                
                decoded.push(word);
            } else {
                decoded.push('<unk>');
            }
        }
        
        // Simple post-processing to join words
        let text = decoded.join(' ');
        
        // Fix spacing around punctuation
        text = text.replace(/\s+([.,!?;:)])/g, '$1');
        text = text.replace(/([(\[])\s+/g, '$1');
        text = text.replace(/\s+'/g, "'");
        text = text.replace(/"\s+/g, '"');
        text = text.replace(/\s+"/g, '"');
        
        return text.trim();
    }
    
    isSpecialToken(token) {
        return token in this.specialTokens;
    }
    
    getVocabSize() {
        return this.vocab.size;
    }
    
    tokenToId(token) {
        return this.vocab.get(token) || this.specialTokens['<unk>'];
    }
    
    idToToken(id) {
        return this.reverseVocab.get(id) || '<unk>';
    }
    
    // Batch encoding for efficiency
    batchEncode(texts, addSpecialTokens = true, maxLength = null) {
        const encodedBatch = [];
        let maxLen = 0;
        
        for (const text of texts) {
            const encoded = this.encode(text, addSpecialTokens);
            if (maxLength && encoded.length > maxLength) {
                encoded.length = maxLength;
            }
            encodedBatch.push(encoded);
            maxLen = Math.max(maxLen, encoded.length);
        }
        
        // Pad sequences to same length
        const paddedBatch = encodedBatch.map(seq => {
            const padLength = maxLen - seq.length;
            return seq.concat(new Array(padLength).fill(this.specialTokens['<pad>']));
        });
        
        return paddedBatch;
    }
    
    // Get attention mask for padded sequences
    getAttentionMask(tokenIds) {
        return tokenIds.map(id => id !== this.specialTokens['<pad>'] ? 1 : 0);
    }
}

// Export for use in other modules
window.SimpleTokenizer = SimpleTokenizer;
