#!/bin/bash
# Quick start script for LLM Mini

echo "🚀 Starting LLM Mini..."
echo "===================================="
echo "GPU-Optimized Language Model"
echo "AMD RX 7900 XTX Ready"
echo "===================================="

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "🌐 Starting local server on http://localhost:8080"
echo "📝 Opening chat interface..."
npm start
