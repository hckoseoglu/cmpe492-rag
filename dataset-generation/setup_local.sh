#!/bin/bash
set -e

MODEL="gemma2:9b"
OLLAMA_URL="http://localhost:11434"

echo "=== Local LLM Setup for Agentic Chunking Pipeline ==="
echo ""

# 1. Check/install Ollama
if command -v ollama &>/dev/null; then
    echo "[OK] Ollama is already installed: $(ollama --version)"
else
    echo "[..] Ollama not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "[ERROR] Homebrew is not installed. Install it first: https://brew.sh"
        exit 1
    fi
    brew install ollama
    echo "[OK] Ollama installed: $(ollama --version)"
fi

# 2. Start Ollama server if not running
if curl -s "$OLLAMA_URL" &>/dev/null; then
    echo "[OK] Ollama server is already running"
else
    echo "[..] Starting Ollama server in background..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!

    # Wait for server to be ready (up to 30 seconds)
    for i in $(seq 1 30); do
        if curl -s "$OLLAMA_URL" &>/dev/null; then
            echo "[OK] Ollama server is running (PID: $OLLAMA_PID)"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "[ERROR] Ollama server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
fi

# 3. Pull model if not already available
if ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "[OK] Model '$MODEL' is already pulled"
else
    echo "[..] Pulling '$MODEL' (this may take a while, ~5.5GB)..."
    ollama pull "$MODEL"
    echo "[OK] Model '$MODEL' pulled successfully"
fi

# 4. Verify with a test prompt
echo "[..] Running test prompt..."
RESPONSE=$(ollama run "$MODEL" "Reply with only: OK" 2>/dev/null | head -1)
if [ -n "$RESPONSE" ]; then
    echo "[OK] Model responded: $RESPONSE"
else
    echo "[ERROR] Model did not respond. Check 'ollama serve' logs."
    exit 1
fi

# 5. Next steps
echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the chunking pipeline:"
echo "  cd dataset-generation"
echo "  pip install -r requirements.txt"
echo "  python chunker.py --pdf \"progression_models_in_resistance_training.pdf\""
echo ""
echo "To process all PDFs:"
echo "  python chunker.py"
