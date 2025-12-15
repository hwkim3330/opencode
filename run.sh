#!/bin/bash
# LiquidCode 실행 스크립트 (LFM2-VL 비전 지원)

cd "$(dirname "$0")"

# Check dependencies
python3 -c "import rich, torch, transformers, PIL, mss" 2>/dev/null || {
    echo "Installing dependencies..."
    pip3 install --break-system-packages rich torch transformers accelerate pillow mss
}

# Run LiquidCode with Vision
python3 liquidcode.py "$@"
