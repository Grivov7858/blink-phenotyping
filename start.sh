#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo ""
echo "  Blink Phenotyping Tool is starting..."
echo "  Open your browser to: http://localhost:5000"
echo "  Press Ctrl+C to stop."
echo ""
python3 app.py
