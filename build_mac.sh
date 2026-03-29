#!/bin/bash
# Run this script on a macOS machine to build the standalone app.
# Usage: bash build_mac.sh
set -e

cd "$(dirname "$0")"

echo "=== Setting up Python environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

echo ""
echo "=== Building macOS app ==="
pyinstaller \
    --name "Blink Phenotyping" \
    --onedir \
    --windowed \
    --add-data "templates:templates" \
    --add-data "static:static" \
    --hidden-import mediapipe \
    --hidden-import sklearn \
    --hidden-import sklearn.linear_model \
    --hidden-import sklearn.metrics \
    --hidden-import sklearn.model_selection \
    --hidden-import sklearn.utils._typedefs \
    --hidden-import sklearn.utils._heap \
    --hidden-import sklearn.utils._sorting \
    --hidden-import sklearn.utils._vector_sentinel \
    --collect-data mediapipe \
    --collect-data sklearn \
    app.py

echo ""
echo "=== Done! ==="
echo "The app is at: dist/Blink Phenotyping.app"
echo "Copy it to the Applications folder or share it directly."
