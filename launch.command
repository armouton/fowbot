#!/bin/bash
# Double-click this file on macOS to launch the app

# cd to the folder where this script lives
cd "$(dirname "$0")"

echo "===================================="
echo " Next-Word Predictor - Web Version"
echo "===================================="
echo ""

# Check for Python 3
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Python not found!"
    echo "  Install with: brew install python3"
    echo "  Or download from https://www.python.org"
    echo ""
    echo "Press any key to close..."
    read -n 1
    exit 1
fi

echo "Found Python: $($PYTHON --version)"
echo ""
echo "This will install required packages (numpy, flask) if not already present."
echo "Press any key to continue, or close this window to cancel."
read -n 1
echo ""

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment (first time only)..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "Installing dependencies (if needed)..."
pip install -r requirements.txt --quiet

# Launch server, open browser after short delay
echo ""
echo "Starting server at http://localhost:5001"
echo "Close this window to stop the server."
echo ""

(sleep 2 && open http://localhost:5001) &

python app.py
