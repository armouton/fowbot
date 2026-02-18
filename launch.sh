#!/bin/bash
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
    echo "  Mac: brew install python3"
    echo "  Ubuntu: sudo apt install python3 python3-venv"
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
    echo ""
    echo "Creating virtual environment (first time only)..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Launch server, open browser after short delay
echo ""
echo "Starting server at http://localhost:5001"
echo "Press Ctrl+C to stop the server."
echo ""

# Open browser (sleep gives server time to start)
(sleep 2 && open http://localhost:5001 2>/dev/null || xdg-open http://localhost:5001 2>/dev/null) &

python app.py
