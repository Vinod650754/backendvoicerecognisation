#!/bin/bash
# Start the backend server locally

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "======================================================================"
echo "                   BACKEND SERVER STARTUP"
echo "======================================================================"
echo ""

# Check if running from project root
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run this script from the project root directory"
    echo "Also ensure .venv is activated"
    exit 1
fi

echo "[1/3] Checking environment..."
echo "Backend Directory: $SCRIPT_DIR"
echo ""

echo "[2/3] Creating data directories..."
for DIR in wakeword command open_door close_door; do
    mkdir -p "$SCRIPT_DIR/data/$DIR"
    echo "   - data/$DIR created/exists"
done
echo ""

echo "[3/3] Starting backend server..."
echo ""
echo "Listening on: http://127.0.0.1:8000"
echo "Documentation: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
