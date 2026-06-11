#!/bin/bash
set -e

echo "📦 Installing Python ML dependencies..."
pip install --break-system-packages -r ml/requirements.txt

echo "🤖 Starting ML service on port 8000 in background..."
python -m uvicorn ml.main:app --host 0.0.0.0 --port 8000 &
ML_PID=$!
echo "ML service started with PID $ML_PID"

# Give ML service time to load and train the model
echo "⏳ Waiting for ML service to be ready..."
sleep 15

echo "🚀 Starting Node.js backend..."
node server.js
