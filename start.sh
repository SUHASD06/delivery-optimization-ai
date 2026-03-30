#!/bin/bash
# Start OpenEnv-compliant HTTP server (for hackathon automated evaluator)
python server.py &

# Start Gradio UI (for human judges)
python app.py
