#!/bin/bash
# Start the OpenEnv standard server in the background for the Hackathon's automated evaluator
openenv serve openenv.yaml --port 8000 --host 0.0.0.0 &

# Start the interactive Gradio UI for the human judges
python app.py
