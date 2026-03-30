#!/bin/bash
# Start the unified server using uvicorn
# This loads app.py at module level, registering both API routes and Gradio UI
uvicorn app:app --host 0.0.0.0 --port 7860
