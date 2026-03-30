FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ bash && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

EXPOSE 7860

# Make start script executable
RUN chmod +x start.sh

# Launch unified server (OpenEnv API + Gradio UI on port 7860)
CMD ["bash", "start.sh"]