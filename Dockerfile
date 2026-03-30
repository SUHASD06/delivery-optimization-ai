FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

EXPOSE 7860
EXPOSE 8000

# Make start script executable
RUN chmod +x start.sh

# Launch both the OpenEnv API and the Gradio demo
CMD ["./start.sh"]