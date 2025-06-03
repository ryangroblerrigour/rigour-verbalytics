# Stage 1: install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install into a local folder
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# Stage 2: runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY main.py .
COPY routing_rules.json .

# Expose the port Render expects
ENV PORT 8080
EXPOSE 8080

# Environment variables (placeholder values; override these in Render)
ENV API_KEY="rigour-verbalytics-service"
ENV GOOGLE_BLOCKLIST_SHEET="RigourVerbalytics Blocklist"
ENV GOOGLE_LOGS_SHEET="RigourVerbalytics Logs"
ENV GOOGLE_CREDENTIALS_JSON="{}"

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
