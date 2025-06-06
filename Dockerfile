# 1) Use a lightweight Python base image
FROM python:3.10-slim

# 2) Install system packages needed to build sentence-transformers, gspread, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

# 3) Set /app as the working directory inside the container
WORKDIR /app

# 4) Copy only requirements.txt first so Docker can cache this layer
COPY requirements.txt .

# 5) Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your application code into /app
COPY . .

# 7) Expose port 8000 (where Uvicorn will serve FastAPI)
EXPOSE 8000

# 8) On container start, run Uvicorn to serve main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
