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

# 6) PRE-CACHE the Hugging Face models (SentenceTransformer and Roberta detector)
#    This ensures the final container never tries to download at runtime.
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 6a) Cache the paraphrase-multilingual-MiniLM model
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 6b) Cache the Roberta‐based AI detector
AutoTokenizer.from_pretrained('roberta-base-openai-detector')
AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
print("✅ Model cache complete")
EOF

# 7) Copy the rest of your application code into /app
COPY . .

# 8) Expose port 8000 (where Uvicorn will serve FastAPI)
EXPOSE 8000

# 9) On container start, run Uvicorn to serve main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
