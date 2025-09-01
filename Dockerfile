FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# (Optional) system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ This block forces the real error to show up at build time
RUN python - <<'PY'
import sys
print("Python:", sys.version)
import fastapi, pydantic, openai, requests
print("Deps OK:", fastapi.__version__, pydantic.__version__, openai.__version__, requests.__version__)
import main
print("✅ Imported main.py successfully")
PY

COPY . .

ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
