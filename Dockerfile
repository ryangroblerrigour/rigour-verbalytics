# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (if you use any, add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy & install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy your code
COPY . .

# Optional: keep the sanity check, but make it less brittle
# (do NOT fail the build for missing optional libs; fail only if main/app is missing)
import sys, importlib
print("Python:", sys.version)
for name in ("fastapi","pydantic","requests","openai"):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "unknown")
        print(f"{name}:", v)
    except Exception as e:
        print(f"WARNING: couldn't import {name} -> {e!r}")
try:
    import main
    print("Imported main.py OK. Has app?", hasattr(main, "app"))
    assert hasattr(main, "app"), "main.py must expose FastAPI instance named `app`"
except Exception as e:
    import traceback; traceback.print_exc()
    raise SystemExit(1)
PY

# Render injects $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

