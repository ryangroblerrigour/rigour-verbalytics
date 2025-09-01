FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ Add this block to surface the TRUE error at build time
COPY main.py /app/main.py
RUN python - <<'PY'
import sys
print("Python:", sys.version)
import fastapi, pydantic, openai, requests
print("Deps:", fastapi.__version__, pydantic.__version__, openai.__version__, requests.__version__)
try:
    import main
    print("Imported main.py OK. Has app?", hasattr(main, "app"))
except Exception as e:
    import traceback; traceback.print_exc()
    raise SystemExit(1)
PY

# Now copy the rest of your code
COPY . .

ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
