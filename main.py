# main.py

import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List

import gspread
from google.oauth2.service_account import Credentials

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import openai

# -------------------------------
# 1) DEBUG: Print OpenAI key length at startup
# -------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "")
print(f"🔍 DEBUG: openai.api_key length is {len(openai.api_key)}")

# -------------------------------
# 2) BLOCKLIST LOADING (Google Sheet)
# -------------------------------

GOOGLE_BLOCKLIST_SHEET  = os.getenv("GOOGLE_BLOCKLIST_SHEET")  # e.g., "RigourVerbalytics Blocklist"
GOOGLE_LOGS_SHEET       = os.getenv("GOOGLE_LOGS_SHEET")      # e.g., "RigourVerbalytics Logs"
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

def load_blocklist_from_sheet() -> List[tuple]:
    """
    Read every row in the Google Sheet “RigourVerbalytics Blocklist”
    and return a list of (project_id, phrase) pairs. A blank project_id = global.
    """
    entries = []
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
            tmp.write(GOOGLE_CREDENTIALS_JSON)
            tmp.flush()
            creds = Credentials.from_service_account_file(tmp.name, scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ])
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_BLOCKLIST_SHEET).sheet1
        rows = sheet.get_all_records()
        for row in rows:
            pid = (row.get("project_id") or "").strip()
            phrase = (row.get("phrase") or "").strip().lower()
            if phrase:
                entries.append((pid, phrase))
    except Exception as e:
        print(f"⚠️  Could not load blocklist from sheet: {e}")
    return entries

def is_blocked(project_id: str, answer: str, blocklist: List[tuple]) -> bool:
    """
    Return True if 'answer' contains any phrase from the given blocklist.
    A phrase with empty project_id applies globally.
    """
    ans_low = answer.lower()
    for pid, phrase in blocklist:
        if phrase in ans_low and (pid == "" or pid.lower() == project_id.lower()):
            return True
    return False

# -------------------------------
# 3) LOGGING TO GOOGLE SHEETS
# -------------------------------

def get_sheets_client():
    """
    Return an authorized gspread client for both blocklist and logs.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        tmp.write(GOOGLE_CREDENTIALS_JSON)
        tmp.flush()
        creds = Credentials.from_service_account_file(tmp.name, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
    return gspread.authorize(creds)

try:
    sheets_client = get_sheets_client()
    log_sheet     = sheets_client.open(GOOGLE_LOGS_SHEET).sheet1
except Exception as e:
    print(f"⚠️ Could not open logs sheet: {e}")
    log_sheet = None

def log_to_sheet(row_values: List):
    """
    Append a row to the “RigourVerbalytics Logs” sheet (if available).
    """
    if log_sheet:
        try:
            log_sheet.append_row(row_values)
        except Exception as e:
            print(f"⚠️  Failed to append log row: {e}")

# -------------------------------
# 4) SCORING MODELS SETUP
# -------------------------------

semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
ai_tokenizer   = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
ai_model       = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# UK ↔ US spelling maps
UK_TO_US = {
    "colour": "color",
    "favourite": "favorite",
    "organise": "organize",
    "centre": "center",
    "behaviour": "behavior",
    "analyse": "analyze",
    "theatre": "theater",
    "travelling": "traveling",
    "labour": "labor"
}
US_TO_UK = {us: uk for uk, us in UK_TO_US.items()}

def detect_locale_mismatch(requested_lang: str, answer: str) -> bool:
    """
    Return True if the answer's spelling suggests the OTHER locale.
    """
    ans_low = answer.lower()
    if requested_lang.upper() == "UK":
        for us_spelling in US_TO_UK:
            if f" {us_spelling} " in f" {ans_low} ":
                return True
    if requested_lang.upper() == "USA":
        for uk_spelling in UK_TO_US:
            if f" {uk_spelling} " in f" {ans_low} ":
                return True
    return False

# -------------------------------
# 5) OPENAI SETUP FOR DYNAMIC PROBES
# -------------------------------

def make_snapback_text(question: str, answer: str) -> str:
    """
    Short snapback phrasing: ask them to elaborate on their poor answer.
    """
    return (
        f"Can you elaborate on what you mean by “{answer}” in the context of “{question}”?"
    )

def make_probe_text(question: str, answer: str) -> str:
    """
    A simple fallback probe if GPT fails or API key is missing.
    """
    snippet = answer.strip()
    return (
        f"Can you tell us more about what you mean by “{snippet}”—for instance, "
        f"which particular aspect made that stand out?"
    )

def get_dynamic_probe(question: str, answer: str) -> str:
    """
    Use a few‐shot prompt so GPT will:
      1) Pick out a short key phrase from the answer,
      2) Ask about that exact phrase with a request for example or detail,
      3) Not repeat the full original question verbatim,
      4) Sound conversational.
    Debug prints show whether we hit GPT or fallback.
    """
    # 1) Check for missing API key
    if not openai.api_key:
        print("🔴 get_dynamic_probe: No OPENAI_API_KEY set → using fallback")
        snippet = answer.strip()
        return make_probe_text(question, answer)

    # 2) We have a key—log that we're about to call
    print("🟢 get_dynamic_probe: Calling OpenAI ChatCompletion")

    few_shot = [
        {
            "role": "user",
            "content": (
                "QUESTION: \"What was your experience like using the app?\"\n"
                "ANSWER: \"It was straightforward and intuitive, let me complete tasks easily.\"\n"
                "Write a follow-up that:\n"
                "- Identifies one key phrase from the answer (e.g., “straightforward and intuitive”),\n"
                "- Asks for a specific example or detail about that phrase,\n"
                "- Does not repeat the entire original question,\n"
                "- Sounds like a moderator speaking naturally.\n"
                "Return only the follow-up."
            )
        },
        {
            "role": "assistant",
            "content": (
                "Which feature felt most intuitive, and can you give an example of a time it helped you complete a task quickly?"
            )
        },
        {
            "role": "user",
            "content": (
                "QUESTION: \"Here is a picture of a new brand of milk. What do you like about it?\"\n"
                "ANSWER: \"It looks clean and modern.\"\n"
                "Write a follow-up that:\n"
                "- Identifies one key phrase (e.g., “clean and modern”),\n"
                "- Asks for specifics or an example,\n"
                "- Does not simply restate the question,\n"
                "- Is conversational.\n"
                "Return only the follow-up."
            )
        },
        {
            "role": "assistant",
            "content": (
                "What about the packaging makes it feel clean and modern—can you describe a detail that stands out to you?"
            )
        }
    ]

    # Append our actual question/answer
    few_shot.append({
        "role": "user",
        "content": (
            f"QUESTION: \"{question}\"\n"
            f"ANSWER: \"{answer}\"\n"
            "Write a follow-up that:\n"
            "- Identifies one key phrase from the answer,\n"
            "- Asks for a specific example or deeper detail about that phrase,\n"
            "- Does not repeat the original question verbatim,\n"
            "- Is phrased conversationally.\n"
            "Return only the follow-up sentence."
        )
    })

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=few_shot,
            temperature=0.7,
            max_tokens=60
        )
        probe_text = resp.choices[0].message.content.strip()
        print("🟢 get_dynamic_probe: Received probe →", probe_text)
        return probe_text

    except Exception as e:
        print("🔴 get_dynamic_probe: ChatCompletion exception:", str(e))
        return make_probe_text(question, answer)

# -------------------------------
# 6) FASTAPI SETUP
# -------------------------------

API_KEY = os.getenv("API_KEY", "not-set")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/debug-openai")
async def debug_openai():
    """
    Returns whether OPENAI_API_KEY is set and its length, without revealing the key itself.
    """
    key = openai.api_key or ""
    return {
        "openai_key_present": bool(key),
        "key_length": len(key)
    }

# -------------------------------
# 7) Pydantic Schemas
# -------------------------------

class VerbalyticsRequest(BaseModel):
    respid: str
    project_id: str
    question_id: str
    question: str
    answer: str
    language: str               # "UK" or "USA"
    probe_required: Optional[str] = "no"   # "yes", "no", or "force"
    response_format: Optional[str] = "json"  # "json" or "csv"

class VerbalyticsResponse(BaseModel):
    respid: str
    project_id: str
    question_id: str
    quality_score: int
    ai_likelihood_score: int

    snapback_required: str   # "yes" or "no"
    snapback_text: Optional[str] = ""

    probe_required: str      # "yes", "no", or "force"
    probe_text: Optional[str] = ""

# -------------------------------
# 8) UPDATED SCORING FUNCTION
# -------------------------------

def get_quality_score(question: str, answer: str, lang: str) -> int:
    """
    Revised scoring:
      • Single-word color override (80)
      • If 3–6 words & sim > 0.10 → +0.3 length bonus
      • If > 5 words & sim > 0.15 → length‐boost (×1.5)
      • Otherwise blend 70% sim + 20% keyword + length_bonus
      • –20 for UK/US mismatch
    """
    try:
        clean_ans = answer.strip()
        if not clean_ans:
            return 1

        # 1) Embedding similarity
        embeddings = semantic_model.encode([question, answer])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # 2) Single-word “favorite color” override
        q_low = question.lower()
        a_low = answer.lower().strip()
        if len(answer.split()) == 1:
            if ("favorite color" in q_low or
                "favourite color" in q_low or
                "what color" in q_low or
                "which color" in q_low):
                return 80

        # 3) Compute base keyword overlap (strip punctuation)
        q_tokens = set(q_low.replace("?", "").replace(".", "").split())
        a_tokens = set(a_low.replace(".", "").split())
        overlap = q_tokens & a_tokens
        keyword_score = min(len(overlap) / max(1, len(q_tokens)), 1.0)

        # 4) If “app” in answer & “bank” in question, boost keyword_score to ≥ 0.2
        if "app" in a_tokens and "bank" in q_tokens:
            keyword_score = max(keyword_score, 0.2)

        # 5) Determine length-based bonuses
        length_bonus = 0.0
        num_words = len(answer.split())

        # Short-answer bonus for 3–6 words (if sim > 0.10)
        if 3 <= num_words <= 6 and sim > 0.10:
            length_bonus = 0.3

        # Larger boost for > 5 words & sim > 0.15
        boosted_sim = None
        if num_words > 5 and sim > 0.15:
            boosted_sim = min(sim * 1.5, 1.0)
        else:
            boosted_sim = None

        # 6) Compute final base_score
        if boosted_sim is not None:
            base_score = max(1, min(int(boosted_sim * 99) + 1, 100))
        else:
            combined = (0.7 * sim) + (0.2 * keyword_score) + length_bonus
            base_score = int(min(max(combined, 0.0), 1.0) * 99) + 1

        # 7) Locale mismatch penalty
        if detect_locale_mismatch(lang, answer):
            base_score = max(1, base_score - 20)

        return base_score

    except Exception as e:
        print(f"Error in get_quality_score: {e}")
        return 200

def get_ai_likelihood_score(answer: str) -> int:
    """
    Compute an AI-likelihood score (0–100) using a RoBERTa-based classifier.
    """
    try:
        inputs = ai_tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = ai_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        return max(0, min(int(probs[1] * 100), 100))
    except Exception as e:
        print(f"Error in get_ai_likelihood_score: {e}")
        return 200

# -------------------------------
# 9) API ENDPOINTS
# -------------------------------

@app.options("/check-verbalytics")
async def options_handler():
    return {}

@app.post("/check-verbalytics", response_model=VerbalyticsResponse)
async def check_verbalytics(
    payload: VerbalyticsRequest,
    x_api_key: Optional[str] = Header(None)
):
    # 1) Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2) Reload blocklist on every request
    current_blocklist = load_blocklist_from_sheet()

    # 3) Blocklist check
    if is_blocked(payload.project_id, payload.answer, current_blocklist):
        response_data = {
            "respid":              payload.respid,
            "project_id":          payload.project_id,
            "question_id":         payload.question_id,
            "quality_score":       0,
            "ai_likelihood_score": 0,
            "snapback_required":   "yes",
            "snapback_text":       "Your response contains disallowed content. Please re-answer.",
            "probe_required":      "no",
            "probe_text":          ""
        }
        log_to_sheet([
            datetime.utcnow().isoformat(),
            payload.respid,
            payload.project_id,
            payload.question_id,
            payload.question,
            payload.answer,
            payload.language,
            response_data["quality_score"],
            response_data["ai_likelihood_score"],
            response_data["snapback_required"],
            response_data["probe_required"],
            response_data["snapback_text"],
            response_data["probe_text"]
        ])
        return JSONResponse(content=response_data)

    # 4) Compute scores
    q_score  = get_quality_score(payload.question, payload.answer, payload.language)
    ai_score = get_ai_likelihood_score(payload.answer)

    # DEBUG: Log q_score and incoming_probe
    incoming_probe = (payload.probe_required or "no").lower()
    print(f"🔍 DEBUG check_verbalytics: q_score={q_score}, incoming_probe='{incoming_probe}'")

    # 5) Determine snapback vs probe
    if q_score <= 20 and incoming_probe != "force":
        print("🔍 DEBUG check_verbalytics: Taking SNAPBACK path")
        snapback_required = "yes"
        snap_text = make_snapback_text(payload.question, payload.answer)

        probe_required = "no"
        probe_text = ""
    else:
        print("🔍 DEBUG check_verbalytics: Taking PROBE path (or force)")
        snapback_required = "no"
        snap_text = ""

        if incoming_probe == "force":
            print("🔍 DEBUG check_verbalytics: INCOMING_PROBE == 'force' → calling get_dynamic_probe")
            probe_required = "yes"
            probe_text = get_dynamic_probe(payload.question, payload.answer)
        elif incoming_probe == "yes" and q_score > 20:
            print("🔍 DEBUG check_verbalytics: incoming_probe == 'yes' && q_score>20 → calling get_dynamic_probe")
            probe_required = "yes"
            probe_text = get_dynamic_probe(payload.question, payload.answer)
        else:
            print("🔍 DEBUG check_verbalytics: No probe or force → no probe_text")
            probe_required = "no"
            probe_text = ""

    # 6) Assemble response object
    response_data = {
        "respid":               payload.respid,
        "project_id":           payload.project_id,
        "question_id":          payload.question_id,
        "quality_score":        q_score,
        "ai_likelihood_score":  ai_score,
        "snapback_required":    snapback_required,
        "snapback_text":        snap_text,
        "probe_required":       probe_required,
        "probe_text":           probe_text
    }

    # 7) Log to Google Sheets
    log_to_sheet([
        datetime.utcnow().isoformat(),
        payload.respid,
        payload.project_id,
        payload.question_id,
        payload.question,
        payload.answer,
        payload.language,
        response_data["quality_score"],
        response_data["ai_likelihood_score"],
        response_data["snapback_required"],
        response_data["probe_required"],
        response_data["snapback_text"],
        response_data["probe_text"]
    ])

    # 8) Return in requested format
    if payload.response_format.lower() == "csv":
        text_field = response_data["snapback_text"] or response_data["probe_text"]
        csv_line = ",".join([
            str(response_data["quality_score"]),
            str(response_data["ai_likelihood_score"]),
            response_data["snapback_required"],
            text_field.replace(",", ";")
        ])
        return PlainTextResponse(csv_line)
    else:
        return JSONResponse(content=response_data)

# -------------------------------
# 10) Local Uvicorn Runner
# -------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
