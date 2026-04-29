"""
Verbalytics — Fast, Reliable API for Library Check, Scoring, Acknowledgement, and Follow-up
----------------------------------------------------------------------------------------
- FastAPI with asyncio orchestration
- Parallel generation (ack + snapback + follow-up)
- Deterministic scoring (no LLM needed for score)
- Streaming JSONL endpoint for low perceived latency
- Uses OpenAI Python SDK >= 1.0 (client.chat.completions.create)

ENV VARS:
  - OPENAI_API_KEY
  - MODEL_ACK (default: gpt-4o-mini)
  - MODEL_SNAPBACK (default: gpt-4o-mini)
  - MODEL_FOLLOWUP (default: gpt-4o-mini)
  - VERBALYTICS_MAX_TOKENS (default: 200)
  - OPENAI_TIMEOUT (default: 12.0)
  - GOOGLE_BLOCKLIST_SHEET (publish-to-web CSV url)
  - VERBALYTICS_BLOCKLIST_REFRESH_SECONDS (default: 300)
  - VERBALYTICS_BLOCK_INPUT_MODE (off | flag | reject)  # default: flag

Run locally:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import re
import time
import asyncio
import csv
import json
import requests
import unicodedata
from typing import Optional, Literal, Dict, Any
from time import monotonic
from threading import RLock

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_MODEL_ACK = os.getenv("MODEL_ACK", "gpt-4o-mini")
DEFAULT_MODEL_SNAPBACK = os.getenv("MODEL_SNAPBACK", "gpt-4o-mini")
DEFAULT_MODEL_FOLLOWUP = os.getenv("MODEL_FOLLOWUP", "gpt-4o-mini")

MAX_TOKENS = int(os.getenv("VERBALYTICS_MAX_TOKENS", "200"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "12.0"))

BLOCKLIST_CSV_URL = os.getenv("GOOGLE_BLOCKLIST_SHEET", "")
BLOCKLIST_REFRESH_SECONDS = int(os.getenv("VERBALYTICS_BLOCKLIST_REFRESH_SECONDS", "300"))
INPUT_BLOCK_MODE = os.getenv("VERBALYTICS_BLOCK_INPUT_MODE", "flag").lower()


# ---------------------------------------------------------------------
# Locale / language helpers
# ---------------------------------------------------------------------
def _clean_text(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s


def _norm(s: str) -> str:
    return _clean_text(s).lower()


def _normalize_for_matching(s: Optional[str]) -> str:
    s = _norm(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_greek_text(s: Optional[str]) -> bool:
    text = _clean_text(s)
    if not text:
        return False
    greek_chars = sum(1 for ch in text if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF")
    letters = sum(1 for ch in text if ch.isalpha())
    return greek_chars >= 2 and (greek_chars / max(letters, 1)) >= 0.25


def _language_bucket(locale: Optional[str], question: Optional[str] = None, response: Optional[str] = None) -> str:
    loc = (locale or "").strip().lower().replace("_", "-")
    if loc.startswith("el"):
        return "el"
    if loc.startswith("nl"):
        return "nl"
    if _is_greek_text(question) or _is_greek_text(response):
        return "el"
    return "en"


def _locale_instruction(locale: Optional[str]) -> str:
    loc = (locale or "en").strip().lower().replace("_", "-")

    if loc.startswith("ar"):
        return "Respond in Arabic only."
    if loc.startswith("de"):
        return "Respond in German only."
    if loc.startswith("fr"):
        return "Respond in French only."
    if loc.startswith("es"):
        return "Respond in Spanish only."
    if loc.startswith("it"):
        return "Respond in Italian only."
    if loc.startswith("tr"):
        return "Respond in Turkish only."
    if loc.startswith("zh"):
        return "Respond in Chinese only."
    if loc.startswith("pl"):
        return "Respond in Polish only."
    if loc.startswith("ko"):
        return "Respond in Korean only."
    if loc.startswith("el"):
        return "Respond in Greek only."
    if loc.startswith("nl"):
        return "Respond in Dutch only."
    return "Respond in English only."


# ---------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai>=1.0.0")
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
TaskName = Literal["check", "score", "ack", "followup"]


class VerbalyticsInput(BaseModel):
    question: str
    response: str
    tasks: list[TaskName] = Field(default_factory=lambda: ["score", "ack", "followup"])
    locale: Optional[str] = "en"
    context: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None


class VerbalyticsOutput(BaseModel):
    subscores: Optional[Dict[str, int]] = None
    score: Optional[int] = None
    ai_likelihood: Optional[int] = None
    ack: Optional[str] = None
    snapback: Optional[str] = None
    followup: Optional[str] = None
    input_blocked_phrases: Optional[list[str]] = None
    latency_ms: Optional[int] = None


# ---------------------------------------------------------------------
# Blocklist Loader
# ---------------------------------------------------------------------
_block_lock = RLock()
_block_global: set[str] = set()
_block_projects: dict[str, set[str]] = {}
_block_loaded_at: float = 0.0
_block_version: int = 0


def _load_blocklist_csv(url: str) -> tuple[set[str], dict[str, set[str]]]:
    g, p = set(), {}
    if not url:
        return g, p
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        reader = csv.DictReader(resp.content.decode("utf-8", errors="ignore").splitlines())
        for row in reader:
            scope = _norm(row.get("scope", ""))
            proj = _norm(row.get("project", ""))
            phrase = _normalize_for_matching(row.get("phrase", ""))
            if not phrase:
                continue
            if scope == "global" or scope == "":
                g.add(phrase)
            elif scope == "project" and proj:
                p.setdefault(proj, set()).add(phrase)
    except Exception:
        pass
    return g, p


def _refresh_blocklist(force: bool = False) -> None:
    global _block_global, _block_projects, _block_loaded_at, _block_version
    now = monotonic()
    with _block_lock:
        if not force and (now - _block_loaded_at) < BLOCKLIST_REFRESH_SECONDS:
            return
        g, p = _load_blocklist_csv(BLOCKLIST_CSV_URL)
        if g or p or force:
            _block_global = g
            _block_projects = p
            _block_loaded_at = now
            _block_version += 1


def get_block_phrases(project_id: Optional[str]) -> set[str]:
    _refresh_blocklist(False)
    with _block_lock:
        phrases = set(_block_global)
        if project_id:
            phrases |= _block_projects.get(_norm(project_id), set())
        return phrases


def contains_blocked_phrase(text: str, project_id: Optional[str]) -> bool:
    if not text:
        return False
    lt = _normalize_for_matching(text)
    return any(p in lt for p in get_block_phrases(project_id))


def find_blocked_phrases(text: str, project_id: Optional[str]) -> list[str]:
    if not text:
        return []
    lt = _normalize_for_matching(text)
    phrases = get_block_phrases(project_id)
    hits = {p for p in phrases if p and p in lt}
    return sorted(list(hits))


# ---------------------------------------------------------------------
# Language packs
# ---------------------------------------------------------------------
LANG_RULES = {
    "en": {
        "empty_values": {"", "na", "n/a", "none", "no idea", "idk", "?"},
        "reason_markers": {"because", "since", "due", "therefore", "so"},
        "emphasis_markers": {"especially", "particularly"},
        "first_hand_markers": {
            "i use", "i used", "i tried", "i've used", "i have used", "i bought", "we bought",
            "my kids use", "my kid uses", "my children", "my family", "we use", "i saw", "i heard", "i've seen"
        },
        "usage_context": {
            "at home", "at work", "for work", "on the train", "on the commute", "in the car",
            "at night", "in the morning", "on weekends", "after school", "during football", "during ads",
            "when cooking", "when cleaning", "while driving", "before bed"
        },
        "feature_sensory": {
            "logo", "pack", "packaging", "design", "music", "jingle", "voiceover", "scene", "actor", "character", "dog",
            "taste", "smell", "texture", "colour", "color", "price", "offer", "discount", "durable", "battery", "speed",
            "instructions", "interface", "app", "sound", "volume", "quality", "resolution", "camera", "label", "slogan"
        },
        "example_markers": {"for example", "such as", "like when", "like the time", "e.g.", "especially", "particularly"},
        "opinion_q": {"think", "opinion", "feel", "like", "dislike", "impression", "favourite", "favorite", "rate"},
        "subjective_resp": {
            "good", "bad", "great", "love", "hate", "nice", "funny", "boring", "memorable", "confusing", "clear",
            "useful", "annoying", "enjoy", "enjoyed", "liked", "disliked", "amazing", "awful", "meh"
        },
        "refers_to_item": {"ad", "advert", "advertisement", "commercial", "spot", "it", "this"},
        "fillers": {"like", "basically", "sort of", "kind of", "you know", "stuff", "things"},
        "templ": [
            "as an ai", "as a language model", "overall,", "moreover,", "furthermore,",
            "in summary", "in conclusion", "additionally,", "importantly,"
        ],
        "invalid_short": {"ok", "yes", "no", "maybe", "idk", "don't know", "dont know", "not sure", "nothing"},
        "fallback_short_followup": "Could you share a bit more detail about that?",
        "fallback_followup": "What specifically makes you say that?",
        "repair_numeric": "Could you explain what you mean by '{value}' in relation to: {question}",
        "repair_general": "Could you expand a bit more on that in relation to: {question}",
        "deepdive_questions": {
            "low_score": {
                "ask_why_that_wording": "What is it about that wording or phrase that doesn’t work for you?",
                "ask_which_words": "Which words or phrases made it feel that way to you?",
                "ask_specific_words": "What specifically about the tone or wording didn’t work for you?",
                "ask_which_visuals": "What specifically in the visuals or design made it feel that way?",
                "ask_specific_visuals": "What specifically about the visuals or design didn’t work for you?",
                "ask_meaning": "What about the message or meaning didn’t feel right to you?",
                "ask_specifics": "What specifically about it didn’t work for you?",
            },
            "mid_score": {
                "explore_both_sides_words": "Which parts of the wording worked for you, and which didn’t?",
                "explore_both_sides_visuals": "What about the visuals or design worked for you, and what didn’t?",
                "explore_both_sides": "What worked for you, and what didn’t?",
            },
            "high_score": {
                "ask_why_that_wording": "What is it about that wording or phrase that resonated with you?",
                "ask_specific_words": "What specifically about the wording or tone resonated with you?",
                "ask_specific_visuals": "What specifically about the visuals or design resonated with you?",
                "ask_meaning": "What about the message or meaning connected with you?",
                "ask_specifics": "What specifically made it resonate with you?",
            },
            "neutral": {
                "ask_which_words": "Which words or phrases made you feel that way?",
                "ask_specific_words": "What specifically about the wording or tone stood out to you?",
                "ask_specific_visuals": "What specifically about the visuals or design stood out to you?",
                "ask_meaning": "What about the message or meaning stood out to you?",
                "clarify": "What specifically do you mean by that?",
            }
        },
        "deepdive_eval": {
            "criticism_words": [
                "generic", "bland", "unclear", "confusing", "overused",
                "flat", "vague", "weak", "boring", "too much", "too little"
            ],
            "mentions_language": ["tone", "word", "words", "phrase", "phrases", "wording", "language", "message", "line"],
            "mentions_visuals": ["colour", "color", "visual", "visuals", "design", "layout", "look", "font", "image", "images"],
            "mentions_meaning": ["meaning", "means", "suggests", "implies", "comes across", "feels", "felt"],
            "mentions_relevance": ["for me", "to me", "personally", "relevant", "connect", "doesn't connect", "doesnt connect"],
        },
    },
    "el": {
        "empty_values": {"", "δκ", "δ/ξ", "δεν ξερω", "δεν ξέρω", "κανενα", "κανένα", "?"},
        "reason_markers": {"γιατί", "επειδή", "λόγω", "διοτι", "διότι", "οπότε", "άρα"},
        "emphasis_markers": {"ειδικά", "ιδιαίτερα", "κυρίως"},
        "first_hand_markers": {
            "το χρησιμοποιώ", "το χρησιμοποίησα", "το έχω χρησιμοποιήσει", "το εχω χρησιμοποιησει",
            "το αγόρασα", "το αγορασα", "το είδα", "το ειδα", "το άκουσα", "το ακουσα",
            "στην οικογένειά μου", "στην οικογενεια μου", "στο σπίτι", "στο σπιτι", "το χρησιμοποιούμε", "το χρησιμοποιουμε"
        },
        "usage_context": {
            "στο σπίτι", "στο σπιτι", "στη δουλειά", "στη δουλεια", "για τη δουλειά", "για τη δουλεια",
            "στο αυτοκίνητο", "στο αυτοκινητο", "το βράδυ", "το βραδυ", "το πρωί", "το πρωι",
            "τα σαββατοκύριακα", "τα σαββατοκυριακα", "πριν τον ύπνο", "πριν τον υπνο"
        },
        "feature_sensory": {
            "λογότυπο", "λογοτυπο", "συσκευασία", "σχεδιασμός", "σχεδιασμος", "μουσική", "μουσικη",
            "φωνή", "φωνη", "σκηνή", "σκηνη", "χαρακτήρας", "χαρακτηρας", "γεύση", "γευση",
            "μυρωδιά", "μυρωδια", "υφή", "υφη", "χρώμα", "χρωμα", "τιμή", "τιμη", "προσφορά",
            "έκπτωση", "εκπτωση", "μπαταρία", "μπαταρια", "ταχύτητα", "ταχυτητα",
            "οδηγίες", "οδηγιες", "διεπαφή", "διεπαφη", "εφαρμογή", "εφαρμογη", "ήχος", "ηχος",
            "ποιότητα", "ποιοτητα", "κάμερα", "καμερα", "σλόγκαν", "συνθημα"
        },
        "example_markers": {"για παράδειγμα", "για παραδειγμα", "όπως όταν", "οπως οταν", "ειδικά", "ιδιαίτερα"},
        "opinion_q": {"πιστεύεις", "πιστευεις", "γνώμη", "γνωμη", "νιώθεις", "νιωθεις", "σου αρέσει", "σου αρεσει", "βαθμολογείς", "βαθμολογεις"},
        "subjective_resp": {
            "καλό", "καλο", "κακή", "κακη", "κακό", "κακο", "πολύ καλό", "τέλειο", "τελειο", "μου αρέσει",
            "μου αρεσει", "δεν μου αρέσει", "δεν μου αρεσει", "βαρετό", "βαρετο", "ξεκάθαρο", "ξεκαθαρο",
            "χρήσιμο", "χρησιμο", "ενοχλητικό", "ενοχλητικο", "δυνατό", "αδύναμο", "αδυναμο",
            "συγχυτικό", "συγχυτικο", "ελκυστικό", "ελκυστικο"
        },
        "refers_to_item": {"διαφήμιση", "διαφημιση", "σποτ", "αυτό", "αυτο", "αυτό το", "αυτο το"},
        "fillers": {"δηλαδή", "δηλαδη", "κάπως", "καπως", "ξέρεις", "ξερεις", "πράγματα", "πραγματα"},
        "templ": [
            "ως ai", "ως τεχνητή νοημοσύνη", "ως τεχνητη νοημοσυνη",
            "συνολικά", "συνολικα", "επιπλέον", "επιπλεον", "συμπερασματικά", "συμπερασματικα"
        ],
        "invalid_short": {"οκ", "ναι", "όχι", "οχι", "ίσως", "ισως", "δκ", "δεν ξέρω", "δεν ξερω", "τίποτα", "τιποτα"},
        "fallback_short_followup": "Θα μπορούσατε να δώσετε λίγο περισσότερη λεπτομέρεια;",
        "fallback_followup": "Τι ακριβώς σας κάνει να το λέτε αυτό;",
        "repair_numeric": "Θα μπορούσατε να εξηγήσετε τι εννοείτε με το «{value}» σε σχέση με: {question}",
        "repair_general": "Θα μπορούσατε να το εξηγήσετε λίγο περισσότερο σε σχέση με: {question}",
        "deepdive_questions": {
            "low_score": {
                "ask_why_that_wording": "Τι είναι αυτό στη διατύπωση ή στη φράση που δεν σας λειτουργεί;",
                "ask_which_words": "Ποιες λέξεις ή φράσεις σας έκαναν να το νιώσετε αυτό;",
                "ask_specific_words": "Τι συγκεκριμένα στον τόνο ή στη διατύπωση δεν σας λειτούργησε;",
                "ask_which_visuals": "Τι συγκεκριμένα στα οπτικά ή στο design σας έκανε να το νιώσετε αυτό;",
                "ask_specific_visuals": "Τι συγκεκριμένα στα οπτικά ή στο design δεν σας λειτούργησε;",
                "ask_meaning": "Τι στο μήνυμα ή στο νόημα δεν σας φάνηκε σωστό;",
                "ask_specifics": "Τι συγκεκριμένα δεν λειτούργησε για εσάς;",
            },
            "mid_score": {
                "explore_both_sides_words": "Ποια σημεία της διατύπωσης λειτούργησαν για εσάς και ποια όχι;",
                "explore_both_sides_visuals": "Τι στα οπτικά ή στο design λειτούργησε για εσάς και τι όχι;",
                "explore_both_sides": "Τι λειτούργησε για εσάς και τι όχι;",
            },
            "high_score": {
                "ask_why_that_wording": "Τι είναι αυτό στη διατύπωση ή στη φράση που σας έκανε θετική εντύπωση;",
                "ask_specific_words": "Τι συγκεκριμένα στη διατύπωση ή στον τόνο σας άρεσε;",
                "ask_specific_visuals": "Τι συγκεκριμένα στα οπτικά ή στο design σας άρεσε;",
                "ask_meaning": "Τι στο μήνυμα ή στο νόημα συνδέθηκε με εσάς;",
                "ask_specifics": "Τι συγκεκριμένα το έκανε να σας ταιριάζει;",
            },
            "neutral": {
                "ask_which_words": "Ποιες λέξεις ή φράσεις σας έκαναν να νιώσετε έτσι;",
                "ask_specific_words": "Τι συγκεκριμένα στη διατύπωση ή στον τόνο ξεχώρισε για εσάς;",
                "ask_specific_visuals": "Τι συγκεκριμένα στα οπτικά ή στο design ξεχώρισε για εσάς;",
                "ask_meaning": "Τι στο μήνυμα ή στο νόημα ξεχώρισε για εσάς;",
                "clarify": "Τι ακριβώς εννοείτε με αυτό;",
            }
        },
        "deepdive_eval": {
            "criticism_words": [
                "γενικό", "γενικη", "γενικόλογο", "αόριστο", "αοριστο", "μπερδεμένο", "μπερδεμενο",
                "αδύναμο", "αδυναμο", "βαρετό", "βαρετο", "πολύ", "λίγο", "λιγο"
            ],
            "mentions_language": [
                "τόνος", "τονος", "λέξη", "λεξη", "λέξεις", "λεξεις",
                "φράση", "φραση", "φράσεις", "φρασεις", "διατύπωση", "διατυπωση", "γλώσσα", "γλωσσα", "μήνυμα", "μηνυμα"
            ],
            "mentions_visuals": [
                "χρώμα", "χρωμα", "οπτικό", "οπτικο", "οπτικά", "οπτικα", "σχεδιασμός", "σχεδιασμος",
                "διάταξη", "διαταξη", "γραμματοσειρά", "γραμματοσειρα", "εικόνα", "εικονα", "εικόνες", "εικονες"
            ],
            "mentions_meaning": [
                "νόημα", "νοημα", "σημαίνει", "σημαινει", "υπονοεί", "υπονοει",
                "δείχνει", "δειχνει", "ακούγεται", "ακουγεται", "νιώθω", "νιωθω", "ένιωσα", "ενιωσα"
            ],
            "mentions_relevance": [
                "για μένα", "για μενα", "σε μένα", "σε μενα", "προσωπικά", "προσωπικα",
                "σχετικό", "σχετικο", "με συνδέει", "δεν με συνδέει", "δεν συνδέεται", "δεν συνδεεται"
            ],
        },
    },
    "nl": {
        "empty_values": {"", "nvt", "n.v.t", "geen idee", "weet ik niet", "?", "niks"},
        "reason_markers": {"omdat", "want", "doordat", "daardoor", "dus"},
        "emphasis_markers": {"vooral", "met name", "eigenlijk"},
        "first_hand_markers": {
            "ik gebruik", "ik gebruikte", "ik heb gebruikt", "ik heb het gebruikt",
            "ik kocht", "ik heb gekocht", "ik zag", "ik heb gezien", "ik hoorde",
            "wij gebruiken", "mijn familie", "mijn kinderen", "thuis gebruiken we"
        },
        "usage_context": {
            "thuis", "op het werk", "voor werk", "in de trein", "in de auto",
            "s avonds", "in de ochtend", "in het weekend", "voor het slapengaan"
        },
        "feature_sensory": {
            "logo", "verpakking", "ontwerp", "design", "muziek", "stem", "scene",
            "personage", "smaak", "geur", "textuur", "kleur", "prijs", "aanbieding",
            "korting", "duurzaam", "batterij", "snelheid", "instructies", "interface",
            "app", "geluid", "volume", "kwaliteit", "resolutie", "camera", "label", "slogan"
        },
        "example_markers": {"bijvoorbeeld", "zoals", "vooral", "met name"},
        "opinion_q": {"denken", "vind", "vindt", "gevoel", "mening", "leuk", "niet leuk", "indruk", "favoriet", "beoordelen"},
        "subjective_resp": {
            "goed", "slecht", "geweldig", "mooi", "saai", "duidelijk", "nuttig",
            "irritant", "verwarrend", "leuk", "niet leuk", "ik vind het goed",
            "ik vind het slecht", "sterk", "zwak", "aantrekkelijk", "onaantrekkelijk"
        },
        "refers_to_item": {"advertentie", "reclame", "spot", "dit", "het", "deze"},
        "fillers": {"zeg maar", "een beetje", "soort van", "dingen", "enzo"},
        "templ": [
            "als ai", "als taalmodel", "over het algemeen", "bovendien",
            "samenvattend", "concluderend", "daarnaast", "belangrijk is"
        ],
        "invalid_short": {"ok", "ja", "nee", "misschien", "geen idee", "weet ik niet", "niks"},
        "fallback_short_followup": "Kunt u daar iets meer detail over geven?",
        "fallback_followup": "Wat maakt dat u dat zegt?",
        "repair_numeric": "Kunt u uitleggen wat u bedoelt met '{value}' in relatie tot: {question}",
        "repair_general": "Kunt u daar iets verder op ingaan in relatie tot: {question}",
        "deepdive_questions": {
            "low_score": {
                "ask_why_that_wording": "Wat is er aan die formulering of zin dat voor u niet werkt?",
                "ask_which_words": "Welke woorden of zinnen gaven u dat gevoel?",
                "ask_specific_words": "Wat werkte er precies niet aan de toon of formulering?",
                "ask_which_visuals": "Wat in de visuals of het design gaf u precies dat gevoel?",
                "ask_specific_visuals": "Wat werkte er precies niet aan de visuals of het design?",
                "ask_meaning": "Wat voelde er niet goed aan in de boodschap of betekenis?",
                "ask_specifics": "Wat werkte er precies niet voor u?",
            },
            "mid_score": {
                "explore_both_sides_words": "Welke delen van de formulering werkten wel voor u en welke niet?",
                "explore_both_sides_visuals": "Wat werkte er aan de visuals of het design wel voor u en wat niet?",
                "explore_both_sides": "Wat werkte er voor u wel en wat niet?",
            },
            "high_score": {
                "ask_why_that_wording": "Wat is het aan die formulering of zin dat bij u aansloeg?",
                "ask_specific_words": "Wat sprak u precies aan in de formulering of toon?",
                "ask_specific_visuals": "Wat sprak u precies aan in de visuals of het design?",
                "ask_meaning": "Wat in de boodschap of betekenis sloot bij u aan?",
                "ask_specifics": "Wat maakte dat het voor u werkte?",
            },
            "neutral": {
                "ask_which_words": "Welke woorden of zinnen gaven u dat gevoel?",
                "ask_specific_words": "Wat viel u precies op aan de formulering of toon?",
                "ask_specific_visuals": "Wat viel u precies op aan de visuals of het design?",
                "ask_meaning": "Wat viel u op aan de boodschap of betekenis?",
                "clarify": "Wat bedoelt u daar precies mee?",
            }
        },
        "deepdive_eval": {
            "criticism_words": [
                "algemeen", "vaag", "onduidelijk", "verwarrend", "saai",
                "zwak", "te veel", "te weinig", "standaard", "flauw"
            ],
            "mentions_language": [
                "toon", "woord", "woorden", "zin", "zinnen",
                "formulering", "taal", "boodschap", "tekst"
            ],
            "mentions_visuals": [
                "kleur", "visueel", "visuals", "design", "ontwerp",
                "layout", "lettertype", "afbeelding", "afbeeldingen"
            ],
            "mentions_meaning": [
                "betekenis", "betekent", "suggereert", "impliceert",
                "komt over", "voelt", "gevoel"
            ],
            "mentions_relevance": [
                "voor mij", "persoonlijk", "relevant", "past bij mij",
                "sluit aan", "sluit niet aan", "verbinding"
            ],
        },
    },
}


# ---------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------
class ScoreEngine:
    WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

    def _tokens(self, s: str) -> list[str]:
        text = _normalize_for_matching(s)
        return self.WORD_RE.findall(text)

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _has_any_phrase(self, hay: str, phrases: set[str]) -> bool:
        return any(p in hay for p in phrases)

    def subscores(self, question: str, response: str, locale: Optional[str] = None) -> dict:
        q = _clean_text(question)
        r = _clean_text(response)
        lang = _language_bucket(locale, q, r)
        rules = LANG_RULES[lang]

        r_norm = _normalize_for_matching(r)
        q_norm = _normalize_for_matching(q)

        if not r_norm or r_norm in rules["empty_values"]:
            return {"specificity": 0, "concreteness": 0, "relevance": 0, "clarity": 0}

        q_toks = set(self._tokens(q_norm))
        r_toks = self._tokens(r_norm)
        r_set = set(r_toks)
        wc = len(r_toks)

        has_reason = any(k in r_set for k in rules["reason_markers"]) or self._has_any_phrase(r_norm, set(rules["reason_markers"]))
        has_emphasis = any(k in r_set for k in rules["emphasis_markers"]) or self._has_any_phrase(r_norm, set(rules["emphasis_markers"]))
        unique_terms = len([t for t in r_set if t not in q_toks and len(t) > 3])

        spec_base = 55 * (1 - (2.71828 ** (-wc / 16)))
        spec_bonus = (12 if has_reason else 0) + (8 if has_emphasis else 0) + min(22, unique_terms)
        specificity = max(0, min(100, int(spec_base + spec_bonus)))

        first_hand = self._has_any_phrase(r_norm, rules["first_hand_markers"])
        context_hit = self._has_any_phrase(r_norm, rules["usage_context"])
        example_hit = self._has_any_phrase(r_norm, rules["example_markers"])
        feature_hits = sum(1 for f in rules["feature_sensory"] if f in r_set or f in r_norm)

        properish = sum(1 for w in response.split() if (w[:1].isupper() and len(w) > 3))
        length_credit = min(28, wc // 5)

        conc_score = (
            (22 if first_hand else 0) +
            (14 if context_hit else 0) +
            (12 if example_hit else 0) +
            min(30, feature_hits * 7) +
            min(10, properish * 2) +
            length_credit
        )
        concreteness = max(0, min(100, int(conc_score)))

        rel_overlap = 100 * self._jaccard(q_toks, r_set)
        opinion_q = any(k in q_toks for k in rules["opinion_q"]) or self._has_any_phrase(q_norm, set(rules["opinion_q"]))
        subjective_resp = any(k in r_set for k in rules["subjective_resp"]) or self._has_any_phrase(r_norm, set(rules["subjective_resp"]))
        refers_to_item = any(k in r_set for k in rules["refers_to_item"]) or self._has_any_phrase(r_norm, set(rules["refers_to_item"]))

        rel_heur = 0
        if opinion_q and subjective_resp:
            rel_heur += 65
        if refers_to_item:
            rel_heur += 15
        relevance = max(0, min(100, int(max(rel_overlap, rel_heur))))

        sentences = [s for s in re.split(r"[.!;;;\n]+", r) if s.strip()]
        avg_len = (sum(len(self._tokens(s)) for s in sentences) / len(sentences)) if sentences else wc
        too_long = avg_len > 28
        too_short = avg_len < 3
        hedge = any(f in r_norm for f in rules["fillers"])

        clarity_base = 82
        clarity_pen = (14 if too_long else 0) + (14 if too_short else 0) + (8 if hedge else 0)
        clarity = max(0, min(100, int(clarity_base - clarity_pen)))

        return {
            "specificity": specificity,
            "concreteness": concreteness,
            "relevance": relevance,
            "clarity": clarity,
        }

    def quality_score(self, q: str, r: str, locale: Optional[str] = None) -> int:
        ss = self.subscores(q, r, locale=locale)
        avg = (ss["specificity"] + ss["concreteness"] + ss["relevance"] + ss["clarity"]) / 4
        return max(0, min(100, int(round(avg / 10) * 10)))

    def ai_likelihood(self, r: str, locale: Optional[str] = None) -> int:
        if not r:
            return 0

        lang = _language_bucket(locale, response=r)
        rules = LANG_RULES[lang]

        toks = self._tokens(r)
        wc = len(toks)
        r_norm = _normalize_for_matching(r)

        base = 0
        if wc >= 8:
            base = 5
        if wc >= 20:
            base = 10

        templ_hits = sum(1 for t in rules["templ"] if t in r_norm)

        sents = [s.strip() for s in re.split(r"[.!;;;\n]+", r_norm) if s.strip()]
        lengths = [len(self._tokens(s)) for s in sents] or [wc]
        var = max(lengths) - min(lengths)
        low_var = 12 if (len(lengths) >= 3 and var <= 4) else 0

        uniq_ratio = len(set(toks)) / (wc or 1)
        rep = 14 if uniq_ratio < 0.48 else 0

        verbose = 16 if wc > 180 else 0
        very_verbose = 18 if wc > 350 else 0

        score = base + templ_hits * 22 + low_var + rep + verbose + very_verbose
        return int(max(0, min(100, score)))


score_engine = ScoreEngine()


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
SYSTEM_ACK = (
    "You are a concise, friendly research assistant. "
    "Respond with a 3–10 word acknowledgement that mirrors the respondent's sentiment. "
    "Return only the acknowledgement."
)

SYSTEM_SNAPBACK = (
    "You are a professional market research moderator. "
    "If the respondent's answer is unclear, nonsense, or irrelevant, politely re-ask the original question, "
    "making clear the first answer did not address it. Keep a neutral tone. Return only the re-asked question."
)

SYSTEM_FOLLOWUP = (
    "You are a professional market research moderator. "
    "Generate ONE open-ended follow-up question (max 22 words). "
    "It must be a single sentence, >=6 words, end with '?' and be neutral. "
    "Return only the question."
)


# ---------------------------------------------------------------------
# Generators (OpenAI)
# ---------------------------------------------------------------------
async def _call_openai_chat(messages: list[dict], model: str, max_tokens: int) -> str:
    client = get_client()
    loop = asyncio.get_running_loop()

    def _block_call():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=["\n"],
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _block_call), timeout=OPENAI_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Model '{model}' timed out")


async def generate_ack(q: str, r: str, project_id: Optional[str], locale: Optional[str], model: str = DEFAULT_MODEL_ACK) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_ACK + " " + _locale_instruction(locale)},
        {"role": "user", "content": f"Q: {q.strip()} A: {r.strip()}"},
    ]
    text = await _call_openai_chat(messages, model, max_tokens=min(16, MAX_TOKENS))
    ack = text.split("\n")[0][:160]

    if contains_blocked_phrase(ack, project_id):
        avoid = ", ".join(sorted(get_block_phrases(project_id)))
        messages[0]["content"] = SYSTEM_ACK + " " + _locale_instruction(locale) + (f" Avoid these terms: {avoid}" if avoid else "")
        ack2 = await _call_openai_chat(messages, model, max_tokens=min(16, MAX_TOKENS))
        ack2 = (ack2 or "").split("\n")[0][:160]
        if ack2 and not contains_blocked_phrase(ack2, project_id):
            return ack2

    return ack


async def generate_snapback(q: str, r: str, project_id: Optional[str], locale: Optional[str], model: str = DEFAULT_MODEL_SNAPBACK) -> Optional[str]:
    if not r or len(score_engine._tokens(r)) <= 3:
        messages = [
            {"role": "system", "content": SYSTEM_SNAPBACK + " " + _locale_instruction(locale)},
            {"role": "user", "content": f"Original question: {q.strip()} Respondent's answer: {r.strip()}"},
        ]
        text = await _call_openai_chat(messages, model, max_tokens=min(48, MAX_TOKENS))
        snap = text.split("\n")[0][:240]
        if not contains_blocked_phrase(snap, project_id):
            return snap
    return None


def _fallback_followup(q: str, r: str, locale: Optional[str]) -> str:
    lang = _language_bucket(locale, q, r)
    rules = LANG_RULES[lang]
    if len(score_engine._tokens(r)) <= 3:
        return rules["fallback_short_followup"]
    return rules["fallback_followup"]


async def generate_followup(
    question: str,
    response: str,
    project_id: Optional[str],
    locale: Optional[str],
    model: str = DEFAULT_MODEL_FOLLOWUP,
    context: Optional[dict] = None
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_FOLLOWUP + " " + _locale_instruction(locale)},
        {"role": "user", "content": f"Original question: {question.strip()} Respondent's answer: {response.strip()}"},
    ]
    text = await _call_openai_chat(messages, model=model, max_tokens=min(48, MAX_TOKENS))
    t = (text or "").strip().replace("\n", " ")

    if not t or t == "?" or len(score_engine._tokens(t)) < 6:
        t = _fallback_followup(question, response, locale)

    if not t.endswith("?") and not t.endswith("؟"):
        t = t.rstrip(" .!") + "?"

    if contains_blocked_phrase(t, project_id):
        avoid = ", ".join(sorted(list(get_block_phrases(project_id))))
        messages2 = [
            {"role": "system", "content": SYSTEM_FOLLOWUP + " " + _locale_instruction(locale) + (f" Avoid these terms: {avoid}" if avoid else "")},
            {"role": "user", "content": f"Original question: {question.strip()} Respondent's answer: {response.strip()}"},
        ]
        text2 = await _call_openai_chat(messages2, model=model, max_tokens=min(48, MAX_TOKENS))
        t2 = (text2 or "").strip().replace("\n", " ")
        if t2 and len(score_engine._tokens(t2)) >= 6 and (t2.endswith("?") or t2.endswith("؟")) and not contains_blocked_phrase(t2, project_id):
            return t2

    return t


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Verbalytics API", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_blocklist():
    _refresh_blocklist(True)


@app.get("/health")
def health() -> dict:
    try:
        _refresh_blocklist(False)
    except Exception:
        pass
    with _block_lock:
        global_count = len(_block_global)
        project_count = len(_block_projects)
    return {
        "ok": True,
        "model_ack": DEFAULT_MODEL_ACK,
        "model_snapback": DEFAULT_MODEL_SNAPBACK,
        "model_followup": DEFAULT_MODEL_FOLLOWUP,
        "blocklist_version": _block_version,
        "blocklist_global_count": global_count,
        "blocklist_projects": project_count,
    }


@app.post("/verbalytics", response_model=VerbalyticsOutput)
async def verbalytics(payload: VerbalyticsInput = Body(...)):
    start = time.time()

    input_hits = sorted(set(
        find_blocked_phrases(payload.question, payload.project_id) +
        find_blocked_phrases(payload.response, payload.project_id)
    ))

    if input_hits and INPUT_BLOCK_MODE == "reject":
        raise HTTPException(status_code=422, detail={"message": "Input contains blocked phrases", "phrases": input_hits})

    want_check = "check" in payload.tasks
    want_score = "score" in payload.tasks
    want_ack = "ack" in payload.tasks
    want_follow = "followup" in payload.tasks

    if want_check and not want_score and not want_ack and not want_follow:
        latency_ms = int((time.time() - start) * 1000)
        return VerbalyticsOutput(
            subscores=None,
            score=None,
            ai_likelihood=None,
            ack=None,
            snapback=None,
            followup=None,
            input_blocked_phrases=input_hits if input_hits else None,
            latency_ms=latency_ms,
        )

    snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response, payload.project_id, payload.locale))
    ack_task = asyncio.create_task(generate_ack(payload.question, payload.response, payload.project_id, payload.locale)) if want_ack else None
    follow_task = asyncio.create_task(generate_followup(
        payload.question, payload.response, payload.project_id, payload.locale, context=payload.context
    )) if want_follow else None

    subs = None
    score = ai_like = None
    if want_score:
        subs = score_engine.subscores(payload.question, payload.response, locale=payload.locale)
        score = score_engine.quality_score(payload.question, payload.response, locale=payload.locale)
        ai_like = score_engine.ai_likelihood(payload.response, locale=payload.locale)

    ack_res = follow_res = snap_res = None
    try:
        snap_res = await snap_task
        if ack_task:
            ack_res = await ack_task
        if follow_task:
            follow_res = await follow_task
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)
    return VerbalyticsOutput(
        subscores=subs,
        score=score,
        ai_likelihood=ai_like,
        ack=ack_res if want_ack else None,
        snapback=snap_res,
        followup=follow_res if want_follow else None,
        input_blocked_phrases=input_hits if input_hits else None,
        latency_ms=latency_ms,
    )


@app.post("/verbalytics/stream")
async def verbalytics_stream(payload: VerbalyticsInput = Body(...)):
    async def event_gen():
        start = time.time()

        want_check = "check" in payload.tasks
        want_score = "score" in payload.tasks
        want_ack = "ack" in payload.tasks
        want_follow = "followup" in payload.tasks

        input_hits = sorted(set(
            find_blocked_phrases(payload.question, payload.project_id) +
            find_blocked_phrases(payload.response, payload.project_id)
        ))

        if want_check and not want_score and not want_ack and not want_follow:
            yield JSONResponse(content={
                "type": "check",
                "input_blocked_phrases": input_hits or None
            }).body + b"\n"
            latency_ms = int((time.time() - start) * 1000)
            yield JSONResponse(content={"type": "done", "latency_ms": latency_ms}).body + b"\n"
            return

        if want_score:
            subs = score_engine.subscores(payload.question, payload.response, locale=payload.locale)
            score = score_engine.quality_score(payload.question, payload.response, locale=payload.locale)
            ai_like = score_engine.ai_likelihood(payload.response, locale=payload.locale)
            yield JSONResponse(content={
                "type": "score",
                "subscores": subs,
                "score": score,
                "ai_likelihood": ai_like,
                "input_blocked_phrases": input_hits or None,
            }).body + b"\n"

        snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response, payload.project_id, payload.locale))
        ack_task = asyncio.create_task(generate_ack(payload.question, payload.response, payload.project_id, payload.locale)) if want_ack else None
        follow_task = asyncio.create_task(generate_followup(
            payload.question, payload.response, payload.project_id, payload.locale, context=payload.context
        )) if want_follow else None

        snap = await snap_task
        if snap:
            yield JSONResponse(content={"type": "snapback", "snapback": snap}).body + b"\n"

        if ack_task:
            ack = await ack_task
            yield JSONResponse(content={"type": "ack", "ack": ack}).body + b"\n"

        if follow_task:
            follow = await follow_task
            yield JSONResponse(content={"type": "followup", "followup": follow}).body + b"\n"

        latency_ms = int((time.time() - start) * 1000)
        yield JSONResponse(content={"type": "done", "latency_ms": latency_ms}).body + b"\n"

    return StreamingResponse(event_gen(), media_type="application/jsonl")


# ---------------- Admin: Blocklist Introspection & Refresh ----------------
@app.get("/admin/blocklist")
async def admin_blocklist(project: Optional[str] = None):
    phrases = get_block_phrases(project)
    with _block_lock:
        return {
            "version": _block_version,
            "global_count": len(_block_global),
            "projects_map_count": len(_block_projects),
            "project": _norm(project) if project else None,
            "project_specific_count": len(_block_projects.get(_norm(project), set())) if project else None,
            "merged_count": len(phrases),
            "phrases": sorted(list(phrases)),
        }


@app.post("/admin/blocklist/refresh")
async def admin_blocklist_refresh():
    _refresh_blocklist(True)
    with _block_lock:
        return {
            "ok": True,
            "version": _block_version,
            "global_count": len(_block_global),
            "projects_map_count": len(_block_projects)
        }


# =========================================================
# DEEPDIVE MODULE
# =========================================================

SESSIONS = {}

TESCO_PROJECT_ID = "p338836335523"


class DeepDiveRequest(BaseModel):
    project_id: str
    respondent_id: str
    question_id: str
    response: str
    prior_answers: Optional[Dict[str, float]] = None
    locale: Optional[str] = None


def get_session_key(project_id, respondent_id, question_id):
    return f"{project_id}_{respondent_id}_{question_id}"


def load_question_config(project_id, question_id):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        base_dir,
        "configs",
        "projects",
        project_id,
        f"{question_id}.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_route(config, prior_answers):
    context = config.get("context", {})
    inputs = context.get("inputs", {})
    score_source = inputs.get("score_source")
    routes = context.get("routes", [])
    fallback_route = context.get("fallback_route", "neutral")

    score_val = None
    if score_source:
        score_val = prior_answers.get(score_source)

    if score_val is not None:
        for route in routes:
            rng = route.get("range")
            if not rng or len(rng) != 2:
                continue
            if rng[0] <= score_val <= rng[1]:
                return {
                    "label": route.get("label", fallback_route),
                    "focus": route.get("focus", "")
                }

    return {
        "label": fallback_route,
        "focus": "Explore the respondent's answer in depth and clarify what they mean."
    }


# =========================================================
# TESCO PROJECT-SPECIFIC LOGIC
# =========================================================

def is_tesco_low_content_response(response: str, locale: Optional[str] = None) -> bool:
    r = _clean_text(response)
    rl = _normalize_for_matching(r)

    low_content = {
        "dont know", "don't know",
        "i dont know", "i don't know",
        "idk", "dk",
        "not sure", "unsure",
        "nothing", "none",
        "na", "n/a",
        "no idea",
        "cant say", "can't say",
        "not really",
        "no", "nope"
    }

    if not r:
        return True

    if r.isdigit():
        return True

    if rl in low_content:
        return True

    return False


async def extract_tesco_improvement(response: str, locale: Optional[str] = None) -> str:
    r = _clean_text(response)

    if is_tesco_low_content_response(r, locale=locale):
        return ""

    rl = _normalize_for_matching(r)

    # Deterministic shortcuts for common expected themes
    if "better reporting" in rl:
        return "better reporting"

    if "clearer reporting" in rl:
        return "clearer reporting"

    if "more detailed reporting" in rl:
        return "more detailed reporting"

    if "reporting" in rl:
        return "better reporting"

    if "transparency" in rl or "transparent" in rl:
        return "greater transparency"

    if "roi" in rl or "return on investment" in rl:
        return "clearer proof of ROI"

    if "measurement" in rl or "measure" in rl:
        return "clearer measurement"

    if "audience data" in rl:
        return "stronger audience data"

    if "data" in rl:
        return "stronger data"

    if "insight" in rl or "insights" in rl:
        return "better insight"

    if "support" in rl or "service" in rl:
        return "more proactive support"

    if "communication" in rl or "communicate" in rl:
        return "clearer communication"

    if "case study" in rl or "case studies" in rl or "proof points" in rl:
        return "stronger proof points"

    prompt = f"""
Extract the main improvement or suggestion from this respondent answer.

Question context:
"What would Tesco Media need to do to strengthen your trust in them as your ideal media partner?"

Respondent answer:
"{r}"

Return ONLY a short noun phrase.
If multiple improvements are mentioned, return the single most important one.
Do not return a full sentence.
Do not include Tesco Media.
Do not add explanation.

Examples:
better reporting
greater transparency
clearer measurement
stronger audience data
more proactive support

If there is no clear improvement, return: unclear
"""

    try:
        result = await asyncio.to_thread(
            client.chat.completions.create,
            model=os.getenv("MODEL_DEEPDIVE", os.getenv("MODEL_FOLLOWUP", "gpt-4o-mini")),
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise research themes from survey responses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=20,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "12.0"))
        )

        extracted = result.choices[0].message.content.strip()
        extracted = extracted.strip("\"'“”‘’., ")

        if not extracted:
            return ""

        if extracted.lower() in {"unclear", "none", "nothing", "n/a", "na"}:
            return ""

        if len(extracted.split()) > 6:
            return ""

        return extracted

    except Exception:
        return ""


def build_tesco_followup_1(
    response: str,
    improvement: str,
    locale: Optional[str] = None
) -> str:
    if is_tesco_low_content_response(response, locale=locale):
        return "Is there another media network that you think does this well, and what do they do?"

    if improvement:
        return f"Is there another network you think already delivers {improvement} well, and what do they provide?"

    return "Is there another network you think already delivers this well, and what do they provide?"


def build_tesco_followup_2(
    first_response: str,
    improvement: str,
    locale: Optional[str] = None
) -> str:
    if is_tesco_low_content_response(first_response, locale=locale):
        return "What is it about that network that makes you feel confident choosing or working with them?"

    if improvement:
        return f"Why is {improvement} important to you, and what would it enable you to do?"

    return "Why is that important to you, and what would it enable you to do?"


# =========================================================
# GENERIC RESPONSE EVALUATION
# =========================================================

def evaluate_response_veronica(response: str, locale: Optional[str] = None):
    lang = _language_bucket(locale, response=response)
    eval_rules = LANG_RULES[lang]["deepdive_eval"]

    rl = _normalize_for_matching(response)
    quoted_phrase = ("'" in response) or ('"' in response) or ("«" in response) or ("»" in response)

    return {
        "mentions_language": any(x in rl for x in eval_rules["mentions_language"]),
        "mentions_visuals": any(x in rl for x in eval_rules["mentions_visuals"]),
        "mentions_meaning": any(x in rl for x in eval_rules["mentions_meaning"]),
        "mentions_relevance": any(x in rl for x in eval_rules["mentions_relevance"]),
        "is_vague": len(score_engine._tokens(rl)) < 5,
        "has_criticism_word": any(x in rl for x in eval_rules["criticism_words"]),
        "quoted_phrase": quoted_phrase,
        "lang": lang,
    }


def is_invalid_response(response: str, locale: Optional[str] = None) -> bool:
    r = _clean_text(response)
    rl = _normalize_for_matching(r)
    lang = _language_bucket(locale, response=r)
    rules = LANG_RULES[lang]

    if not r:
        return True

    if r.isdigit():
        return True

    if rl in rules["invalid_short"]:
        return True

    if len(score_engine._tokens(r)) <= 1:
        return True

    return False


def generate_repair_question(previous_question: str, bad_response: str, locale: Optional[str] = None) -> str:
    lang = _language_bucket(locale, previous_question, bad_response)
    rules = LANG_RULES[lang]

    r = _clean_text(bad_response)

    if r.isdigit():
        return rules["repair_numeric"].format(value=r, question=previous_question)

    return rules["repair_general"].format(question=previous_question)


def choose_next_question(route_label: str, evaluation: dict, locale: Optional[str] = None):
    lang = evaluation.get("lang") or _language_bucket(locale)
    qset = LANG_RULES[lang]["deepdive_questions"]

    if route_label == "low_score":
        if evaluation["mentions_language"]:
            if evaluation["quoted_phrase"]:
                return qset["low_score"]["ask_why_that_wording"], "ask_why_that_wording"
            if evaluation["has_criticism_word"]:
                return qset["low_score"]["ask_which_words"], "ask_which_words"
            return qset["low_score"]["ask_specific_words"], "ask_specific_words"

        elif evaluation["mentions_visuals"]:
            if evaluation["has_criticism_word"]:
                return qset["low_score"]["ask_which_visuals"], "ask_which_visuals"
            return qset["low_score"]["ask_specific_visuals"], "ask_specific_visuals"

        elif evaluation["mentions_meaning"]:
            return qset["low_score"]["ask_meaning"], "ask_meaning"

        else:
            return qset["low_score"]["ask_specifics"], "ask_specifics"

    elif route_label == "mid_score":
        if evaluation["mentions_language"]:
            return qset["mid_score"]["explore_both_sides_words"], "explore_both_sides"
        elif evaluation["mentions_visuals"]:
            return qset["mid_score"]["explore_both_sides_visuals"], "explore_both_sides"
        else:
            return qset["mid_score"]["explore_both_sides"], "explore_both_sides"

    elif route_label == "high_score":
        if evaluation["mentions_language"]:
            if evaluation["quoted_phrase"]:
                return qset["high_score"]["ask_why_that_wording"], "ask_why_that_wording"
            return qset["high_score"]["ask_specific_words"], "ask_specific_words"

        elif evaluation["mentions_visuals"]:
            return qset["high_score"]["ask_specific_visuals"], "ask_specific_visuals"

        elif evaluation["mentions_meaning"]:
            return qset["high_score"]["ask_meaning"], "ask_meaning"

        else:
            return qset["high_score"]["ask_specifics"], "ask_specifics"

    else:
        if evaluation["mentions_language"]:
            if evaluation["has_criticism_word"]:
                return qset["neutral"]["ask_which_words"], "ask_which_words"
            return qset["neutral"]["ask_specific_words"], "ask_specific_words"

        elif evaluation["mentions_visuals"]:
            return qset["neutral"]["ask_specific_visuals"], "ask_specific_visuals"

        elif evaluation["mentions_meaning"]:
            return qset["neutral"]["ask_meaning"], "ask_meaning"

        else:
            return qset["neutral"]["clarify"], "clarify"


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/deepdive")
async def deepdive(req: DeepDiveRequest):
    try:
        session_key = get_session_key(
            req.project_id,
            req.respondent_id,
            req.question_id
        )

        if session_key not in SESSIONS:
            SESSIONS[session_key] = {
                "history": [],
                "turn_count": 0,
                "repair_count": 0,
                "current_question_code": None,
                "current_question_text": None,
                "last_valid_question_text": None
            }

        session = SESSIONS[session_key]

        config = load_question_config(req.project_id, req.question_id)
        route = evaluate_route(config, req.prior_answers or {})
        route_label = route.get("label", "neutral")

        # -------------------------------------------------
        # PROJECT-SPECIFIC CARVE OUT: Tesco Media trust
        # -------------------------------------------------
        if req.project_id == TESCO_PROJECT_ID:
            session.setdefault("tesco", {
                "first_response": None,
                "improvement": None,
                "followup_1_response": None
            })

            tesco_state = session["tesco"]
            turn_count = session["turn_count"]

            session["history"].append({"role": "user", "text": req.response})

            # Turn 0: respondent answered the base question
            if turn_count == 0:
                improvement = await extract_tesco_improvement(req.response, locale=req.locale)

                tesco_state["first_response"] = req.response
                tesco_state["improvement"] = improvement

                next_question = build_tesco_followup_1(
                    response=req.response,
                    improvement=improvement,
                    locale=req.locale
                )

                question_code = f"{req.question_id}_followup_1"

                session["current_question_code"] = question_code
                session["current_question_text"] = next_question
                session["last_valid_question_text"] = next_question
                session["history"].append({"role": "assistant", "text": next_question})
                session["turn_count"] += 1

                return {
                    "route": "tesco_media_trust",
                    "decision": {
                        "action": "ask_comparator_delivery_example",
                        "reason": "project_specific_carve_out"
                    },
                    "deepdive": {
                        "question_code": question_code,
                        "next_question": next_question,
                        "action": "ask_comparator_delivery_example",
                        "should_continue": True,
                        "is_loop": False
                    }
                }

            # Turn 1: respondent answered follow-up-1
            if turn_count == 1:
                tesco_state["followup_1_response"] = req.response

                next_question = build_tesco_followup_2(
                    first_response=tesco_state.get("first_response") or "",
                    improvement=tesco_state.get("improvement") or "",
                    locale=req.locale
                )

                question_code = f"{req.question_id}_followup_2"

                session["current_question_code"] = question_code
                session["current_question_text"] = next_question
                session["last_valid_question_text"] = next_question
                session["history"].append({"role": "assistant", "text": next_question})
                session["turn_count"] += 1

                return {
                    "route": "tesco_media_trust",
                    "decision": {
                        "action": "ask_importance_and_enablement",
                        "reason": "project_specific_carve_out"
                    },
                    "deepdive": {
                        "question_code": question_code,
                        "next_question": next_question,
                        "action": "ask_importance_and_enablement",
                        "should_continue": True,
                        "is_loop": False
                    }
                }

            # Turn 2: respondent answered follow-up-2, so stop
            return {
                "route": "tesco_media_trust",
                "decision": {
                    "action": "stop",
                    "reason": "project_specific_session_complete"
                },
                "deepdive": {
                    "should_continue": False,
                    "stop_reason": "project_specific_session_complete"
                }
            }

        # -------------------------------------------------
        # GENERIC DEEPDIVE LOGIC
        # -------------------------------------------------

        session["history"].append({"role": "user", "text": req.response})

        if is_invalid_response(req.response, locale=req.locale):
            session["repair_count"] += 1

            if session["repair_count"] >= 3:
                return {
                    "route": route_label,
                    "decision": {
                        "action": "stop",
                        "reason": "repair_exhausted"
                    },
                    "deepdive": {
                        "should_continue": False,
                        "stop_reason": "repair_exhausted"
                    }
                }

            previous_question = (
                session.get("current_question_text")
                or session.get("last_valid_question_text")
                or config.get("question_context", {}).get("main_question", "Please answer in words.")
            )

            question_code = session.get("current_question_code") or f"{req.question_id}_1"
            repair_question = generate_repair_question(previous_question, req.response, locale=req.locale)

            session["history"].append({"role": "assistant", "text": repair_question})
            session["current_question_text"] = previous_question

            return {
                "route": route_label,
                "decision": {
                    "action": "repair",
                    "reason": "invalid_response_same_turn"
                },
                "deepdive": {
                    "question_code": question_code,
                    "next_question": repair_question,
                    "action": "repair",
                    "should_continue": True,
                    "is_loop": True
                }
            }

        session["repair_count"] = 0

        evaluation = evaluate_response_veronica(req.response, locale=req.locale)
        question_code = f"{req.question_id}_{session['turn_count'] + 1}"

        next_question, action = choose_next_question(route_label, evaluation, locale=req.locale)

        session["current_question_code"] = question_code
        session["current_question_text"] = next_question
        session["last_valid_question_text"] = next_question
        session["history"].append({"role": "assistant", "text": next_question})
        session["turn_count"] += 1

        return {
            "route": route_label,
            "decision": {
                "action": action,
                "reason": "route_and_response_selected"
            },
            "deepdive": {
                "question_code": question_code,
                "next_question": next_question,
                "action": action,
                "should_continue": True,
                "is_loop": False
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
