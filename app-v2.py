import hashlib
import html
import os
import re
import tempfile
from typing import Optional

import genanki
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
APP_TITEL = "Entscheidkarten für Studierende"
APP_UNTERTITEL = (
    "Schweizer Bundesgerichtsentscheide in Kernaussagen fassen und direkt als Anki-Karteikarte exportieren."
)
FEST_EINGEBAUTER_KEY = ""  # Für Deployments besser Streamlit Secrets verwenden.
MAX_TEXT_LENGTH = 40_000

st.set_page_config(
    page_title=APP_TITEL,
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------
class CaseBrief(BaseModel):
    titel: str = Field(
        description="Kurzer Titel für die Zusammenfassung, ohne erfundene Angaben."
    )
    entscheidnummer: str = Field(
        description=(
            "Exakte BGE- oder BGer-Geschäftsnummer, soweit sie im Urteilstext steht; "
            "sonst leer lassen. Niemals eine Nummer erfinden."
        )
    )
    rechtsfrage: str = Field(
        description=(
            "Die zentrale Rechtsfrage als direkte, präzise Frage formulieren, "
            "z. B. 'Welche Voraussetzungen gelten für eine Vertrauenshaftung?' "
            "Mit Fragezeichen abschliessen; noch keine Antwort vorwegnehmen. Keine Entscheidnummer "
            "in die Rechtsfrage schreiben, diese gehört nur in entscheidnummer."
        )
    )
    sachverhalt: str = Field(
        description="Relevanter Sachverhalt in genau einem prägnanten Satz."
    )
    erwaegungen: str = Field(
        description="Wichtigste rechtliche Erwägungen in ein bis zwei Sätzen."
    )
    fazit: str = Field(
        description="Zentrale Erkenntnis bzw. Rechtssatz in ein bis zwei präzisen Sätzen."
    )


# ---------------------------------------------------------------------------
# Entscheid abrufen und Text bereinigen
# ---------------------------------------------------------------------------
def extract_decision_number(text: str) -> str:
    """Extrahiert eine ausdrücklich genannte BGE- oder BGer-Nummer, ohne zu raten."""
    text = str(text or "")
    patterns = (
        r"\bBGE\s+\d{1,3}\s+[IVX]{1,4}\s+\d{1,4}\b",
        r"\b\d{1,2}[A-Z]{1,2}_\d{1,5}/\d{4}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if value.upper().startswith("BGE"):
                return "BGE " + value[3:].strip()
            return value.upper()
    return ""


def format_as_question(value: str) -> str:
    """Bereinigt die KI-Rechtsfrage und stellt sicher, dass sie Frageform hat."""
    question = re.sub(r"^\s*(rechtsfrage|frage)\s*:\s*", "", str(value or ""), flags=re.IGNORECASE)
    # Falls die KI trotzdem die Referenz anhängt, entfernen wir sie hier; sie wird separat
    # genau einmal in Klammern an die Vorderseite gesetzt.
    question = re.sub(r"\s*\(?\bBGE\s+\d{1,3}\s+[IVX]{1,4}\s+\d{1,4}\)?", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s*\(?\b\d{1,2}[A-Z]{1,2}_\d{1,5}/\d{4}\)?", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\b(?:nach|gemäss|gemäß|laut|im|in)\s*\??\s*$", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s+", " ", question).strip().strip('"“”„')
    if not question:
