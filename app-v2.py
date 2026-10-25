import os
import re
import random
import tempfile
import requests
from bs4 import BeautifulSoup
import streamlit as st
import genanki
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- EINSTELLUNGEN FÜR TITEL & KEY ---
# Hier kannst du den Namen deiner Website jederzeit anpassen:
APP_TITEL = "Entscheidkarten"
APP_UNTERTITEL = "Füge eine Bundesgerichtsentscheidnummer oder den Urteilstext ein. BGer-to-Flashcard fasst den Entscheid in vier prägnanten Punkten zusammen und erstellt eine fertige Anki-Karteikarte zum Download."

# Optional für lokalen Test: Du kannst deinen kostenlosen Google-Key auch direkt
# hier zwischen die Anführungszeichen setzen (Achtung: Nicht öffentlich auf GitHub hochladen,
# für GitHub/Streamlit Cloud stattdessen die "Secrets"-Funktion nutzen!)
FEST_EINGEBAUTER_KEY = ""

# --- 1. DATENSTRUKTUR FÜR DIE KI ---
class CaseBrief(BaseModel):
    titel: str = Field(description="Kurzer, prägnanter Titel inkl. BGE/BGer-Nummer (z. B. 'BGE 148 III 12 - Mietrecht / Kündigungsschutz')")
    rechtsfrage: str = Field(description="Die zentrale Rechtsfrage des Entscheids in einem präzisen Satz (für die Kartenvorderseite)")
    sachverhalt: str = Field(description="Exakt 1 prägnanter Satz zum relevanten Sachverhalt")
    erwaegungen: str = Field(description="1 bis maximal 2 Sätze zu den wichtigsten rechtlichen Erwägungen des Bundesgerichts")
    fazit: str = Field(description="Exakt 1 Satz zum Urteilsspruch / Rechtssatz (Ratio Decidendi)")

# --- 2. AUTOMATISCHER ABRUF ÜBER ENTSCHEIDSUCHE.CH ---
def fetch_bger_decision(case_number: str) -> str:
    """Sucht einen Entscheid über die offene Schnittstelle von entscheidsuche.ch."""
    search_url = "https://entscheidsuche.ch/_search.php"
    clean_num = case_number.strip()
    
    query_payload = {
        "query": {
            "simple_query_string": {
                "query": f'"{clean_num}"',
                "fields": ["title^3", "Kopfzeile^3", "Meta^2", "Signatur^3", "attachment.content"]
            }
        },
        "size": 1
    }
    
    headers = {"User-Agent": "BGer-Flashcard-Generator/1.0 (LegalTech App)"}
    response = requests.post(search_url, json=query_payload, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        raise ValueError(
            f"Kein Entscheid unter '{clean_num}' gefunden. "
            "Tipp: Prüfe die Schreibweise (z. B. 'BGE 148 III 12' oder '6B_450/2024') "
            "oder nutze den zweiten Tab 'Text manuell einfügen'."
        )
    
    source = hits[0].get("_source", {})
    
    # Option A: Volltext ist direkt im Suchindex vorhanden
    text = source.get("attachment", {}).get("content", "")
    if text and len(text) > 200:
        return clean_bger_text(text)
    
    # Option B: HTML-Dokument vom Server nachladen
    html_info = source.get("HTML", {})
    file_path = html_info.get("Datei", "") if isinstance(html_info, dict) else ""
    if file_path:
        doc_url = f"https://entscheidsuche.ch/docs/{file_path.lstrip('/')}"
        doc_resp = requests.get(doc_url, headers=headers, timeout=15)
        doc_resp.raise_for_status()
        soup = BeautifulSoup(doc_resp.text, "html.parser")
        return clean_bger_text(soup.get_text(separator="\n"))
        
    raise ValueError(
        "Der Entscheid wurde gefunden, aber der Volltext liegt nur als gescanntes PDF vor. "
        "Bitte kopiere den Text manuell in den zweiten Tab."
    )

def clean_bger_text(raw_text: str) -> str:
    """Schneidet das Rubrum (Parteien/Richter) ab und behält Regeste, Sachverhalt & Erwägungen."""
    markers = [
        r"Regeste\b",
        r"Sachverhalt\s*:",
        r"A\.\-",
        r"A\.\s",
        r"Faits\s*:",
        r"Fatti\s*:",
        r"Erwägungen\s*:"
    ]
    pattern = re.compile("|".join(markers), re.IGNORECASE)
    match = pattern.search(raw_text)
    
    cleaned = raw_text[match.start():] if match else raw_text
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned[:40000]

# --- 3. KI-ZUSAMMENFASSUNG MIT GOOGLE GEMINI GENERIEREN ---
def generate_brief(bger_text: str, api_key: str) -> CaseBrief:
    client = genai.Client(api_key=api_key.strip())
    
    system_prompt = (
        "Du bist ein Schweizer Rechtsprofessor. Analysiere den folgenden Entscheid des "
        "Schweizerischen Bundesgerichts (BGer/BGE). Fasse ihn extrem präzise in exakt "
        "4 kurzen Kernaussagen (abstrakter Sachverhalt, Rechtsfrage, Erwägungen, Fazit) auf Deutsch zusammen. "
        "Verwende korrekte schweizerische juristische Terminologie und nenne die jeweilig einschlägigen Rechtsnormen (OR, ZGB, StGB, BV etc.)."
        "Falls nötig, konsultiere weitere Quellen im Zusammenhang mit dem Entscheid." 
        "Gib die Entscheidnummer an, falls es nah verwandte Entscheide gibt, die im Kontext des Entscheids relevant sind."
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=bger_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=CaseBrief,
            temperature=0.2,
        ),
    )
    if getattr(response, "parsed", None) and isinstance(response.parsed, CaseBrief):
        return response.parsed
    return CaseBrief.model_validate_json(response.text)

# --- 4. ANKI-DATEI (.apkg) ERSTELLEN (Windows- & Mac-sicher) ---
def create_anki_deck(brief: CaseBrief) -> bytes:
    model_id = 1607392319
    deck_id = random.randrange(1 << 30, 1 << 31)

    bger_model = genanki.Model(
        model_id,
        'BGer Case Brief Modell',
        fields=[
            {'name': 'Titel'},
            {'name': 'Rechtsfrage'},
            {'name': 'Bulletpoints'},
        ],
        templates=[
            {
                'name': 'BGer Karte',
                'qfmt': '<h3>{{Titel}}</h3><hr><p><b>Rechtsfrage:</b><br>{{Rechtsfrage}}</p>',
                'afmt': '{{FrontSide}}<hr id="answer"><ul>{{Bulletpoints}}</ul>',
            },
        ],
        css="""
        .card { font-family: Arial, sans-serif; font-size: 16px; text-align: left; color: #1e1e1e; background-color: white; padding: 15px; }
        h3 { color: #b71c1c; margin-bottom: 5px; }
        ul { line-height: 1.5; padding-left: 20px; }
        li { margin-bottom: 10px; }
        """
    )

    bulletpoints_html = (
        f"<li><b>Sachverhalt:</b> {brief.sachverhalt}</li>"
        f"<li><b>Rechtsfrage:</b> {brief.rechtsfrage}</li>"
        f"<li><b>Erwägungen:</b> {brief.erwaegungen}</li>"
        f"<li><b>Fazit / Rechtssatz:</b> {brief.fazit}</li>"
    )

    note = genanki.Note(
        model=bger_model,
        fields=[brief.titel, brief.rechtsfrage, bulletpoints_html]
    )

    deck = genanki.Deck(deck_id, 'Schweizer Recht - BGer Entscheide')
    deck.add_note(note)

    with tempfile.TemporaryDirectory() as tmp_dir:
        apkg_path = os.path.join(tmp_dir, "bger_flashcard.apkg")
        genanki.Package(deck).write_to_file(apkg_path)
        with open(apkg_path, "rb") as f:
            return f.read()


# Elegante Kopfzeile mit Titel und "Built by Bastian Stählin"-Badge
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; margin-bottom: 16px;">
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;">
            ⚖️ {APP_TITEL}
        </h1>
        <span style="font-size: 0.85rem; color: #6b7280; font-weight: 500; background-color: #f3f4f6; padding: 4px 10px; border-radius: 9999px; border: 1px solid #e5e7eb;">
            Built by Bastian Stählin
        </span>
    </div>
    <p style="font-size: 1.05rem; color: #4b5563; margin-bottom: 24px;">
        {APP_UNTERTITEL}
    </p>
    """,
    unsafe_allow_html=True,
)

# API-Key automatisch aus Streamlit-Secrets oder der Konstante oben laden
default_key = FEST_EINGEBAUTER_KEY.strip()
if not default_key:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# Eingabefeld in der Sidebar NUR anzeigen, wenn noch kein Key fest hinterlegt ist
if default_key:
    api_key = default_key
else:
    api_key = st.sidebar.text_input(
        "🔑 Google Gemini API-Key hier eingeben:",
        type="password",
        help="Kostenlos erhältlich unter aistudio.google.com/apikey"
    )

tab1, tab2 = st.tabs(["🔍 Per Entscheidnummer suchen", "📋 Text manuell einfügen"])

with tab1:
    case_num = st.text_input("Geschäftsnummer oder BGE:", placeholder="z. B. BGE 148 III 12 oder 6B_450/2024")
    fetch_btn = st.button("Entscheid automatisch laden & zusammenfassen", type="primary", key="btn_fetch")

with tab2:
    manual_text = st.text_area(
        "BGer-Entscheid Text hier einfügen:",
        height=220,
        placeholder="Kopiere hier den Text von bger.ch hinein (Sachverhalt A. ... Erwägung 2.1 ...)"
    )
    manual_btn = st.button("Manuellen Text zusammenfassen", type="primary", key="btn_manual")

# Verarbeitung starten
if fetch_btn or manual_btn:
    if not api_key:
        st.error("⚠️ Bitte gib zuerst links in der Seitenleiste deinen kostenlosen Google Gemini API-Key ein.")
    else:
        target_text = ""
        if fetch_btn:
            if not case_num.strip():
                st.warning("Bitte gib eine BGE- oder BGer-Nummer ein.")
            else:
                with st.spinner(f"Suche '{case_num}' auf entscheidsuche.ch..."):
                    try:
                        target_text = fetch_bger_decision(case_num)
                        st.session_state["raw_preview"] = target_text[:1200] + "\n\n[...]"
                    except Exception as e:
                        st.error(f"Fehler beim Abruf: {e}")
        elif manual_btn:
            if not manual_text.strip():
                st.warning("Bitte füge zuerst einen Urteilstext ein.")
            else:
                target_text = clean_bger_text(manual_text)
                st.session_state["raw_preview"] = target_text[:1200] + "\n\n[...]"

        if target_text:
            with st.spinner("Ich analysiert den Entscheid und baue die Anki-Karte..."):
                try:
                    brief = generate_brief(target_text, api_key)
                    st.session_state["brief"] = brief
                    st.session_state["anki_bytes"] = create_anki_deck(brief)
                except Exception as e:
                    st.error(f"Fehler bei der KI-Generierung: {e}")

# Ergebnisse anzeigen
if "brief" in st.session_state:
    brief = st.session_state["brief"]
    st.divider()
    st.subheader(f"📌 {brief.titel}")
    
    st.markdown(f"""
* **Sachverhalt:** {brief.sachverhalt}
* **Rechtsfrage:** {brief.rechtsfrage}
* **Erwägungen:** {brief.erwaegungen}
* **Fazit / Rechtssatz:** {brief.fazit}
""")

    safe_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", brief.titel)[:50]
    st.download_button(
        label="📥 Als Anki-Datei (.apkg) herunterladen",
        data=st.session_state["anki_bytes"],
        file_name=f"{safe_filename}.apkg",
        mime="application/octet-stream"
    )
    
    if "raw_preview" in st.session_state:
        with st.expander("🔍 Vorschau des eingelesenen Texts (ohne Rubrum)"):
            st.text(st.session_state["raw_preview"])
