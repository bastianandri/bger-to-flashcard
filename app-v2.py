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
APP_UNTERTITEL = (
    "Füge eine Bundesgerichtsentscheidnummer oder den Urteilstext ein. "
    "<em>Entscheidkarten</em> fasst den Entscheid in vier prägnanten Punkten zusammen "
    "und erstellt eine fertige Anki-Karteikarte zum Download."
)

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
        "Verwende korrekte schweizerische juristische Terminologie und nenne die jeweilig einschlägigen Rechtsnormen (OR, ZGB, StGB, BV etc.). "
        "Falls nötig, konsultiere weitere Quellen im Zusammenhang mit dem Entscheid. "
        "Gib die Entscheidnummer an, falls es nah verwandte Entscheide gibt, die im Kontext des Entscheids relevant sind."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=CaseBrief,
        temperature=0.2,
    )

    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]
    
    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=bger_text,
                config=config,
            )
            if getattr(response, "parsed", None) and isinstance(response.parsed, CaseBrief):
                return response.parsed
            return CaseBrief.model_validate_json(response.text)
        except Exception as e:
            last_error = e
            if "404" in str(e) or "NOT_FOUND" in str(e):
                continue
            raise e

    raise last_error

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

# --- 5. WEBSITE UI (STREAMLIT) ---
# Setzt den Tab-Titel im Browser auf "Entscheidkarten" und das Tab-Icon (Favicon) auf das Waage-Emoji
st.set_page_config(page_title=APP_TITEL, page_icon="⚖️", layout="centered")

# CSS für volle Lesbarkeit in Light & Dark Mode (nutzt Streamlits native Theme-Variablen)
# sowie den unten fixierten und an die Textbreite angepassten Disclaimer
st.markdown(
    """
    <style>
        /* Genug Abstand unten im Hauptcontainer, damit der fixierte Footer nichts überdeckt */
        .block-container {
            padding-bottom: 7.5rem !important;
        }

        /* Kopfzeilen-Styling (passt sich automatisch an Light & Dark Mode von Streamlit an) */
        .custom-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 10px;
            border-bottom: 2px solid rgba(128, 128, 128, 0.25);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .custom-title {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
            color: var(--text-color) !important;
        }
        .author-badge {
            font-size: 0.85rem;
            color: var(--text-color) !important;
            font-weight: 500;
            background-color: var(--secondary-background-color);
            padding: 5px 12px;
            border-radius: 9999px;
            border: 1px solid rgba(128, 128, 128, 0.3);
            white-space: nowrap;
        }
        .author-badge a {
            color: var(--text-color) !important;
            text-decoration: underline;
            text-underline-offset: 3px;
            font-weight: 600;
            transition: opacity 0.2s ease;
        }
        .author-badge a:hover {
            opacity: 0.75;
        }
        .custom-subtitle {
            font-size: 1.02rem;
            color: var(--text-color) !important;
            opacity: 0.88;
            margin-bottom: 24px;
            line-height: 1.55;
        }

        /* Fixierter Footer unten: exakt so breit und zentriert wie der Streamlit-Textbereich */
        .footer-wrapper {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            pointer-events: none;
            z-index: 999;
            display: flex;
            justify-content: center;
            background: linear-gradient(to top, var(--background-color) 80%, transparent);
            padding: 12px 1rem 10px 1rem;
            box-sizing: border-box;
        }
        .footer-disclaimer {
            pointer-events: auto;
            width: 100%;
            max-width: 704px; /* Exakte Standard-Textbreite von Streamlit (layout="centered") */
            background-color: var(--secondary-background-color);
            color: var(--text-color) !important;
            font-size: 0.74rem;
            line-height: 1.45;
            padding: 10px 14px;
            text-align: justify;
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            box-sizing: border-box;
        }
        .footer-disclaimer strong {
            color: var(--text-color) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Elegante Kopfzeile: Titel ohne Emoji, rechts "Built by Bastian Stählin" mit klickbarem LinkedIn-Link
st.markdown(
    f"""
    <div class="custom-header">
        <h1 class="custom-title">{APP_TITEL}</h1>
        <span class="author-badge">
            Built by <a href="https://www.linkedin.com/in/bastian-st%C3%A4hlin" target="_blank" rel="noopener noreferrer">Bastian Stählin</a>
        </span>
    </div>
    <p class="custom-subtitle">
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
    # st.form sorgt dafür, dass ein Druck auf die Enter-Taste im Textfeld das Formular direkt absendet
    with st.form(key="search_form", border=False):
        case_num = st.text_input("Geschäftsnummer oder BGE:", placeholder="z. B. BGE 148 III 12 oder 6B_450/2024")
        fetch_btn = st.form_submit_button("Entscheid automatisch laden & zusammenfassen", type="primary")

with tab2:
    # Im grossen Textfeld (text_area) macht Enter einen Zeilenumbruch, Strg+Enter (bzw. Cmd+Enter) sendet ab
    with st.form(key="manual_form", border=False):
        manual_text = st.text_area(
            "BGer-Entscheid Text hier einfügen:",
            height=220,
            placeholder="Kopiere hier den Text von bger.ch hinein (Sachverhalt A. ... Erwägung 2.1 ...)"
        )
        manual_btn = st.form_submit_button("Manuellen Text zusammenfassen", type="primary")

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
            with st.spinner("Ich analysiere den Entscheid und baue die Anki-Karte..."):
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
- **Sachverhalt:** {brief.sachverhalt}
- **Rechtsfrage:** {brief.rechtsfrage}
- **Erwägungen:** {brief.erwaegungen}
- **Fazit / Rechtssatz:** {brief.fazit}
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

# --- 6. FUSSZEILE / DISCLAIMER (UNTEN FIXIERT & IN TEXTBREITE) ---
st.markdown(
    """
    <div class="footer-wrapper">
        <div class="footer-disclaimer">
           </strong> Die Zusammenfassungen werden KI-gestützt generiert, können Fehler enthalten und sind stets anhand der amtlichen Originalquelle zu überprüfen. 
            Kurzzusammenfassungen dienen der effizienten Repetition; für ein vertieftes juristisches Verständnis sowie die Erfassung des Gesamtkontexts und der vollständigen Argumentation wird die Lektüre des gesamten Entscheids empfohlen.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
