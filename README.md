# AI Assistant with RAG & Calculator

En AI-assistent bygget med LangChain og Streamlit som kombinerer Retrieval-Augmented Generation (RAG) med kalkulatorfunksjonalitet. Assistenten kan søke i PDF-dokumenter og utføre matematiske beregninger.

## Funksjoner

- 🤖 **AI Agent**: Intelligent assistent med flere verktøy
- 📚 **RAG (Retrieval-Augmented Generation)**: Søk i PDF-dokumenter for kontekstuell informasjon
- 🧮 **Kalkulator**: Utfør komplekse matematiske beregninger
- 💬 **Chat Interface**: Interaktiv samtale med minnefunksjon
- 🌐 **Web Interface**: Streamlit-basert nettgrensesnitt
- 📊 **LangSmith Tracing**: Sporing og debugging av AI-agenten

## Forutsetninger

- Python 3.8+
- OpenAI API-nøkkel
- LangSmith-konto (valgfritt, for sporing)

## Installasjon

1. **Klon repositoryet:**
   ```bash
   git clone <repository-url>
   cd MyFirstLangSmithAgent
   ```

2. **Opprett virtuelt miljø:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # eller
   source venv/bin/activate  # macOS/Linux
   ```

3. **Installer avhengigheter:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Opprett `.env` fil:**
   ```bash
   cp .env.example .env  # eller opprett manuelt
   ```

## Miljøvariabler

Opprett en `.env` fil i rotmappen med følgende variabler:

```env
# PÅKREVD: OpenAI API-nøkkel
OPENAI_API_KEY=your_openai_api_key_here

# VALGFRITT: LangSmith for sporing og debugging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name

# VALGFRITT: Andre LangChain-innstillinger
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Hvordan få API-nøkler:

1. **OpenAI API-nøkkel** (påkrevd):
   - Gå til [OpenAI Platform](https://platform.openai.com/)
   - Opprett konto eller logg inn
   - Naviger til "API Keys" i brukermenyen
   - Opprett en ny API-nøkkel
   - **VIKTIG**: Denne nøkkelen må holdes hemmelig!

2. **LangSmith API-nøkkel** (valgfritt):
   - Gå til [LangSmith](https://smith.langchain.com/)
   - Opprett konto eller logg inn
   - Gå til innstillinger og opprett en API-nøkkel
   - Dette gir deg sporing og debugging av AI-agenten

## Bruk

### Streamlit Web-app (anbefalt)
```bash
streamlit run agent_multitool_with_rag_streamlit.py
```
Åpne nettleseren på http://localhost:8501

### Kommandolinje-versjoner
```bash
# Enkel agent med verktøy
python agent_multitool.py

# Agent med RAG
python agent_multitool_with_rag.py

# Agent med RAG og chat-minnefunksjon
python agent_multitool_with_rag_chat.py
```

### RAG-søk versjoner
```bash
# PDF-søk
python rag_search_pdf.py

# Web-søk
python rag_search_web.py

# Fusion search
python rag_fusion_search.py
```

## Filstruktur

```
MyFirstLangSmithAgent/
├── agent_multitool.py                    # Grunnleggende multi-tool agent
├── agent_multitool_with_rag.py          # Agent med RAG-funksjonalitet
├── agent_multitool_with_rag_chat.py     # Agent med RAG og chat-minne
├── agent_multitool_with_rag_streamlit.py # Streamlit web-app
├── rag_search_pdf.py                    # RAG PDF-søk
├── rag_search_web.py                    # RAG web-søk
├── rag_fusion_search.py                 # Avansert fusion search
├── web_search.py                        # Web-søkeverktøy
├── main.py                              # Hovedfil
├── requirements.txt                      # Python-avhengigheter
├── .env                                 # Miljøvariabler (opprett selv)
├── .gitignore                           # Git ignore-fil
└── docs_for_rag/                        # PDF-dokumenter for RAG
    ├── impromptu.pdf                    # Impromptu speaking guide
    └── GEP-June-2025.pdf               # Økonomisk rapport
```

## PDF-dokumenter

Legg PDF-filer i `docs_for_rag/` mappen. Eksempler på inkluderte dokumenter:
- `impromptu.pdf`: Guide til impromptu speaking
- `GEP-June-2025.pdf`: Økonomisk rapport

## Verktøy tilgjengelig for AI-assistenten

1. **RAG PDF Search**: Søker i PDF-dokumenter for relevant informasjon
2. **Calculator**: Utfører matematiske beregninger og operasjoner
3. **Web Search**: Søker på internett (i noen versjoner)

## Feilsøking

### Vanlige problemer:

1. **"OpenAI API key not found"**
   - Sjekk at `.env` filen inneholder `OPENAI_API_KEY`
   - Sørg for at `.env` er i samme mappe som Python-filene

2. **"No module named 'streamlit'"**
   - Installer avhengigheter: `pip install -r requirements.txt`
   - Sjekk at du har aktivert det virtuelle miljøet

3. **PDF-filer ikke funnet**
   - Sørg for at PDF-filene ligger i `docs_for_rag/` mappen
   - Sjekk filnavn og stier i koden

4. **ChromaDB-feil**
   - Slett `chroma_db/` mappen for å resette vektordatabasen
   - Restart appen for å bygge ny database

## Lisens

[Spesifiser lisens her]

## Bidrag

[Instruksjoner for hvordan andre kan bidra til prosjektet]

## Support

For spørsmål eller problemer, opprett en issue i GitHub-repositoryet.
