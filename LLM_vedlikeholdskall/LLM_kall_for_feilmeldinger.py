from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate        # OpenAI as llm
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class FeilmeldingState(TypedDict):
    tittel:         str = Field(description="Tittel på feilmeldingen")
    beskrivelse:    str = Field(description="Beskrivelse av feilen i feilmeldingen")
    prioritet:      Literal["1", "2", "3", "4","5","6"] = Field(description="Prioritet på feilmeldingen")
    teknisk_plass:  str = Field(description="Teknisk plassering av feilen")


# Augment the LLM with schema for structured output
feilmelding = llm.with_structured_output(FeilmeldingState)

def _build_title_prompt(self, problem_description: str) -> str:
    """Build the prompt for title generation."""
    return f"""
    Basert på følgende problembeskrivelse, lag en kort, konsis og beskrivende tittel for feilmeldingen:

    Tittel skal være:
    - Maksimalt 2-4 ord
    - Beskrive hva som observeres, ikke årsaken
    - Bruke teknisk terminologi
    - Skal KUN inneholde HOVEDESSENSEN av meldingen, og skal IKKE inneholde detaljer om feilmeldingen
    - Skal enkelt kunne forstå HELE problemet, kun ved å lese tittelen

    Eksempel på en god tittel:
    - Utslitte luftslanger

    Problembeskrivelse: {problem_description}

    Svar kun med tittelen, ingen ekstra formatering:
    """



PROMPT_TEMPLATE = """
Analyser førbrukskontrollen førbrukskontrollen og basert på denne foreslå en tittel og beskrivelse av feilmeldingen basert på kriterier:
Beskirvelsen av feilmeldingen skal være så detaljert og presis som du klarer ved å bruke informasjonen fra HELE førbrukskontrollen, dersom 
det ikke er nok informasjon til å lage en fullstendig beskrivelse, beskriv så mye du kan og skriv tydelig hva som mangler

MANGLENE INFORMASJON SOM ER NØDVENDIG FOR EN GOD FEILMELDING:
-(...)
-(...)

Tips til hva man kan be om å utdype/be om informasjon om:
- Hvilket utstyr/system/lokasjon problemet gjelder
- Spesifikke symptomer, lyder, lukter, synlige problemer
- Når problemet oppstår eller hvor ofte
- Når/hvordan/under hvilke forhold feilen oppstod

Førbrukskontroll: "{Forbrukskontroll}"

En feilmelding skal inneholde DETALJERT beskrivelse av hva som observeres, IKKE nødvendigvis hva som forårsaker problemet.

En GODKJENT feilmelding må inneholde:
- Detaljert beskrivelse av hva som observeres - spesifikke symptomer, lyder, lukter, synlige problemer
- Hvor observasjonen er gjort - hvilket utstyr/system/lokasjon
- Eventuelt når problemet oppstår eller hvor ofte
- Når/hvordan/under hvilke forhold feilen oppstod - f.eks. ved oppstart av maskinen, ved vanlig bruk, ved høy last

En feilmelding er godkjent hvis den inneholder ALLE punktene for en godkjent feilmelding!
En feilmelding er IKKE godkjent hvis den er vag, generisk eller mangler spesifikke observasjoner.

Eksempler på GODKJENTE feilmeldinger:
- "ASI-Bus 10 viser gjentagende feilmeldinger på displayet hver 30 sekund"
- "Alle slangene på pumpe 5 har synlige sprekker og det lekker væske"
- "Ovn 10 Pfa2 starter ikke og displayet viser feilkode E23"
- "Kabelbro for kommunikasjonskabel henger løst og er fysisk ødelagt"
- "Pumpe 3 lager høye, skrapende lyder når den kjører"
- "Det kommer røyk fra motor på kompressor 5"

Eksempler på IKKE GODKJENTE feilmeldinger (IKKE NOK DETALJER):
- "ASI-Bus 10 har problemer" (hva slags problemer?)
- "Slangen er ødelagt" (hvilken slange? hvor? hva ser man?)
- "Pumpen funker ikke" (hva skjer når den ikke funker?)
- "Det er feil" (hvilken type feil? hva observeres?)
- "Funker ikke" (hva observeres når det ikke funker?)
- "Starter ikke" (hva skjer når man prøver å starte?)
- "Lekker" (hva lekker? hvor? hvor mye?)

Vær STRENG: Kun marker problem_described som true hvis feilmeldingen inneholder SPESIFIKKE og DETALJERTE observasjoner.
"""
