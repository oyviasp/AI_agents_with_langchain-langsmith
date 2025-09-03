from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

load_dotenv()

# Initialize OpenAI LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_react_agent(
    model   =   model,
    tools   =   [add, multiply],
    name    =   "math_expert",
    prompt  =   "You are a math expert specializing in calculations. Always use tools for ANY mathematical operations - addition, multiplication, etc. Never calculate manually. When given multiple numbers to add, use the add tool step by step."
)

research_agent = create_react_agent(
    model   =   model,
    tools   =   [web_search],
    name    =   "research_expert",
    prompt  =   "You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    agents          =   [research_agent, math_agent],
    model           =   model,
    output_mode     =   "full_history", # could give "last_message" for less context
    prompt          =   """You are a team supervisor managing a research expert and a math expert.

DELEGATION RULES:
- For current events and data gathering: use research_expert
- For ANY mathematical calculations (addition, multiplication, percentages, etc.): use math_expert
- If research_expert provides data that needs calculation, ALWAYS delegate to math_expert
- NEVER do calculations yourself - always use math_expert for any math operations

WORKFLOW:
1. If user asks for data + calculation: first research_expert, then math_expert
2. Always clearly state which agent you're delegating to and why"""
)

# Compile and run
app = workflow.compile()

png_bytes = app.get_graph().draw_mermaid_png()
with open("./LangGraph_2_Agents/LG_2_multi_agent_1_supervisor.png", "wb") as f:
    f.write(png_bytes)

result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})