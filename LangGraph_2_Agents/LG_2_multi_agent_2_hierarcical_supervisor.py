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

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

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

def write_article(topic: str, data: str) -> str:
    """Write an article on a given topic."""
    return f"Article on {topic}:\n\n{data}"

def format_article(article: str) -> str:
    """Format an article for publication."""
    return f"*** {article} ***"

def publish_article(article: str) -> str:
    """Simulate publishing an article."""
    return f"Published the following article:\n\n{article}"


math_agent = create_react_agent(
    model   =   model,
    tools   =   [add, multiply, divide],
    name    =   "math_expert",
    prompt  =   "You are a math expert specializing in calculations. Always use tools for ANY mathematical operations - addition, multiplication, etc. Never calculate manually. When given multiple numbers to add, use the add tool step by step."
)

research_agent = create_react_agent(
    model   =   model,
    tools   =   [web_search],
    name    =   "research_expert",
    prompt  =   "You are a world class researcher with access to web search, but never do summarizations. Do not do any math, or summarization or any writing - just facts retrieval."
)

writing_agent = create_react_agent(
    model   =   model,
    tools   =   [write_article, format_article],
    name    =   "writing_expert",
    prompt  =   "You are a writing expert specializing in article creation. Always use tools for ANY writing tasks - drafting, formatting, publishing, etc. Never write manually. Never do any research - you should have a list of facts first before you write. never do do any calculations. "
)

publishing_agent = create_react_agent(
    model   =   model,
    tools   =   [publish_article],
    name    =   "publishing_expert",
    prompt  =   "You are a publishing expert specializing in article publication. Always use tools for ANY publishing tasks - publishing articles, etc. Never publish manually.Never write manually. Never do any research - you should have a list of facts first before you write. never do do any calculations. "
)

# Compile and run
research_team = create_supervisor(
    [research_agent, math_agent],
    model=model,
    supervisor_name="research_supervisor",
    #parallel_tool_calls= True,
    #output_mode="full_history",
    prompt  =   "You orchistrate a research team. Always use tools for ANY research tasks and calculations - fact retrieval, data analysis, etc. Never do any writing manually."
).compile(name="research_team")

writing_team = create_supervisor(
    [writing_agent, publishing_agent],
    model=model,
    supervisor_name="writing_supervisor",
    prompt  =   "You orchistrate a writing team. Always use tools for ANY writing tasks - drafting, formatting, publishing, etc. Never do any research or calculations."
).compile(name="writing_team")

top_level_supervisor = create_supervisor(
    [research_team, writing_team],
    model=model,
    supervisor_name="top_level_supervisor",
    prompt  =   "You orchistrate two teams, always delegate to the right team based on the task at hand. never do math, research or write articles on your own"
).compile(name="top_level_supervisor")


# png_bytes = top_level_supervisor.get_graph().draw_mermaid_png()
# with open("./LangGraph_2_Agents/LG_2_multi_agent_2_hierarcical_supervisor.png", "wb") as f:
#     f.write(png_bytes)



result = top_level_supervisor.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is the headcount of the two largest FAANG companies in 2024, and what is the total of those two multiplied by 23.4 and again divided by 8.9? write an short article about the process of calculating this"
        }
    ]
})