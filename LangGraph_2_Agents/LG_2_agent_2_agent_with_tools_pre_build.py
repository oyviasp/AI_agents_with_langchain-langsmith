from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.
        
    Use this tool when you need to perform multiplication or find the product of two numbers.
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
        
    Returns:
        The product of a and b
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.
    
    Use this tool when you need to perform addition or sum two numbers.
    
    Args:
        a: The first number to add
        b: The second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide one number by another.
    
    Use this tool when you need to perform division. Note that this returns a float result.
    Warning: Division by zero will raise an error.
    
    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by) - must not be zero
        
    Returns:
        The quotient of a divided by b as a float
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# Nodes
# Pass in:
# (1) the augmented LLM with tools
# (2) the tools list (which is used to create the tool node)
pre_built_agent = create_react_agent(llm, tools=tools)


# Invoke
messages = [HumanMessage(content="Add 3 and 4 and divide by 2")]
messages = pre_built_agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

# Show workflow and save to file
png_bytes = pre_built_agent.get_graph().draw_mermaid_png()
with open("./LangGraph_2_Agents/LG_2_agent_2_agent_with_tools_pre_build.png", "wb") as f:
    f.write(png_bytes)
