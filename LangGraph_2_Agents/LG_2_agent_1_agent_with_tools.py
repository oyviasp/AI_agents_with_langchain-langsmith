from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from typing_extensions import  Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

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
def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or not"""
    
    system_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
    messages_to_send = [system_message] + state["messages"]
    llm_response = llm_with_tools.invoke(messages_to_send)
    result = {"messages": [llm_response]}
    
    return result


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "environment"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "environment": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Invoke
messages = [HumanMessage(content="Add 3 and 4 and multiply it by 4")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

# Show workflow and save to file
png_bytes = agent.get_graph().draw_mermaid_png()
with open("./LangGraph_2_Agents/LG_2_agent_1_agent_with_tools.png", "wb") as f:
    f.write(png_bytes)






