from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langsmith import traceable
from dotenv import load_dotenv
import math

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Custom calculator function
@traceable(name="calculator_function")
def calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        # Only allow safe mathematical operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Create calculator tool
calc_tool = Tool(
    name="calculator",
    description="A mathematical calculator tool for performing arithmetic operations, calculations, and mathematical functions. Use this when you need to solve math problems, calculate numbers, or perform any mathematical operations like addition, subtraction, multiplication, division, square roots, etc.",
    func=calculator
)

# Initialize Wikipedia tool with custom description
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(),
    name="wikipedia",
    description="A comprehensive encyclopedia tool for searching factual information about people, places, events, concepts, and general knowledge. Use this when you need to find reliable information about historical facts, biographical data, geographical information, scientific concepts, or any general knowledge questions."
)

# Tools list with descriptions
tools = [wikipedia, calc_tool]

print("=== AVAILABLE TOOLS ===")
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print("-" * 50)

# Create explicit prompt template that you can see and modify
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to various tools. 

Always choose the most appropriate tool for the user's question:
- Use wikipedia for factual information, historical data, biographical info, or general knowledge
- Use calculator for mathematical problems, calculations, or numerical operations

Think carefully about which tool will best answer the user's question before making your choice."""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

print("=== CUSTOM PROMPT TEMPLATE ===")
print("System message: 'You are a helpful assistant'")
print("Input variables:", prompt.input_variables)
print("=" * 40)

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Create agent executor med mindre verbose output
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Run the agent with different types of questions
@traceable(name="multi_tool_agent_demo")
def run_agent_demo():
    """Demonstrates the multi-tool agent with different question types"""
    questions = [
        "What is the capital of Japan?",  # Should use Wikipedia
        "What is 25 * 47 + 123?",        # Should use Calculator
        "Who invented the telephone?"     # Should use Wikipedia
    ]

    results = []
    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print('='*60)
        
        print("🤖 Agent is working...")
        result = agent_executor.invoke({"input": question})
        print(f"✅ ANSWER: {result['output']}")
        print("-" * 60)
        
        results.append({
            "question": question,
            "answer": result['output']
        })
    
    return results

# Run the demo
if __name__ == "__main__":
    demo_results = run_agent_demo()