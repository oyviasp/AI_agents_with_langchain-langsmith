from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate as RAGPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv
import math
import os

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#### RAG SETUP ####

# Load PDF Documents
pdf_files = [
    "docs_for_rag/impromptu.pdf",
    "docs_for_rag/GEP-June-2025.pdf"
]

all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    all_docs.extend(docs)
    print(f"Loaded {len(docs)} pages from {pdf_file}")

# Split with metadata preservation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

print("=== RAG SETUP ===")
print(f"Total loaded pages: {len(all_docs)}")
print(f"Created {len(splits)} text chunks")
print("=" * 40)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG System Prompt - STRICT about not making things up
rag_system_prompt = """You are a document-based question-answering assistant. You must ONLY answer based on the provided context.

CRITICAL INSTRUCTIONS:
- ONLY use information that is explicitly stated in the provided context
- If the answer is not clearly available in the context, say "I cannot find this information in the provided documents"
- Do NOT make assumptions or add information from your general knowledge
- Do NOT guess or infer beyond what is directly stated
- Be specific about what the documents actually say

If you can answer from the context, be concise and accurate. Always cite that your answer comes from the provided documents.

Context: {context}"""

rag_prompt = RAGPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("human", "{question}")
])

# Simple document formatting without page numbers
def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        # Simple formatting - just content with document number
        content = f"Document {i}: {doc.page_content}"
        formatted_docs.append(content)
    return "\n\n".join(formatted_docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

#### TOOL FUNCTIONS ####

@traceable(name="rag_pdf_search")
def rag_pdf_search(query: str) -> str:
    """Search information from PDF documents - impromptu speaking and economic reports"""
    try:
        # Get the answer from RAG chain
        result = rag_chain.invoke(query)
        
        # Show how many documents were consulted
        retrieved_docs = retriever.invoke(query)
        num_docs = len(retrieved_docs)
        
        return f"{result}\n\n(Consulted {num_docs} document sections)"
        
    except Exception as e:
        return f"Error searching documents: {str(e)}"

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

#### CREATE TOOLS ####

# RAG PDF tool
rag_tool = Tool(
    name="rag_pdf_search",
    description="Search for information from PDF documents including topics about impromptu speaking, public speaking, communication skills, AND economic data, financial reports, economic outlooks, and economic analysis. Use this when users ask about speaking techniques, presentation skills, communication strategies, OR economic questions, financial data, economic forecasts, market analysis, or economic policy topics.",
    func=rag_pdf_search
)

# Calculator tool
calc_tool = Tool(
    name="calculator", 
    description="A mathematical calculator tool for performing arithmetic operations, calculations, and mathematical functions. Use this when you need to solve math problems, calculate numbers, or perform any mathematical operations like addition, subtraction, multiplication, division, square roots, etc.",
    func=calculator
)

# Tools list
tools = [rag_tool, calc_tool]

print("=== AVAILABLE TOOLS ===")
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print("-" * 50)

#### CREATE AGENT ####

# Create explicit prompt template
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to specialized tools. 

Always choose the most appropriate tool for the user's question:
- Use rag_pdf_search for questions about impromptu speaking, public speaking, communication skills, OR economic topics, financial data, economic forecasts, market analysis
- Use calculator for mathematical problems, calculations, or numerical operations

Think carefully about which tool will best answer the user's question. If a question is about speaking, communication, presentation skills, economics, finance, or market data, use the RAG PDF search. If it's about numbers or math calculations, use the calculator."""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

print("=== AGENT PROMPT TEMPLATE ===")
print("System message focuses on tool selection for speaking/communication vs math")
print("Input variables:", agent_prompt.input_variables)
print("=" * 50)

# Create agent
agent = create_openai_functions_agent(llm, tools, agent_prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#### DEMO FUNCTION ####

@traceable(name="rag_multitool_agent_demo")
def run_rag_multitool_demo():
    """Demonstrates the RAG multitool agent with 3 focused question types"""
    questions = [
        "What is 25 * 15 + 88?",                                    # Should use Calculator
        "What is the global economic outlook for 2025?",           # Should use RAG (Economic)
        "How can I improve my impromptu speaking skills?"          # Should use RAG (Speaking)
    ]

    results = []
    for question in questions:
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print('='*80)
        
        print("🤖 Agent is working...")
        result = agent_executor.invoke({"input": question})
        print(f"✅ ANSWER: {result['output']}")
        print("-" * 80)
        
        results.append({
            "question": question,
            "answer": result['output']
        })
    
    return results

# Run the demo
if __name__ == "__main__":
    demo_results = run_rag_multitool_demo()
