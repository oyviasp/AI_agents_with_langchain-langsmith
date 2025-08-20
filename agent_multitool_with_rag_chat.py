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
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
from dotenv import load_dotenv
import math
import os
from datetime import datetime

load_dotenv()

print("🚀 Loading AI Assistant with RAG capabilities...")
print("=" * 60)

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
    try:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"✅ Loaded {len(docs)} pages from {pdf_file}")
    except Exception as e:
        print(f"❌ Error loading {pdf_file}: {e}")

# Split with metadata preservation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

print(f"📚 Total loaded pages: {len(all_docs)}")
print(f"🔤 Created {len(splits)} text chunks")
print("=" * 60)

# Embed
print("🔄 Creating vector database...")
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Vector database ready!")

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

#### CREATE AGENT WITH MEMORY ####

# Create explicit prompt template with chat history
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to specialized tools and the ability to remember our entire conversation. 

You have access to:
- rag_pdf_search: For questions about impromptu speaking, public speaking, communication skills, OR economic topics, financial data, economic forecasts, market analysis
- calculator: For mathematical problems, calculations, or numerical operations

IMPORTANT CONVERSATION MEMORY:
- Always review the chat history to understand context and references
- If a user asks for clarification or correction (like "I meant 2025" after asking about 2024), understand they are referring to their previous question
- Use conversational context to interpret ambiguous requests
- Reference previous questions and answers when relevant
- If a user says "I meant..." or "I was asking about..." refer back to the previous topic

TOOL USAGE:
- If using rag_pdf_search, only provide information that is explicitly found in the documents
- Do not make up or infer information beyond what is clearly stated
- Be conversational and acknowledge when you're building on previous parts of our conversation"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, agent_prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#### CHAT INTERFACE ####

class ChatBot:
    def __init__(self):
        self.chat_history = []
        self.session_start = datetime.now()
        
    def save_chat_history(self):
        """Save chat history to a file"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Chat Session Started: {self.session_start}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, message in enumerate(self.chat_history, 1):
                if isinstance(message, HumanMessage):
                    f.write(f"[{i//2 + 1}] USER: {message.content}\n\n")
                elif isinstance(message, AIMessage):
                    f.write(f"[{i//2 + 1}] ASSISTANT: {message.content}\n\n")
                    f.write("-" * 60 + "\n\n")
        
        print(f"💾 Chat history saved to: {filename}")
    
    def display_welcome(self):
        print("🤖 AI ASSISTANT WITH RAG & CALCULATOR")
        print("=" * 60)
        print("📚 Available capabilities:")
        print("  • Impromptu speaking & public speaking advice")
        print("  • Economic data & financial analysis") 
        print("  • Mathematical calculations")
        print("  • Conversational memory & context awareness")
        print("\n💡 Tips:")
        print("  • I remember our entire conversation!")
        print("  • You can say 'I meant...' to clarify previous questions")
        print("  • I understand references to earlier topics")
        print("  • Type 'quit', 'exit', or 'bye' to end chat")
        print("  • Type 'history' to see conversation count")
        print("  • Type 'save' to save chat history")
        print("=" * 60)
    
    @traceable(name="chat_interaction")
    def chat_loop(self):
        """Main chat loop with memory"""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for chatting! Saving your conversation...")
                    self.save_chat_history()
                    print("🔚 Goodbye!")
                    break
                
                if user_input.lower() == 'history':
                    print(f"\n📊 Conversation: {len(self.chat_history)//2} exchanges")
                    continue
                
                if user_input.lower() == 'save':
                    self.save_chat_history()
                    continue
                
                if not user_input:
                    print("💭 Please enter a question or message.")
                    continue
                
                # Add user message to history
                self.chat_history.append(HumanMessage(content=user_input))
                
                # Get AI response with chat history
                print(f"\n🤖 Assistant (remembering {len(self.chat_history)//2} previous exchanges): ", end="")
                try:
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": self.chat_history[:-1]  # Exclude current message
                    })
                    
                    ai_response = response['output']
                    print(ai_response)
                    
                    # Add AI response to history
                    self.chat_history.append(AIMessage(content=ai_response))
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    print(error_msg)
                    self.chat_history.append(AIMessage(content=error_msg))
                    
            except KeyboardInterrupt:
                print("\n\n⏹️  Chat interrupted. Saving conversation...")
                self.save_chat_history()
                print("👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue

# Run the chatbot
if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.chat_loop()
