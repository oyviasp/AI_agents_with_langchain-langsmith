import streamlit as st
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
from datetime import datetime
import tempfile
import os
import shutil

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Assistant with RAG & Calculator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e8f4f8;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "initialization_attempted" not in st.session_state:
    st.session_state.initialization_attempted = False

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with PDF documents"""
    
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Check if we have a persistent vector database
        db_path = "./LangChain/chroma_db"
        if os.path.exists(db_path) and os.listdir(db_path):
            # Load existing vector database (fast)
            vectorstore = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            pdf_status = ["✅ Loaded existing vector database (fast startup)"]
        else:
            # Load PDF Documents (first time setup)
            pdf_files = [
                "./LangChain/docs_for_rag/impromptu.pdf",
                "./LangChain/docs_for_rag/GEP-June-2025.pdf"
            ]
            
            all_docs = []
            pdf_status = []
            
            for pdf_file in pdf_files:
                try:
                    if os.path.exists(pdf_file):
                        loader = PyPDFLoader(pdf_file)
                        docs = loader.load()
                        all_docs.extend(docs)
                        pdf_status.append(f"✅ Loaded {len(docs)} pages from {pdf_file}")
                    else:
                        pdf_status.append(f"❌ File not found: {pdf_file}")
                except Exception as e:
                    pdf_status.append(f"❌ Error loading {pdf_file}: {e}")
            
            if not all_docs:
                return None, None, None, ["❌ No PDF documents could be loaded!"]
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            # Create persistent vector database
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=OpenAIEmbeddings(),
                persist_directory=db_path
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            pdf_status.append(f"📚 Total loaded pages: {len(all_docs)}")
            pdf_status.append(f"🔤 Created {len(splits)} text chunks")
            pdf_status.append("💾 Vector database saved for future use")
        
        # RAG System Prompt
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

        # Simple document formatting
        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
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
        
        return llm, rag_chain, retriever, pdf_status
        
    except Exception as e:
        return None, None, None, [f"❌ Error initializing system: {e}"]

@st.cache_resource
def create_agent_system(_llm, _rag_chain, _retriever):
    """Create the agent system with tools"""
    
    # Tool functions
    @traceable(name="rag_pdf_search")
    def rag_pdf_search(query: str) -> str:
        """Search information from PDF documents - impromptu speaking and economic reports"""
        try:
            result = _rag_chain.invoke(query)
            retrieved_docs = _retriever.invoke(query)
            num_docs = len(retrieved_docs)
            return f"{result}\n\n(Consulted {num_docs} document sections)"
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    @traceable(name="calculator_function")
    def calculator(expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"

    # Create tools
    rag_tool = Tool(
        name="rag_pdf_search",
        description="Search for information from PDF documents including topics about impromptu speaking, public speaking, communication skills, AND economic data, financial reports, economic outlooks, and economic analysis. Use this when users ask about speaking techniques, presentation skills, communication strategies, OR economic questions, financial data, economic forecasts, market analysis, or economic policy topics.",
        func=rag_pdf_search
    )

    calc_tool = Tool(
        name="calculator", 
        description="A mathematical calculator tool for performing arithmetic operations, calculations, and mathematical functions. Use this when you need to solve math problems, calculate numbers, or perform any mathematical operations like addition, subtraction, multiplication, division, square roots, etc.",
        func=calculator
    )

    tools = [rag_tool, calc_tool]

    # Create agent prompt with memory
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
    agent = create_openai_functions_agent(_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return agent_executor, tools

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("AI Assistant with RAG & Calculator")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 System Information")
        
        # Initialize system if not done
        if not st.session_state.chat_initialized:
            # Show initialization status
            init_placeholder = st.empty()
            init_placeholder.info("🔄 Initializing AI Assistant... This may take a moment on first run.")
            
            llm, rag_chain, retriever, pdf_status = initialize_rag_system()
            
            if llm is not None:
                agent_executor, tools = create_agent_system(llm, rag_chain, retriever)
                st.session_state.agent_executor = agent_executor
                st.session_state.chat_initialized = True
                
                # Clear initialization message and show success
                init_placeholder.empty()
                st.success("✅ System initialized successfully!")
                
                with st.expander("📄 PDF Loading Status"):
                    for status in pdf_status:
                        st.write(status)
                
                with st.expander("🛠️ Available Tools"):
                    for tool in tools:
                        st.write(f"**{tool.name}**: {tool.description[:100]}...")
            else:
                init_placeholder.empty()
                st.error("Failed to initialize the system. Please check the PDF files.")
                return
        else:
            st.success("✅ System ready!")
            
            # Chat controls
            st.subheader("💬 Chat Controls")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("� Reset Vector Database"):
                import shutil
                db_path = "./Langchain/chroma_db"
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                st.session_state.chat_initialized = False
                st.session_state.agent_executor = None
                st.cache_resource.clear()
                st.info("Vector database reset. Please refresh the page.")
            
            if st.button("�💾 Download Chat History"):
                if st.session_state.messages:
                    chat_text = ""
                    for msg in st.session_state.messages:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        chat_text += f"{role}: {msg['content']}\n\n"
                    
                    st.download_button(
                        label="📥 Download as Text",
                        data=chat_text,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No chat history to download.")
            
            # Chat statistics
            if st.session_state.messages:
                st.metric("📊 Messages", len(st.session_state.messages))
                st.metric("🔄 Exchanges", len(st.session_state.messages) // 2)
    
    # Main chat area
    if st.session_state.chat_initialized:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about impromptu speaking, economics, or math calculations..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Convert messages to LangChain format for context
                        chat_history = []
                        for msg in st.session_state.messages[:-1]:  # Exclude current message
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            else:
                                chat_history.append(AIMessage(content=msg["content"]))
                        
                        # Get response from agent
                        response = st.session_state.agent_executor.invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        
                        assistant_response = response['output']
                        st.markdown(assistant_response)
                        
                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    else:
        st.info("⏳ Initializing the AI Assistant... Please wait.")
        
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - I remember our entire conversation and can understand context!
    - Ask about impromptu speaking, public speaking, or communication skills
    - Ask about economic data, forecasts, or financial analysis  
    - Ask me to perform mathematical calculations
    - You can say "I meant..." to clarify previous questions
    """)

if __name__ == "__main__":
    main()
