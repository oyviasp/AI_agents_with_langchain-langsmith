#RAG PDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#### INDEXING ####

# Load PDF Document
loader = PyPDFLoader("./LangChain/docs_for_rag/impromptu.pdf")
docs = loader.load()

# Split with metadata preservation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Check what metadata is automatically available
print("=== AUTOMATIC METADATA EXAMPLE ===")
if splits:
    print(f"Sample metadata: {splits[0].metadata}")
    print(f"Available keys: {list(splits[0].metadata.keys())}")
print("=" * 50)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#### RETRIEVAL and GENERATION ####

# System Prompt
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Always include references to the source chunks when you use information from them.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

print("=== SYSTEM PROMPT ===")
print(system_prompt)
print("=" * 50)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing with automatic PDF metadata
def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        # Use automatic metadata from PyPDFLoader
        page_label = doc.metadata.get('page_label', 'Unknown')
        page_num = doc.metadata.get('page', 'Unknown')
        
        # Create readable reference using the actual page label from PDF
        if page_label != 'Unknown':
            chunk_ref = f"Page {page_label}"
        elif page_num != 'Unknown':
            chunk_ref = f"Page {page_num + 1}"
        else:
            chunk_ref = f"Chunk {i}"
            
        content = f"[{chunk_ref}]: {doc.page_content}"
        formatted_docs.append(content)
    return "\n\n".join(formatted_docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
question = "What is impromptu speaking and how can it be improved?"

# Show retrieved chunks with PDF metadata
print("=== RETRIEVED CHUNKS ===")
retrieved_docs = retriever.invoke(question)
for i, doc in enumerate(retrieved_docs, 1):
    page_label = doc.metadata.get('page_label', 'Unknown')
    page_num = doc.metadata.get('page', 'Unknown')
    
    if page_label != 'Unknown':
        page_ref = f"Page {page_label}"
    elif page_num != 'Unknown':
        page_ref = f"Page {page_num + 1}"
    else:
        page_ref = f"Chunk {i}"
    
    print(f"{i}. {page_ref}")
print("=" * 50)

result = rag_chain.invoke(question)
print("=== RAG RESULT ===")
print(result)