from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_community.tools import BraveSearch

load_dotenv()

# Wrap OpenAI klienten for LangSmith
openai_client = wrap_openai(OpenAI())

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

@traceable(name="router")
def router(query: str):
    system_message = "Based on the query, decide if it is a web search(return ONLY: web) or a RAG question(return ONLY: rag). Return only the word 'web' or 'rag', nothing else."
    tool_to_use = llm.invoke(system_message + " Query: " + query)
    return tool_to_use.content.strip().lower()

def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

@traceable(name="rag_chain")
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )


@traceable(name="web_search")
def web_search(query: str):
    import os
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    search_engine = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
    search_results = search_engine.run(query)

    system_message = """Answer the users question using only the provided information below:
    {docs}
    User question:
    {question}
    """.format(docs=search_results, question=query)
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        model="gpt-4o-mini",
    )


query= "Search information about empire state building in the US"
response = web_search(query)
print(response.choices[0].message.content)