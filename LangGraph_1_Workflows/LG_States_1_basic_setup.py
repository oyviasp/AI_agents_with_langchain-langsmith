import os
import getpass
from dotenv import load_dotenv                      # load .env variables
from langsmith import traceable                     # trace in LangSmith
from langchain_openai import ChatOpenAI             # OpenAI as llm
from pydantic import BaseModel, Field               # Schema for structured output

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define a schema for structured output
class SearchQuery(BaseModel):
    search_query:   str = Field(None, description="Query that is optimized web search.")
    justification:  str = Field(None, description="Why this query is relevant to the user's request.")

# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

# Define a tool
@traceable(name="multiply_function")
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_toolsr.invoke("What is 2 times 3?")


# Get the tool call
msg.tool_calls