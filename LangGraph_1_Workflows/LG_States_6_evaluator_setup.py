from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Graph state
class State(TypedDict):
    joke:           str
    topic:          str
    feedback:       str
    funny_or_not:   str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade:      Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
    feedback:   str                           = Field(description="If the joke is not funny, provide feedback on how to improve it.")



# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)


# Nodes
def llm_call_generator(state: State) -> dict[str, str]:
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State) -> dict[str, str]:
    """LLM evaluates the joke"""

    evaluated_joke = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": evaluated_joke.grade, "feedback": evaluated_joke.feedback}


# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State) -> Literal["Accepted", "Rejected + Feedback"]:
    """Route back to joke generator or end based upon feedback from the evaluator"""
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges("llm_call_evaluator", route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show workflow and save to file
png_bytes = optimizer_workflow.get_graph().draw_mermaid_png()
with open("./LangGraph_1_Workflows/LG_States_6_evaluator_setup.png", "wb") as f:
    f.write(png_bytes)

# Invoke
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])






