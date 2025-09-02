from dotenv import load_dotenv                      # load .env variables
from langchain_openai import ChatOpenAI             # OpenAI as llm
from typing_extensions import TypedDict
from typing import Annotated, List
import operator
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Schema for structured output to use in planning
class Section(BaseModel):
    name:           str = Field(description="Name for this section of the report.")
    description:    str = Field(description="Brief overview of the main topics and concepts to be covered in this section.")

class Sections(BaseModel):
    sections:       List[Section] = Field(description="Sections of the report.")

# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

# Graph state
class OrchestratorState(TypedDict):
    topic:              str                             # Report topic
    sections:           list[Section]                   # List of report sections
    completed_sections: Annotated[list, operator.add]   # All workers write to this key in parallel
    final_report:       str                             # Final report


# Worker state
class WorkerState(TypedDict):
    section:            Section                         # Individual section (won't conflict)
    completed_sections: Annotated[list, operator.add]   # Shared state


# Nodes
def orchestrator(state: OrchestratorState) -> OrchestratorState:
    """Orchestrator that generates a plan for the report"""
    # Generate queries
    system_message      = "Generate a plan for the report."
    human_message       = f"Here is the report topic: {state['topic']}"
    report_sections     = planner.invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])
    state["sections"]   = report_sections.sections
    return state  

def llm_call(state: WorkerState) -> dict[str, list[str]]:
    """Worker writes a section of the report"""
    # Generate section
    system_message  = "Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
    human_message   = f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
    section = llm.invoke([SystemMessage(content=system_message), HumanMessage(content=human_message)])

    # Return only the completed section for this worker as a dictionary
    return {"completed_sections": [section.content]}

def synthesizer(state: OrchestratorState) -> dict[str, str]:
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: OrchestratorState):
    """Assign a worker to each section in the plan"""
    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# Build workflow
orchestrator_worker_builder = StateGraph(OrchestratorState)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Show workflow and save to file
png_bytes = orchestrator_worker.get_graph().draw_mermaid_png()
with open("./LangGraph_1_Workflows/LG_States_5_orchestrator_setup.png", "wb") as f:
    f.write(png_bytes)

# Invoke
state = orchestrator_worker.invoke({
    "topic": "Create a report on LLM scaling laws",
    "completed_sections": []  # Initialize empty list
})

print("=== FINAL REPORT ===")
print(state["final_report"])



