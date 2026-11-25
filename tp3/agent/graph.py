from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
import config
from agent.tool import retriever_tool
from langchain_openai import ChatOpenAI

tools = [retriever_tool]
llm = ChatOpenAI(
    model=config.MODELE_GENERATION,            # e.g. "openai/gpt-oss-120b"
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant answering questions about a PDF document. "
    "Use retriever_tool to find information before answering.\n"
    "Rules:\n"
    "- Break the user's question into small, independent sub-questions.\n"
    "- Call retriever_tool once per sub-question, with a short keyword-style query.\n"
    "  Example: 'what is attention and how was it discovered?' -> "
    "'attention mechanism definition' and 'origin of attention mechanism history'.\n"
    "- The tool searches both text passages and images/figures. "
    "If the user asks to see a figure or diagram, say so in the query "
    "(e.g. 'Transformer architecture diagram').\n"
    "- If the user provides an image path, pass it in the image_path argument.\n"
    "- If results are insufficient, search again with a different query.\n"
    "- Do not answer from memory. Cite sources as [pdf p.X] and mention "
    "image files when you use them."
))

def agent_node(state: MessagesState):          # MIND
    return {"messages": [llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))     # BODY

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else END
builder.add_edge("tools", "agent")             # Observation goes back to the brain

graph = builder.compile()

def ask(question: str, max_steps: int = config.MAX_STEPS):
    """Run the agent and print each Reasoning / Action / Observation step."""
    final = None
    for state in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": max_steps},
        stream_mode="values",
    ):
        state["messages"][-1].pretty_print()
        final = state
    return final["messages"][-1].content