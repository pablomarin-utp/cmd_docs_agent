import logging, sys
from typing import Annotated, Literal
from io import BytesIO

# from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from PIL import Image
from typing_extensions import TypedDict
from app.config import system_prompt, llm_model
from app.tools.rag.search import rag_tool
from app.tools.rag.create_collection import create_collection_tool
from app.tools.rag.add_documents import add_documents_tool
from app.tools.rag.pdf_chunker import pdf_chunker_tool 

logger = logging.getLogger(__name__)

def logger_setup(logger_level=logging.DEBUG):
    """
    Set up logger configuration for the application.
    """
    logging.basicConfig(
        level=logger_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", mode='a')
        ]
    )
    logging.info("Logger setup complete.")



class AgentState(TypedDict):
    user_id: str
    messages: Annotated[list, add_messages]

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        logger.info(f"Message received: {message.content[:200]}...")
        message.pretty_print()

# Usa herramientas importadas
tools = [rag_tool, create_collection_tool, add_documents_tool, pdf_chunker_tool]


def run_workflow():
    logger.info("Initializing workflow")

    tool_node = ToolNode(tools)

    model = llm_model.bind_tools(tools)

    logger.info(f"Initialized model and loaded {len(tools)} tools")

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Define the function that calls the model
    def call_model(state: AgentState):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    config = {"configurable": {"thread_id": 1}}
    logger.info(f"Set configuration: {config}")

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).show()

    logger.info("Created workflow agent graph")

    logger.info("Starting conversation with initial prompt")
    inputs = {"messages": [("user", system_prompt.content)]}
    print_stream(graph.stream(inputs, config, stream_mode="values"))

    # Start chatbot
    logger.info("Entering interactive chat loop")
    while True:
        user_input = input("User: ")
        logger.info(f"Received user input: {user_input[:200]}...")
        inputs = {"messages": [("user", user_input)]}
        print_stream(graph.stream(inputs, config, stream_mode="values"))