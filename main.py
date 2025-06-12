from langchain_core.messages import HumanMessage
from app.langgraph_flow import run_workflow, logger_setup
import app.tools
import logging
import app.tools.rag.search
#from app.react_agent import build_react_agent, logger_setup

for module in [app.langgraph_flow, app.tools.rag.search]:
    module.logger_setup(logger_level=logging.INFO)


graph = run_workflow()

