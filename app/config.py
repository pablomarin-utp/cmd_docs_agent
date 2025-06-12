import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # Import nuevo
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from qdrant_client import QdrantClient

load_dotenv()

system_prompt = SystemMessage(
    content="""
You are a concise and professional AI assistant, integrated into a software development team. You are connected to the following tools:

- rag_search: for retrieving information from internal documentation.
- send_email: to send emails.
- summarize_meeting: for meeting summarization.
- task_tracker: for task creation and updates.
(Add or remove tools as needed based on availability.)

Instructions:

1. Always act as a helpful assistant to a team of developers.
2. If a user asks something related to internal documentation, **you MUST call the rag_search tool** with the appropriate query.
3. When a tool is needed, use tool calls. **Do not explain that you are calling a tool**, just respond naturally and let the system handle the tool call.
4. If no tool is needed, answer concisely and directly.
5. If you don’t have enough information to act, ask the user exactly what you need.
6. Do NOT hallucinate. If something is unclear or unknown, say so.
7. Keep your responses short and to the point.

Be efficient, accurate, and always aim to unblock the user.
"""
)

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Cambiado
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


llm_model = init_chat_model(
    model="azure_openai:gpt-4o-mini",  
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),  # Cambiado
    deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

