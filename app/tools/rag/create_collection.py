import logging, sys
from langchain.tools import tool, Tool
from typing import Dict, Any
from app.config import qdrant_client

logger = logging.getLogger(__name__)

def logger_setup():
    """
    Set up logger configuration for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode='a')
        ]
    )
    logging.info("logger setup complete.")

@tool
def create_collection(collection_name: str) -> Dict[str, Any]:
    """
    Create a new collection in Qdrant.
    
    Args:
        collection_name (str): Name of the collection to create.
    
    Returns:
        Dict[str, Any]: Result of the collection creation.
    """

    logger.info(f"Creating collection: {collection_name}")
    
    try:
        qdrant_client.create_collection(collection_name=collection_name)
        return {"result": f"Collection '{collection_name}' created successfully."}
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return {"error": str(e)}


create_collection_tool = Tool(
    name="create_collection",
    func=create_collection,
    description="Create a new collection in Qdrant. Provide the collection name as an argument.",
)


