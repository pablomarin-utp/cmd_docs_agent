from typing import Optional, Literal, List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.messages import BaseMessage

class RAGQueryInput(BaseModel):
    """Input schema for RAG (Retrieval-Augmented Generation) queries."""
    query: str = Field(..., description="Texto de consulta para buscar documentos relevantes")
    top_k: Optional[int] = Field(default=5, description="Número máximo de documentos a recuperar")
    collection: str = Field(..., description="Nombre de la colección de Qdrant para buscar documentos")

class RagSearchArgs(BaseModel):
    """Arguments for RAG search tool."""
    query: str
    collection: str
    top_k: int = 3

class PDFChunkerArgs(BaseModel):
    """Arguments for PDF chunking tool."""
    file_path: str = Field(..., description="Ruta del archivo PDF a procesar")
    max_pages: int = Field(default=15, description="Número máximo de páginas a procesar del PDF")
    max_tokens_per_chunk: int = Field(default=650, description="Número máximo de tokens por chunk")

class AddDocumentsArgs(BaseModel):
    """Arguments for adding documents to a collection."""
    collection_name: str = Field(..., description="Nombre de la colección en Qdrant")
    documents: List[str] = Field(..., description="Lista de documentos a agregar a la colección")