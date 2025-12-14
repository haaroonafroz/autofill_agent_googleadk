"""
Tool for managing the RAG pipeline: embedding chunks and retrieving information.
Uses OpenAI Embeddings and Qdrant Vector Database.
"""

import os
from typing import List, Optional
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient, models

# Configuration Defaults
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "personal-collection")

class RAGManager:
    """Manages PDF chunk embedding, storage, and retrieval using Qdrant and OpenAI embeddings."""
    
    def __init__(self, 
                 collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Initializes the RAGManager.
        
        Args:
            collection_name: Name of the Qdrant collection to store/retrieve data.
        """
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.collection_name = collection_name
        
        if not all([self.qdrant_url, self.qdrant_api_key, self.openai_api_key]):
            print("Warning: Missing QDRANT_URL, QDRANT_API_KEY, or OPENAI_API_KEY in environment.")

        self.embedding_model = None
        self.vector_store = None
        self.client = None

        self._initialize_resources()

    def _initialize_resources(self):
        """Initializes the embedding model and Qdrant client."""
        try:
            print("Initializing OpenAI Embeddings...")
            self.embedding_model = OpenAIEmbeddings(api_key=self.openai_api_key)
            
            print(f"Connecting to Qdrant at {self.qdrant_url}...")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            print("RAG Resources initialized.")
        except Exception as e:
            print(f"Error initializing RAG resources: {e}")
            raise

    def _create_payload_indexes(self):
        """Creates indexes for faster filtering on metadata fields."""
        try:
            # existing user_id index
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.user_id", # LangChain often nests it
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            # New indexes for structured search (Markdown Headers)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.Header_1", # Note: LangChain nests metadata
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.Header_2",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Payload indexes created/updated.")
        except Exception as e:
            # Indexes might already exist, or 400 if strictly valid format required
            print(f"Index creation note: {e}")

    def initialize_vector_store(self, chunks: List[Document], user_id: str, force_recreate: bool = False):
        """
        Initializes the Qdrant vector store with user-specific chunks.
        """
        if not chunks:
            print("No chunks provided to initialize vector store.")
            return

        # Add user_id to metadata for EVERY chunk
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        try:
            print(f"Indexing {len(chunks)} chunks into Qdrant for user {user_id}...")
            
            # Use LangChain wrapper for convenience in ADDING documents (it handles embedding + upsert well)
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                force_recreate=force_recreate,
                vector_name="user-information" # Matches collection config
            )
            
            # Ensure indexes are set up after collection creation
            self._create_payload_indexes()
            
            print("Vector store successfully initialized and populated.")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def query_vector_store(self, query: str, user_id: str, k: int = 3) -> List[Document]:
        """
        Queries the vector store using raw Qdrant Client to avoid LangChain wrapper issues.
        """
        # 1. Embed query
        try:
            query_vector = self.embedding_model.embed_query(query)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

        # 2. Define Filter
        user_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id),
                ),
            ]
        )

        print(f"Querying Qdrant for user {user_id}: '{query}'")
        
        try:
            # 3. Execute Search via Client directly
            # This bypasses the AttributeError: 'QdrantClient' object has no attribute 'search'
            # which happens inside the LangChain wrapper due to version mismatch.
            response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="user-information",  # optional, only if using named vector
            query_filter=user_filter,
            with_payload=True,
            with_vectors=True,
            limit=k
            )

            # 4. Extract hits
            points = getattr(response, "points", None)
            if points is None:
                # Some versions also use .result
                points = getattr(response, "result", [])

            docs = []
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                content = payload.get("page_content", "")
                meta = payload.get("metadata", {})
                docs.append(Document(page_content=content, metadata=meta))

            print(f"Found {len(docs)} results.")
            return docs

        except Exception as e:
            print(f"Error during query: {e}")
            return []