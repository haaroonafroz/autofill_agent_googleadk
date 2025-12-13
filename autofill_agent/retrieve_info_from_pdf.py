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
DEFAULT_COLLECTION_NAME = "cv_data"

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
            # New indexes for structured search (Markdown Headers)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.Header 1", # Note: LangChain nests metadata
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.Header 2",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Payload indexes created/updated.")
        except Exception as e:
            # Indexes might already exist, which is fine
            print(f"Index creation note: {e}")

    def initialize_vector_store(self, chunks: List[Document], user_id: str, force_recreate: bool = False):
        """
        Initializes the Qdrant vector store with user-specific chunks.
        
        Args:
            chunks: A list of LangChain Document objects.
            user_id: The unique hash ID of the user.
            force_recreate: If True, recreates the collection.
        """
        if not chunks:
            print("No chunks provided to initialize vector store.")
            return

        # Add user_id to metadata for EVERY chunk
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        try:
            # Ensure indexes exist
            # Note: If force_recreate is True, we might wipe indexes if we delete collection, so recreate after.
            if force_recreate:
                print(f"Recreating collection '{self.collection_name}'...")
                # self.client.recreate_collection(...) # LangChain wrapper can handle this via force_recreate=True
            
            print(f"Indexing {len(chunks)} chunks into Qdrant for user {user_id}...")
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                force_recreate=force_recreate 
            )
            
            # Ensure indexes are set up after collection creation
            self._create_payload_indexes()
            
            print("Vector store successfully initialized and populated.")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def query_vector_store(self, query: str, user_id: str, k: int = 3) -> List[Document]:
        """
        Queries the vector store for relevant documents, filtered by user_id.
        """
        if self.vector_store is None:
            # Attempt to connect to existing store
            try:
                self.vector_store = Qdrant(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model
                )
            except Exception as e:
                print(f"Error connecting to existing vector store: {e}")
                return []

        # Define the filter to ISOLATE this user's data
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
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=user_filter
            )
            print(f"Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during query: {e}")
            return []
