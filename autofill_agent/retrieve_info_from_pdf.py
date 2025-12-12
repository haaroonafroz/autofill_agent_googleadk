"""
Tool for managing the RAG pipeline: embedding chunks and retrieving information.
Uses OpenAI Embeddings and Qdrant Vector Database.
"""

import os
from typing import List, Optional
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

# Configuration Defaults
DEFAULT_COLLECTION_NAME = "personal-collection"

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

        self.embedding_model = 'text-embedding-3-small'
        self.vector_store = None
        self.client = None

        self._initialize_resources()

    def _initialize_resources(self):
        """Initializes the embedding model and Qdrant client."""
        try:
            print("Initializing OpenAI Embeddings...")
            self.embedding_model = OpenAIEmbeddings(api_key=self.openai_api_key, model=self.embedding_model)
            
            print(f"Connecting to Qdrant at {self.qdrant_url}...")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            print("RAG Resources initialized.")
        except Exception as e:
            print(f"Error initializing RAG resources: {e}")
            raise

    def initialize_vector_store(self, chunks: List[Document], force_recreate: bool = False):
        """
        Initializes the Qdrant vector store with the given document chunks.
        
        Args:
            chunks: A list of LangChain Document objects.
            force_recreate: If True, recreates the collection.
        """
        if not chunks:
            print("No chunks provided to initialize vector store.")
            return

        try:
            if force_recreate:
                print(f"Recreating collection '{self.collection_name}'...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.client.get_collections().collections[0].config.params.vectors if self.client.get_collections().collections else {} # Simplified check, usually handled by Qdrant langchain wrapper or handled manually if strict config needed
                    # Note: LangChain's from_documents usually handles collection creation if not exists.
                    # For force_recreate with LangChain, we might just want to let LangChain handle it or delete explicitly.
                )
                # Simpler approach: let LangChain manage it, but we can delete if forced
                self.client.delete_collection(collection_name=self.collection_name)
            
            print(f"Indexing {len(chunks)} chunks into Qdrant...")
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                force_recreate=force_recreate # LangChain Qdrant wrapper supports this
            )
            print("Vector store successfully initialized and populated.")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def query_vector_store(self, query: str, k: int = 3) -> List[Document]:
        """
        Queries the vector store for relevant documents.
        """
        if self.vector_store is None:
            # Attempt to connect to existing store if not explicitly initialized with chunks this session
            try:
                self.vector_store = Qdrant(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model
                )
            except Exception as e:
                print(f"Error connecting to existing vector store: {e}")
                return []

        print(f"Querying Qdrant for: '{query}'")
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during query: {e}")
            return []
