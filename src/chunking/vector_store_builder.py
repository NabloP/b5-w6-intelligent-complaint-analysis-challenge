# ------------------------------------------------------------------------------
# 📄 VectorStoreBuilder Module for B5W6 – Intelligent Complaint Analysis
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Builds and persists a ChromaDB vector store using precomputed text embeddings
#   and associated metadata for efficient semantic search in RAG pipelines.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import os  # For directory handling

# ---------------------------
# Third-Party Imports
# ---------------------------
from langchain.vectorstores import Chroma  # ChromaDB for vector storage
from langchain.schema.embeddings import Embeddings  # Compliance (unused but required)

# ---------------------------
# VectorStoreBuilder Class
# ---------------------------


class VectorStoreBuilder:
    """
    A class to build and persist a ChromaDB vector store using precomputed embeddings and associated metadata.
    """

    def __init__(
        self,
        persist_directory: str = "vector_store/chroma_db",
        collection_name: str = "complaint_chunks",
    ):
        """
        Initializes the vector store builder.

        Args:
            persist_directory (str): Path to save the ChromaDB vector store.
            collection_name (str): Optional name for the Chroma collection.
        """
        try:
            self.persist_directory = (
                persist_directory  # Save location for the vector store
            )
            self.collection_name = collection_name  # Logical collection name

            # ✅ Create the directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize VectorStoreBuilder: {e}")

    def build_chroma_store(self, documents: list, embeddings, metadatas: list):
        """
        Builds and persists a ChromaDB vector store using precomputed embeddings and metadata.

        Args:
            documents (list): List of chunk texts.
            embeddings (np.ndarray): Precomputed embeddings for the chunks.
            metadatas (list): List of metadata dictionaries.

        Returns:
            Chroma: Persisted ChromaDB vector store instance or None on failure.
        """
        try:
            if not documents or embeddings is None or not metadatas:
                raise ValueError(
                    "❌ One or more inputs are empty or invalid for vector store creation."
                )

            # ✅ Create the ChromaDB vector store from embeddings
            vector_store = Chroma.from_embeddings(
                texts=documents,  # Input texts
                embeddings=embeddings,  # Precomputed dense vectors
                metadatas=metadatas,  # Metadata for each chunk
                collection_name=self.collection_name,  # Name for collection
                persist_directory=self.persist_directory,  # Persistence directory
            )

            vector_store.persist()  # Persist index to disk

            print(
                f"✅ ChromaDB vector store created successfully: {len(documents):,} chunks indexed."
            )
            print(f"📁 Stored at: {self.persist_directory}")

            return vector_store  # Return vector store instance

        except Exception as e:
            print(f"❌ Failed to build ChromaDB vector store: {e}")
            return None  # Return None safely on failure
