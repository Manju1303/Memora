"""
Long-term Memory Module
Implements persistent vector-based memory using ChromaDB.
Enables semantic search for memory retrieval.
"""
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('..')
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    MEMORY_RETRIEVAL_COUNT
)


class LongTermMemory:
    """
    Persistent vector-based memory using ChromaDB.
    
    Algorithms used:
    1. Embedding Generation - Sentence Transformers (MiniLM)
    2. Vector Similarity Search - Cosine Similarity / ANN
    3. Semantic Retrieval - Find relevant past memories by meaning
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self._setup_storage()
        self._setup_embedding_model()
    
    def _setup_storage(self) -> None:
        """Initialize ChromaDB with persistent storage."""
        # Ensure directory exists
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Get or create collection for this user
        collection_name = f"{CHROMA_COLLECTION_NAME}_{self.user_id}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def _setup_embedding_model(self) -> None:
        """Initialize the sentence transformer for embeddings."""
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        memory_type: str = "conversation"
    ) -> str:
        """
        Store a new memory in long-term storage.
        
        Args:
            content: The text content to remember
            metadata: Optional additional metadata
            memory_type: Type of memory (conversation, fact, preference)
        
        Returns:
            The ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Prepare metadata
        full_metadata = {
            "user_id": self.user_id,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "content_length": len(content)
        }
        if metadata:
            full_metadata.update(metadata)
        
        # Generate embedding and store
        embedding = self._generate_embedding(content)
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[full_metadata]
        )
        
        return memory_id
    
    def retrieve_memories(
        self,
        query: str,
        n_results: int = MEMORY_RETRIEVAL_COUNT,
        memory_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant memories using semantic search.
        
        Algorithm: Vector Similarity Search
        1. Convert query to embedding
        2. Find nearest neighbors in vector space
        3. Return top-k most similar memories
        
        Args:
            query: The search query
            n_results: Number of memories to retrieve
            memory_type: Optional filter by memory type
        
        Returns:
            List of relevant memories with content and metadata
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build where filter if memory_type specified
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memory = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "relevance_score": 1 - results["distances"][0][i] if results["distances"] else 0
                }
                memories.append(memory)
        
        return memories
    
    def get_formatted_memories(self, query: str, n_results: int = 3) -> str:
        """Get memories formatted as context for the LLM."""
        memories = self.retrieve_memories(query, n_results)
        
        if not memories:
            return "No relevant past memories found."
        
        formatted = ["Relevant memories from past conversations:"]
        for i, mem in enumerate(memories, 1):
            score = mem.get("relevance_score", 0)
            formatted.append(f"{i}. [{score:.0%} relevant] {mem['content']}")
        
        return "\n".join(formatted)
    
    def store_user_preference(self, preference: str) -> str:
        """Store a user preference for personalization."""
        return self.store_memory(
            content=preference,
            memory_type="preference",
            metadata={"category": "user_preference"}
        )
    
    def store_important_fact(self, fact: str) -> str:
        """Store an important fact about the user or context."""
        return self.store_memory(
            content=fact,
            memory_type="fact",
            metadata={"category": "important_fact"}
        )
    
    def get_all_memories(self, limit: int = 100) -> List[Dict]:
        """Get all stored memories (for debugging/visualization)."""
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memory = {
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                }
                memories.append(memory)
        
        return memories
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def clear_all_memories(self) -> None:
        """Clear all memories for this user (use with caution!)."""
        # Delete and recreate collection
        collection_name = f"{CHROMA_COLLECTION_NAME}_{self.user_id}"
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_memory_count(self) -> int:
        """Get the total number of stored memories."""
        return self.collection.count()
    
    def __repr__(self) -> str:
        return f"LongTermMemory(user={self.user_id}, memories={self.get_memory_count()})"
