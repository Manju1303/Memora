"""
Core Agent Module
Main AI agent with memory-augmented generation capabilities.
Implements RAG (Retrieval-Augmented Generation) pattern.
"""
from typing import Optional, Dict, Generator
import sys
sys.path.append('..')

from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .memory_manager import MemoryManager
from .prompts import get_memory_aware_prompt, SYSTEM_PROMPT
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS,
    DEFAULT_USER_ID
)


class MemoryAgent:
    """
    AI Agent with Long-term Memory Capabilities.
    
    Core Algorithm: Retrieval-Augmented Generation (RAG)
    
    Flow:
    1. User sends query
    2. Agent retrieves relevant memories from vector DB
    3. Agent combines memories with recent context
    4. LLM generates response using augmented context
    5. Conversation is stored for future reference
    
    Features:
    - Persistent memory across sessions
    - Semantic memory retrieval
    - Personalized responses
    - Context-aware conversations
    """
    
    def __init__(self, user_id: str = DEFAULT_USER_ID):
        self.user_id = user_id
        self.memory = MemoryManager(user_id=user_id)
        self._setup_llm()
    
    def _setup_llm(self) -> None:
        """Initialize the Ollama LLM."""
        try:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=MAX_TOKENS
            )
            self._llm_available = True
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            self._llm_available = False
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.
        
        This is the main entry point implementing RAG:
        1. Store user message in short-term memory
        2. Retrieve relevant long-term memories
        3. Construct memory-aware prompt
        4. Generate response with LLM
        5. Store response in memory
        6. Return response
        
        Args:
            user_message: The user's input text
        
        Returns:
            The assistant's response
        """
        # Step 1: Store user message
        self.memory.add_user_message(user_message)
        
        # Step 2 & 3: Get context and build prompt
        short_term, long_term = self.memory.get_context_for_query(user_message)
        prompt = get_memory_aware_prompt(
            user_message=user_message,
            short_term_context=short_term,
            long_term_memories=long_term
        )
        
        # Step 4: Generate response
        if not self._llm_available:
            response = self._fallback_response(user_message)
        else:
            try:
                response = self.llm.invoke(prompt)
                response = response.strip()
            except Exception as e:
                print(f"LLM error: {e}")
                response = self._fallback_response(user_message)
        
        # Step 5: Store response
        self.memory.add_assistant_message(response)
        
        # Periodic summarization check
        self.memory.periodic_summarization()
        
        return response
    
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream a response token by token.
        
        Args:
            user_message: The user's input text
        
        Yields:
            Response tokens as they're generated
        """
        # Store user message
        self.memory.add_user_message(user_message)
        
        # Get context and build prompt
        short_term, long_term = self.memory.get_context_for_query(user_message)
        prompt = get_memory_aware_prompt(
            user_message=user_message,
            short_term_context=short_term,
            long_term_memories=long_term
        )
        
        full_response = ""
        
        if not self._llm_available:
            response = self._fallback_response(user_message)
            yield response
            full_response = response
        else:
            try:
                # Stream response
                for chunk in self.llm.stream(prompt):
                    yield chunk
                    full_response += chunk
            except Exception as e:
                error_msg = f"Error: {e}"
                yield error_msg
                full_response = error_msg
        
        # Store complete response
        self.memory.add_assistant_message(full_response.strip())
        self.memory.periodic_summarization()
    
    def _fallback_response(self, user_message: str) -> str:
        """Fallback response when LLM is not available."""
        return (
            "I apologize, but I'm having trouble connecting to my language model. "
            "Please make sure Ollama is running:\n"
            "1. Install Ollama from https://ollama.ai\n"
            "2. Run: ollama pull mistral\n"
            "3. Run: ollama serve\n\n"
            f"Your message was: {user_message}"
        )
    
    def remember(self, fact: str) -> str:
        """Explicitly remember a fact."""
        memory_id = self.memory.store_memory(fact, memory_type="explicit")
        return f"I'll remember that: {fact}"
    
    def recall(self, query: str, n_results: int = 5) -> str:
        """Search memories related to a query."""
        memories = self.memory.search_memories(query, n_results)
        
        if not memories:
            return "I don't have any memories related to that."
        
        result = ["Here's what I remember:"]
        for i, mem in enumerate(memories, 1):
            score = mem.get("relevance_score", 0)
            result.append(f"{i}. [{score:.0%}] {mem['content']}")
        
        return "\n".join(result)
    
    def new_conversation(self) -> None:
        """Start a new conversation (clears short-term, preserves long-term)."""
        self.memory.clear_short_term()
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        return self.memory.get_stats()
    
    def get_all_memories(self):
        """Get all stored memories for visualization."""
        return self.memory.get_all_memories()
    
    def clear_all_memories(self) -> None:
        """Clear all memories (use with caution!)."""
        self.memory.clear_all_memories()
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to chat."""
        return self._llm_available
    
    def get_status(self) -> Dict:
        """Get agent status information."""
        return {
            "llm_available": self._llm_available,
            "model": OLLAMA_MODEL,
            "user_id": self.user_id,
            "memory_stats": self.get_memory_stats()
        }
    
    def __repr__(self) -> str:
        status = "ready" if self._llm_available else "offline"
        return f"MemoryAgent(user={self.user_id}, status={status})"
