"""
Core Agent Module
Main AI agent with memory-augmented generation capabilities.
Implements RAG (Retrieval-Augmented Generation) pattern.
Supports both local (Ollama) and cloud (Hugging Face - FREE) LLMs.
"""
from typing import Optional, Dict, Generator
import os
import sys
sys.path.append('..')

import streamlit as st

from .memory_manager import MemoryManager
from .prompts import get_memory_aware_prompt, SYSTEM_PROMPT
from config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    HF_TOKEN,
    HF_MODEL,
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
    - Supports local (Ollama) and cloud (Hugging Face - FREE) LLMs
    """
    
    def __init__(self, user_id: str = DEFAULT_USER_ID):
        self.user_id = user_id
        self.memory = MemoryManager(user_id=user_id)
        self.provider = self._get_provider()
        self._setup_llm()
    
    def _get_provider(self) -> str:
        """Determine which LLM provider to use."""
        # Auto-detect Hugging Face Spaces
        if os.getenv('SPACE_ID'):
            return "huggingface"
        
        # Check Streamlit secrets (for Streamlit Cloud deployment)
        try:
            if hasattr(st, 'secrets'):
                if 'HF_TOKEN' in st.secrets and st.secrets['HF_TOKEN']:
                    return "huggingface"
                if 'LLM_PROVIDER' in st.secrets:
                    return st.secrets['LLM_PROVIDER']
        except:
            pass
        
        # Check environment variables
        if os.getenv('HF_TOKEN'):
            return "huggingface"
        
        return LLM_PROVIDER
    
    def _get_hf_token(self) -> str:
        """Get Hugging Face token from secrets or environment."""
        try:
            if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
                return st.secrets['HF_TOKEN']
        except:
            pass
        return os.getenv('HF_TOKEN', HF_TOKEN)
    
    def _setup_llm(self) -> None:
        """Initialize the LLM based on provider."""
        self._llm_available = False
        self.model_name = ""
        
        if self.provider == "huggingface":
            self._setup_huggingface()
        else:
            self._setup_ollama()
    
    def _setup_huggingface(self) -> None:
        """Initialize Hugging Face cloud LLM (FREE)."""
        try:
            from huggingface_hub import InferenceClient
            
            token = self._get_hf_token()
            is_hf_space = os.getenv('SPACE_ID') is not None
            
            # On HF Spaces, token is optional (uses built-in inference)
            if not token and not is_hf_space:
                print("Warning: HF_TOKEN not found. Get free token at https://huggingface.co/settings/tokens")
                return
            
            # Use InferenceClient for more reliable inference
            self.hf_client = InferenceClient(
                model=HF_MODEL,
                token=token if token else None
            )
            self.model_name = HF_MODEL.split("/")[-1]
            self._llm_available = True
            self._use_hf_client = True
            print(f"Connected to Hugging Face ({self.model_name})")
        except ImportError:
            print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")
        except Exception as e:
            print(f"Warning: Could not connect to Hugging Face: {e}")
    
    def _setup_ollama(self) -> None:
        """Initialize Ollama local LLM."""
        try:
            from langchain_community.llms import Ollama
            
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=MAX_TOKENS
            )
            self.model_name = OLLAMA_MODEL
            self._llm_available = True
            print(f"Connected to Ollama ({OLLAMA_MODEL})")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
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
                # Use HuggingFace InferenceClient if available
                if hasattr(self, '_use_hf_client') and self._use_hf_client:
                    result = self.hf_client.text_generation(
                        prompt,
                        max_new_tokens=MAX_TOKENS,
                        temperature=LLM_TEMPERATURE
                    )
                    response = result.strip()
                else:
                    # Use LangChain LLM (Ollama)
                    result = self.llm.invoke(prompt)
                    if hasattr(result, 'content'):
                        response = result.content
                    else:
                        response = str(result)
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
        """Stream a response token by token."""
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
                if hasattr(self, '_use_hf_client') and self._use_hf_client:
                    # HF InferenceClient streaming
                    for token in self.hf_client.text_generation(
                        prompt,
                        max_new_tokens=MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        stream=True
                    ):
                        yield token
                        full_response += token
                else:
                    # Stream response with Ollama
                    for chunk in self.llm.stream(prompt):
                        if hasattr(chunk, 'content'):
                            text = chunk.content
                        else:
                            text = str(chunk)
                        yield text
                        full_response += text
            except Exception as e:
                error_msg = f"Error: {e}"
                yield error_msg
                full_response = error_msg
        
        # Store complete response
        self.memory.add_assistant_message(full_response.strip())
        self.memory.periodic_summarization()
    
    def _fallback_response(self, user_message: str) -> str:
        """Fallback response when LLM is not available."""
        if self.provider == "huggingface":
            return (
                "I apologize, but I'm having trouble connecting to the AI service. "
                "Please check that the HF_TOKEN is configured correctly.\n\n"
                "Get your FREE token at: https://huggingface.co/settings/tokens\n\n"
                f"Your message was: {user_message}"
            )
        else:
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
            "provider": self.provider,
            "model": self.model_name,
            "user_id": self.user_id,
            "memory_stats": self.get_memory_stats()
        }
    
    def __repr__(self) -> str:
        status = "ready" if self._llm_available else "offline"
        return f"MemoryAgent(user={self.user_id}, provider={self.provider}, status={status})"
