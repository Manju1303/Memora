"""
Memory Manager Module
Orchestrates short-term and long-term memory operations.
Coordinates memory storage, retrieval, and summarization.
"""
from typing import List, Dict, Optional, Tuple
import sys
sys.path.append('..')

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.summarizer import ConversationSummarizer
from config import DEFAULT_USER_ID


class MemoryManager:
    """
    Central memory orchestration for the AI assistant.
    
    Manages:
    - Short-term memory (recent conversation context)
    - Long-term memory (persistent vector storage)
    - Memory summarization and storage decisions
    
    This implements a hybrid memory architecture:
    - Fast access to recent context (short-term)
    - Semantic search over past memories (long-term)
    """
    
    def __init__(self, user_id: str = DEFAULT_USER_ID):
        self.user_id = user_id
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(user_id=user_id)
        self.summarizer = ConversationSummarizer()
        self._messages_since_summary = 0
    
    def add_user_message(self, content: str) -> None:
        """Add a user message and check for memory opportunities."""
        self.short_term.add_user_message(content)
        self._messages_since_summary += 1
        self._check_for_important_info(content, "user")
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.short_term.add_assistant_message(content)
        self._messages_since_summary += 1
    
    def get_context_for_query(self, query: str) -> Tuple[str, str]:
        """
        Get both short-term and long-term context for a query.
        
        Implements RAG retrieval:
        1. Get recent conversation (short-term)
        2. Search for relevant memories (long-term)
        
        Args:
            query: The user's current query
        
        Returns:
            Tuple of (short_term_context, long_term_memories)
        """
        # Get recent conversation context
        short_term_context = self.short_term.get_formatted_history()
        
        # Retrieve relevant memories from long-term storage
        long_term_memories = self.long_term.get_formatted_memories(query)
        
        return short_term_context, long_term_memories
    
    def _check_for_important_info(self, content: str, role: str) -> None:
        """
        Check if content contains important information to store.
        Automatically extracts and stores user preferences and facts.
        """
        if role != "user":
            return
        
        # Simple heuristics for important information
        important_keywords = [
            "my name is", "i am", "i'm", "i like", "i love",
            "i prefer", "i work", "i live", "i study",
            "my favorite", "i always", "i never", "remember that"
        ]
        
        content_lower = content.lower()
        
        for keyword in important_keywords:
            if keyword in content_lower:
                # Store as a fact
                self.long_term.store_important_fact(content)
                break
    
    def process_end_of_conversation(self) -> None:
        """
        Process and store important information at end of conversation.
        Summarizes the conversation and stores key facts.
        """
        messages = self.short_term.get_langchain_messages()
        
        if len(messages) < 3:
            return  # Not enough to summarize
        
        # Extract and store facts
        facts = self.summarizer.extract_facts(messages)
        for fact in facts:
            self.long_term.store_memory(
                content=fact,
                memory_type="fact"
            )
        
        # Store conversation summary
        if len(messages) >= 5:
            summary = self.summarizer.summarize_conversation(messages)
            if summary:
                self.long_term.store_memory(
                    content=summary,
                    memory_type="conversation_summary",
                    metadata={"session_id": self.short_term.get_session_id()}
                )
        
        self._messages_since_summary = 0
    
    def periodic_summarization(self) -> None:
        """
        Perform periodic summarization if enough messages have accumulated.
        Called automatically during conversation.
        """
        if self._messages_since_summary >= 10:
            messages = self.short_term.get_last_n_messages(10)
            messages_dict = [{"role": m.role, "content": m.content} for m in messages]
            
            facts = self.summarizer.extract_facts(messages_dict)
            for fact in facts:
                self.long_term.store_memory(content=fact, memory_type="fact")
            
            self._messages_since_summary = 0
    
    def store_memory(self, content: str, memory_type: str = "manual") -> str:
        """Manually store a memory."""
        return self.long_term.store_memory(content, memory_type=memory_type)
    
    def search_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search long-term memories."""
        return self.long_term.retrieve_memories(query, n_results)
    
    def get_all_memories(self) -> List[Dict]:
        """Get all stored long-term memories."""
        return self.long_term.get_all_memories()
    
    def clear_short_term(self) -> None:
        """Clear short-term memory (start new conversation)."""
        self.process_end_of_conversation()  # Save important info first
        self.short_term.clear()
    
    def clear_all_memories(self) -> None:
        """Clear all memories (use with caution!)."""
        self.short_term.clear()
        self.long_term.clear_all_memories()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "short_term_messages": len(self.short_term),
            "long_term_memories": self.long_term.get_memory_count(),
            "user_id": self.user_id,
            "session_id": self.short_term.get_session_id(),
            "messages_since_summary": self._messages_since_summary
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"MemoryManager(user={self.user_id}, "
                f"short_term={stats['short_term_messages']}, "
                f"long_term={stats['long_term_memories']})")
