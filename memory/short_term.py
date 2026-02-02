"""
Short-term Memory Module
Implements sliding window for recent conversation context.
Uses in-memory storage (can be upgraded to Redis for production).
"""
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

import sys
sys.path.append('..')
from config import SHORT_TERM_MEMORY_SIZE


@dataclass
class Message:
    """Represents a single conversation message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class ShortTermMemory:
    """
    Sliding window memory for recent conversations.
    
    Algorithm: Sliding Window
    - Keeps only the last N messages
    - Provides immediate context for the LLM
    - Fast O(1) add and O(n) retrieval
    """
    
    def __init__(self, max_size: int = SHORT_TERM_MEMORY_SIZE):
        self.max_size = max_size
        self._messages: deque = deque(maxlen=max_size)
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation history."""
        message = Message(role=role, content=content)
        self._messages.append(message)
    
    def add_user_message(self, content: str) -> None:
        """Convenience method to add a user message."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Convenience method to add an assistant message."""
        self.add_message("assistant", content)
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the current window."""
        return list(self._messages)
    
    def get_formatted_history(self) -> str:
        """Get conversation history formatted for LLM context."""
        if not self._messages:
            return "No recent conversation history."
        
        formatted = []
        for msg in self._messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{prefix}: {msg.content}")
        
        return "\n".join(formatted)
    
    def get_langchain_messages(self) -> List[Dict]:
        """Get messages in LangChain format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._messages
        ]
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages."""
        messages = list(self._messages)
        return messages[-n:] if n < len(messages) else messages
    
    def clear(self) -> None:
        """Clear all messages from short-term memory."""
        self._messages.clear()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id
    
    def get_message_count(self) -> int:
        """Get the number of messages in memory."""
        return len(self._messages)
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self._messages) == 0
    
    def export_session(self) -> Dict:
        """Export the current session for potential long-term storage."""
        return {
            "session_id": self._session_id,
            "messages": [msg.to_dict() for msg in self._messages],
            "message_count": len(self._messages)
        }
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __repr__(self) -> str:
        return f"ShortTermMemory(messages={len(self._messages)}, max={self.max_size})"
