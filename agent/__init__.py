"""
Agent module for AI Assistant
- Core agent logic
- Memory management
- Prompt templates
"""
from .core import MemoryAgent
from .memory_manager import MemoryManager
from .prompts import SYSTEM_PROMPT, get_memory_aware_prompt

__all__ = ["MemoryAgent", "MemoryManager", "SYSTEM_PROMPT", "get_memory_aware_prompt"]
