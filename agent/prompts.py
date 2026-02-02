"""
Prompt Templates for the AI Memory Assistant
Defines system prompts and memory-aware prompt construction.
"""

# =============================================================================
# System Prompt
# =============================================================================
SYSTEM_PROMPT = """You are a helpful AI assistant with long-term memory capabilities. You can remember information from past conversations and use it to provide personalized, contextual responses.

Your key abilities:
1. Remember important facts about the user
2. Recall past conversations and preferences
3. Provide personalized recommendations
4. Maintain continuity across sessions

Guidelines:
- Be friendly, helpful, and conversational
- Use memories naturally without being creepy about it
- If you remember something relevant, mention it naturally
- If asked about something you don't know, be honest
- Help with a wide range of tasks: learning, coding, writing, advice, etc.

You have access to:
- Recent conversation history (short-term memory)
- Relevant past memories (long-term memory)

Use this context to provide the best possible response."""


# =============================================================================
# Memory-Aware Prompt Template
# =============================================================================
MEMORY_PROMPT_TEMPLATE = """You are a helpful AI assistant with memory capabilities.

{system_context}

=== LONG-TERM MEMORIES ===
{long_term_memories}

=== RECENT CONVERSATION ===
{short_term_context}

=== CURRENT USER MESSAGE ===
User: {user_message}

Respond naturally and helpfully, using relevant memories when appropriate:"""


# =============================================================================
# Prompt Construction Functions
# =============================================================================
def get_memory_aware_prompt(
    user_message: str,
    short_term_context: str = "",
    long_term_memories: str = "",
    system_context: str = SYSTEM_PROMPT
) -> str:
    """
    Construct a memory-aware prompt for the LLM.
    
    This implements the RAG (Retrieval-Augmented Generation) pattern:
    1. Query is received
    2. Relevant memories are retrieved
    3. Memories are injected into prompt
    4. LLM generates contextual response
    
    Args:
        user_message: The current user query
        short_term_context: Recent conversation history
        long_term_memories: Retrieved relevant memories
        system_context: System prompt/instructions
    
    Returns:
        Complete prompt string for the LLM
    """
    return MEMORY_PROMPT_TEMPLATE.format(
        system_context=system_context,
        long_term_memories=long_term_memories or "No relevant past memories.",
        short_term_context=short_term_context or "This is the start of a new conversation.",
        user_message=user_message
    )


def get_simple_prompt(user_message: str, context: str = "") -> str:
    """Simple prompt without memory features."""
    if context:
        return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {user_message}\n\nAssistant:"
    return f"{SYSTEM_PROMPT}\n\nUser: {user_message}\n\nAssistant:"


# =============================================================================
# Specialized Prompts
# =============================================================================
MEMORY_EXTRACTION_PROMPT = """Based on this conversation, identify any important information that should be remembered about the user for future conversations.

Conversation:
{conversation}

List important facts to remember (or "None" if nothing notable):"""


PREFERENCE_DETECTION_PROMPT = """Analyze this message and identify any user preferences or personal information mentioned.

Message: {message}

Preferences found (or "None"):"""
