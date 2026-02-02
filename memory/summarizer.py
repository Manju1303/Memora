"""
Conversation Summarizer Module
Extracts key facts and compresses conversations before long-term storage.
Uses LLM-based abstractive summarization.
"""
from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

import sys
sys.path.append('..')
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# Summarization prompt template
SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["conversation"],
    template="""Analyze the following conversation and extract the key information that should be remembered long-term.

Focus on:
1. Important facts about the user (name, preferences, interests)
2. Key decisions or conclusions reached
3. Problems discussed and solutions found
4. Any commitments or action items
5. Personal details shared by the user

Conversation:
{conversation}

Provide a concise summary of key facts to remember (2-4 bullet points):"""
)


EXTRACT_FACTS_PROMPT = PromptTemplate(
    input_variables=["conversation"],
    template="""Extract specific, memorable facts from this conversation that would be useful to recall in future conversations.

Conversation:
{conversation}

List each fact on a separate line, starting with "- ". Only include concrete, specific information.
Facts:"""
)


class ConversationSummarizer:
    """
    Summarizes conversations for efficient long-term storage.
    
    Algorithm: LLM-based Abstractive Summarization
    - Takes full conversation
    - Extracts key facts and important information
    - Produces condensed, meaningful summaries
    """
    
    def __init__(self):
        self._setup_llm()
    
    def _setup_llm(self) -> None:
        """Initialize the LLM for summarization."""
        try:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.3  # Lower temperature for more focused summaries
            )
            self._llm_available = True
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            self._llm_available = False
    
    def summarize_conversation(self, messages: List[Dict]) -> str:
        """
        Summarize a conversation into key points.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Summarized key points as a string
        """
        if not messages:
            return ""
        
        # Format conversation for the prompt
        conversation_text = self._format_messages(messages)
        
        if not self._llm_available:
            return self._fallback_summarize(messages)
        
        try:
            prompt = SUMMARIZE_PROMPT.format(conversation=conversation_text)
            summary = self.llm.invoke(prompt)
            return summary.strip()
        except Exception as e:
            print(f"Summarization error: {e}")
            return self._fallback_summarize(messages)
    
    def extract_facts(self, messages: List[Dict]) -> List[str]:
        """
        Extract individual facts from a conversation.
        
        Args:
            messages: List of message dicts
        
        Returns:
            List of extracted facts
        """
        if not messages:
            return []
        
        conversation_text = self._format_messages(messages)
        
        if not self._llm_available:
            return self._fallback_extract_facts(messages)
        
        try:
            prompt = EXTRACT_FACTS_PROMPT.format(conversation=conversation_text)
            response = self.llm.invoke(prompt)
            
            # Parse bullet points
            facts = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    facts.append(line[2:])
                elif line.startswith("• "):
                    facts.append(line[2:])
                elif line and not line.startswith("#"):
                    facts.append(line)
            
            return facts
        except Exception as e:
            print(f"Fact extraction error: {e}")
            return self._fallback_extract_facts(messages)
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into a readable conversation string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _fallback_summarize(self, messages: List[Dict]) -> str:
        """Simple fallback when LLM is not available."""
        if not messages:
            return ""
        
        # Just take the last few messages as summary
        recent = messages[-3:] if len(messages) > 3 else messages
        summary_parts = []
        
        for msg in recent:
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            summary_parts.append(content)
        
        return " | ".join(summary_parts)
    
    def _fallback_extract_facts(self, messages: List[Dict]) -> List[str]:
        """Simple fallback for fact extraction."""
        facts = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Extract messages that look like facts (contain "my", "I am", etc.)
                if any(keyword in content.lower() for keyword in ["my name", "i am", "i like", "i work", "i live"]):
                    if len(content) < 200:
                        facts.append(content)
        return facts[:5]  # Limit to 5 facts
    
    def should_summarize(self, message_count: int, last_summary_count: int = 0) -> bool:
        """
        Determine if it's time to summarize the conversation.
        Summarize every 10 new messages.
        """
        return (message_count - last_summary_count) >= 10
    
    def is_llm_available(self) -> bool:
        """Check if the LLM is available for summarization."""
        return self._llm_available
