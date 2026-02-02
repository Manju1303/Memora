"""
Configuration settings for Memora - AI Memory Assistant
"""
import os

# =============================================================================
# LLM Provider Configuration
# =============================================================================
# Options: "ollama" (local) or "huggingface" (cloud - FREE)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama Configuration (Local)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # Options: mistral, llama2, codellama, etc.

# Hugging Face Configuration (Cloud - COMPLETELY FREE)
# Get your free token at: https://huggingface.co/settings/tokens
# Just create a free account, no credit card needed!
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Free model

# =============================================================================
# Memory Configuration
# =============================================================================
# Short-term memory: number of recent messages to keep
SHORT_TERM_MEMORY_SIZE = 10

# Long-term memory: ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHROMA_COLLECTION_NAME = "assistant_memory"

# Embedding model for vector storage
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of relevant memories to retrieve
MEMORY_RETRIEVAL_COUNT = 5

# =============================================================================
# Voice Configuration
# =============================================================================
# Whisper model size: tiny, base, small, medium, large
WHISPER_MODEL_SIZE = "base"

# Text-to-speech rate (words per minute)
TTS_RATE = 150

# Audio recording settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# =============================================================================
# Agent Configuration
# =============================================================================
# Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE = 0.7

# Maximum tokens in response
MAX_TOKENS = 512

# =============================================================================
# User Configuration
# =============================================================================
# Default user namespace (for multi-user support)
DEFAULT_USER_ID = "default_user"
