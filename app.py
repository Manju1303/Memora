"""
AI Memory Assistant - Streamlit Frontend
Main application with chat interface, voice interaction, and memory visualization.
"""
import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import MemoryAgent
from voice.text_to_speech import TextToSpeech
from voice.speech_to_text import SpeechToText, record_audio
from config import OLLAMA_MODEL, WHISPER_MODEL_SIZE


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Memora - AI Memory Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Custom CSS for Premium UI
# =============================================================================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: #e0e0e0;
        margin-right: 20%;
        border: 1px solid #3d5a80;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s linear infinite;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    @keyframes gradient {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #3d5a80;
    }
    
    /* Memory items */
    .memory-item {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Voice button */
    .voice-btn {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .voice-btn:hover {
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Status indicator */
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ff6b6b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = MemoryAgent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "tts" not in st.session_state:
        st.session_state.tts = TextToSpeech()
    
    if "stt" not in st.session_state:
        st.session_state.stt = SpeechToText()
    
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False
    
    if "recording" not in st.session_state:
        st.session_state.recording = False


init_session_state()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## 🧠 Memora")
    st.markdown("---")
    
    # Status
    agent_status = st.session_state.agent.get_status()
    if agent_status["llm_available"]:
        st.markdown(f'<p class="status-online">● LLM Connected ({OLLAMA_MODEL})</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">● LLM Offline</p>', unsafe_allow_html=True)
        st.warning("Start Ollama: `ollama serve`")
    
    st.markdown("---")
    
    # Memory Stats
    st.markdown("### 📊 Memory Stats")
    stats = st.session_state.agent.get_memory_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Short-term", stats["short_term_messages"])
    with col2:
        st.metric("Long-term", stats["long_term_memories"])
    
    st.markdown("---")
    
    # Voice Settings
    st.markdown("### 🎤 Voice Settings")
    st.session_state.voice_enabled = st.toggle("Enable Voice Output", value=st.session_state.voice_enabled)
    
    if st.session_state.voice_enabled:
        tts_rate = st.slider("Speech Rate", 100, 250, 150)
        st.session_state.tts.set_rate(tts_rate)
    
    st.markdown("---")
    
    # Actions
    st.markdown("### ⚡ Actions")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.agent.new_conversation()
        st.session_state.messages = []
        st.success("Started new conversation!")
        st.rerun()
    
    if st.button("📋 View Memories", use_container_width=True):
        st.session_state.show_memories = True
    
    if st.button("🗑️ Clear All Memories", use_container_width=True):
        if st.session_state.get("confirm_clear", False):
            st.session_state.agent.clear_all_memories()
            st.session_state.messages = []
            st.session_state.confirm_clear = False
            st.success("All memories cleared!")
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm deletion")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AI Assistant with Long-term Memory**
    
    Features:
    - 💬 Text & voice interaction
    - 🧠 Persistent memory
    - 🔍 Semantic search
    - 🎯 Personalized responses
    """)


# =============================================================================
# Main Content
# =============================================================================
st.markdown('<h1 class="main-header">🧠 Memora</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Your AI companion with long-term memory</p>", unsafe_allow_html=True)

# Memory viewer modal
if st.session_state.get("show_memories", False):
    with st.expander("📚 Stored Memories", expanded=True):
        memories = st.session_state.agent.get_all_memories()
        if memories:
            for mem in memories[:20]:  # Show last 20
                st.markdown(f"""
                <div class="memory-item">
                    <strong>{mem.get('metadata', {}).get('memory_type', 'memory')}</strong><br>
                    {mem['content'][:200]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No memories stored yet. Start chatting to build memory!")
        
        if st.button("Close"):
            st.session_state.show_memories = False
            st.rerun()


# =============================================================================
# Chat Interface
# =============================================================================
# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Input Area
# =============================================================================
st.markdown("---")

# Voice recording section
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Type your message here...")

with col2:
    if st.button("🎤 Voice", use_container_width=True, help="Click to record voice input"):
        with st.spinner("Recording... (5 seconds)"):
            audio = record_audio(duration=5.0)
            if audio is not None:
                with st.spinner("Transcribing..."):
                    if not st.session_state.stt.is_ready():
                        st.session_state.stt.load_model()
                    text = st.session_state.stt.transcribe(audio)
                    if text and not text.startswith("Error"):
                        user_input = text
                        st.success(f"Transcribed: {text}")
                    else:
                        st.error("Could not transcribe audio")


# Process input
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner("Thinking..."):
        response = st.session_state.agent.chat(user_input)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Voice output if enabled
    if st.session_state.voice_enabled:
        st.session_state.tts.speak_async(response)
    
    # Rerun to update UI
    st.rerun()


# =============================================================================
# Quick Actions
# =============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("👋 Introduce yourself", use_container_width=True):
        intro = "Hi! I'm your AI assistant with long-term memory. I can remember our conversations and provide personalized help. What's your name?"
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.rerun()

with col2:
    if st.button("💡 What can you do?", use_container_width=True):
        capabilities = """I can help you with many things:
        
• 💬 Answer questions and have conversations
• 📚 Remember important information about you
• 🔍 Recall past conversations and preferences
• 🎤 Voice interaction (speak and listen)
• 📝 Help with learning, coding, writing, and more

Try telling me your name or interests - I'll remember them!"""
        st.session_state.messages.append({"role": "assistant", "content": capabilities})
        st.rerun()

with col3:
    if st.button("🔍 What do you remember?", use_container_width=True):
        memories = st.session_state.agent.recall("user preferences and information")
        st.session_state.messages.append({"role": "assistant", "content": memories})
        st.rerun()


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #666; font-size: 0.8rem;'>
    Memora - AI Memory Assistant | Built with Streamlit, LangChain, ChromaDB, and Ollama<br>
    💡 Tip: Your conversations are stored locally and help me provide better responses over time.
</p>
""", unsafe_allow_html=True)
