"""
Memora - AI Memory Assistant
Streamlit frontend with chat interface, voice interaction, and memory visualization.
Supports both local (Ollama) and cloud (Groq) deployment.
"""
import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import MemoryAgent
from agent.prompts import (
    SYSTEM_PROMPT, 
    TAMIL_SYSTEM_PROMPT, 
    ACADEMIC_PROMPT, 
    CONTENT_CREATION_PROMPT
)

# Voice features are optional (may not work in cloud deployment)
try:
    from voice.text_to_speech import TextToSpeech
    from voice.speech_to_text import SpeechToText, record_audio
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


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
# Custom CSS for Premium Dark UI
# =============================================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d15 50%, #0a0a12 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat messages with glassmorphism */
    .chat-message {
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        animation: fadeIn 0.4s ease-out;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        color: white;
        margin-left: 15%;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.95) 0%, rgba(36, 59, 85, 0.95) 100%);
        color: #e8e8e8;
        margin-right: 15%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Animated gradient header */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s ease-in-out infinite;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    
    /* Glassmorphism cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
    }
    
    /* Memory items with glow */
    .memory-item {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s ease;
    }
    
    .memory-item:hover {
        background: rgba(102, 126, 234, 0.15);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Premium sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d15 0%, #0a0a12 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Status indicators with glow */
    .status-online {
        color: #00ff88;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .status-offline {
        color: #ff6b6b;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
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
    
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False
    
    # Initialize voice features if available
    if VOICE_AVAILABLE:
        if "tts" not in st.session_state:
            st.session_state.tts = TextToSpeech()
        if "stt" not in st.session_state:
            st.session_state.stt = SpeechToText()


init_session_state()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## 🧠 Memora")
    st.markdown("---")
    
    # Status
    agent_status = st.session_state.agent.get_status()
    provider = agent_status.get("provider", "unknown")
    model = agent_status.get("model", "unknown")
    
    if agent_status["llm_available"]:
        provider_icon = "☁️" if provider == "groq" else "💻"
        st.markdown(f'<p class="status-online">{provider_icon} Connected ({model})</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">● LLM Offline</p>', unsafe_allow_html=True)
        if provider == "groq":
            st.warning("Check GROQ_API_KEY")
        elif provider == "huggingface":
            # Add API Key input field
            api_key = st.text_input("Enter Hugging Face Token:", type="password")
            if api_key:
                st.session_state.HF_TOKEN = api_key
                st.session_state.agent = MemoryAgent() # Re-initialize agent
                st.rerun()
            st.markdown("[Get free token](https://huggingface.co/settings/tokens)")
        else:
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
    
    # Voice Settings (only show if available)
    if VOICE_AVAILABLE:
        st.markdown("### 🎤 Voice Settings")
        st.session_state.voice_enabled = st.toggle("Enable Voice Output", value=st.session_state.voice_enabled)
        
        if st.session_state.voice_enabled:
            tts_rate = st.slider("Speech Rate", 100, 250, 150)
            st.session_state.tts.set_rate(tts_rate)
    else:
        st.markdown("### 🎤 Voice")
        st.info("Voice features not available in cloud mode")
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Assistant Mode
    st.markdown("### 🎭 Assistant Mode")
    mode = st.selectbox("Select Role", ["General Assistant", "Academic Tutor", "Content Creator"])
    
    # Language Selection
    st.markdown("### 🌐 Language")
    language = st.selectbox("Select Language", ["English", "Tamil"])
    
    # Prompt Logic
    if language == "Tamil":
        system_prompt = TAMIL_SYSTEM_PROMPT
        if mode == "Academic Tutor":
            system_prompt += "\n\nNote: Focus on providing clear, educational explanations and answer keys."
        elif mode == "Content Creator":
            system_prompt += "\n\nNote: Focus on creative writing and engaging content."
    else:
        if mode == "Academic Tutor":
            system_prompt = ACADEMIC_PROMPT
        elif mode == "Content Creator":
            system_prompt = CONTENT_CREATION_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT
            
    st.markdown("---")
    
    # Document Upload
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF/DOCX/TXT", type=['pdf', 'docx', 'txt'])
    if uploaded_file:  
        if st.button("Process Document"):
            with st.spinner("Reading document..."):
                # Save to temp file or process directly
                # agent.ingest_file handles the file object
                result = st.session_state.agent.ingest_file(uploaded_file, uploaded_file.name)
                st.success(result)
    
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
# =============================================================================
# Main Interface (Tabs)
# =============================================================================
tab1, tab2 = st.tabs(["💬 Chat", "🎨 Image Generation"])

with tab1:
    # Chat Interface
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

    # Input Area
    st.markdown("---")
    
    if VOICE_AVAILABLE:
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.chat_input("Type your message here...")
        with col2:
            if st.button("🎤 Voice", use_container_width=True):
                with st.spinner("Recording..."):
                    audio = record_audio(duration=5.0)
                    if audio:
                        text = st.session_state.stt.transcribe(audio)
                        if text: user_input = text
    else:
        user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(user_input, system_context=system_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        if VOICE_AVAILABLE and st.session_state.voice_enabled:
            st.session_state.tts.speak_async(response)
        st.rerun()

with tab2:
    st.header("🎨 AI Image Generation")
    st.markdown("Generate images from text descriptions.")
    
    img_prompt = st.text_input("Enter image description...", key="img_prompt")
    
    if st.button("Generate Image", type="primary"):
        if img_prompt:
            with st.spinner("Generating image... (This may take a moment)"):
                image_path = st.session_state.agent.generate_image(img_prompt)
                st.image(image_path, caption=img_prompt)
                st.success(f"Image saved to {image_path}")
        else:
            st.warning("Please enter a description.")


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
