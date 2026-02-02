---
title: Memora
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 Memora

> **Your AI Assistant with Long-term Memory**

An intelligent AI assistant that **remembers** your conversations, preferences, and important information across sessions. Built with Python, LangChain, ChromaDB, and supports both local and **FREE cloud** LLMs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="https://img.shields.io/github/stars/Manju1303/Memora?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/Manju1303/Memora?style=social" alt="Forks">
</p>

---

## 🌐 Try It Online (FREE!)

**Deploy to Streamlit Cloud in minutes - completely FREE!**

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select the Memora repo
4. Add your FREE Hugging Face token in Secrets (see below)
5. Deploy!

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Smart Conversations** | Context-aware multi-turn dialogue |
| 🧠 **Long-term Memory** | Remembers across sessions using vector DB |
| 🎤 **Voice Input** | Speak instead of typing (local only) |
| 🔊 **Voice Output** | Listen to responses (local only) |
| 🔍 **Semantic Search** | Find memories by meaning, not keywords |
| ☁️ **Cloud Deployment** | Deploy FREE on Streamlit Cloud |
| 👤 **Personalization** | Adapts to your preferences |

---

## 🎬 Demo

```
You: My name is Alex and I love Python programming.
Memora: Nice to meet you, Alex! Python is a great choice...

[After restarting the app]

You: What's my name?
Memora: Your name is Alex! You mentioned you love Python programming.
```

---

## 🚀 Deployment Options

### Option 1: ☁️ Cloud (FREE - Recommended!)

Deploy on **Streamlit Community Cloud** with **Hugging Face** (both FREE):

1. **Get FREE Hugging Face Token:**
   - Go to [huggingface.co](https://huggingface.co) and create free account
   - Navigate to Settings → Access Tokens
   - Create a new token (free!)

2. **Deploy to Streamlit Cloud:**
   - Fork this repository
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" → Select your fork
   - Add secrets in Advanced Settings:
     ```toml
     HF_TOKEN = "your-free-token-here"
     LLM_PROVIDER = "huggingface"
     ```
   - Deploy!

### Option 2: 💻 Local (Full Features)

Run locally with Ollama for voice features:

```bash
# Clone the repository
git clone https://github.com/Manju1303/Memora.git
cd Memora

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup Ollama (download from https://ollama.ai)
ollama pull mistral
ollama serve

# Run Memora
streamlit run app.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│        Chat UI  |  Voice Controls  |  Memory Viewer         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      LangChain Agent                         │
│              RAG Engine  |  Memory Manager                   │
└────────────────────────────┬────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────────┐
│  LLM Provider   │  │ Short-term  │  │ ChromaDB (Vector)   │
│ HuggingFace/    │  │ Memory      │  │ Long-term Memory    │
│ Ollama          │  │             │  │                     │
└─────────────────┘  └─────────────┘  └─────────────────────┘
```

---

## 📁 Project Structure

```
memora/
├── app.py                    # 🖥️ Streamlit UI
├── config.py                 # ⚙️ Settings
├── requirements.txt          # 📦 Dependencies
│
├── agent/                    # 🤖 AI Logic
│   ├── core.py               # Main agent (RAG)
│   ├── memory_manager.py     # Memory orchestration
│   └── prompts.py            # Prompt templates
│
├── memory/                   # 💾 Memory Systems
│   ├── short_term.py         # Recent context
│   ├── long_term.py          # Vector storage
│   └── summarizer.py         # Fact extraction
│
├── voice/                    # 🎤 Voice Features (local only)
│   ├── speech_to_text.py     # Whisper STT
│   └── text_to_speech.py     # pyttsx3 TTS
│
└── docs/
    └── PROJECT_DESCRIPTION.md # 📚 Full documentation
```

---

## ⚙️ Configuration

Edit `config.py` or use Streamlit secrets:

```python
# Cloud (FREE)
LLM_PROVIDER = "huggingface"
HF_TOKEN = "your-free-token"

# Local
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "mistral"
```

---

## 🎯 Use Cases

- 📚 **Study Companion** - Remembers what you've learned
- 💼 **Personal Assistant** - Tracks your preferences
- 👨‍💻 **Coding Helper** - Recalls your project context
- 📝 **Note Taking** - Never forget important info

---

## 🔧 Algorithms

| Algorithm | Purpose | File |
|-----------|---------|------|
| **RAG** | Memory-augmented generation | `agent/core.py` |
| **Vector Similarity** | Semantic memory search | `memory/long_term.py` |
| **Sliding Window** | Recent context tracking | `memory/short_term.py` |
| **Abstractive Summarization** | Fact extraction | `memory/summarizer.py` |

---

## 🆓 Cost Breakdown

| Component | Cost |
|-----------|------|
| Streamlit Cloud Hosting | **FREE** |
| Hugging Face API | **FREE** |
| ChromaDB | **FREE** |
| Total | **$0/month** |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - Agent framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Hugging Face](https://huggingface.co) - FREE cloud LLMs
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Streamlit](https://streamlit.io) - UI framework & FREE hosting

---

<p align="center">
  Made with ❤️ | 100% FREE and Open Source
</p>
