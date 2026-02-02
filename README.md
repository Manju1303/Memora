# 🧠 Memora

> **Your AI Assistant with Long-term Memory**

An intelligent AI assistant that **remembers** your conversations, preferences, and important information across sessions. Built with Python, LangChain, ChromaDB, and local LLMs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="https://img.shields.io/github/stars/Manju1303/Memora?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/Manju1303/Memora?style=social" alt="Forks">
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Smart Conversations** | Context-aware multi-turn dialogue |
| 🧠 **Long-term Memory** | Remembers across sessions using vector DB |
| 🎤 **Voice Input** | Speak instead of typing (Whisper) |
| 🔊 **Voice Output** | Listen to responses (TTS) |
| 🔍 **Semantic Search** | Find memories by meaning, not keywords |
| 🔒 **Privacy First** | 100% local, no data sent to cloud |
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
│   Ollama LLM    │  │ Short-term  │  │ ChromaDB (Vector)   │
│   (Local AI)    │  │ Memory      │  │ Long-term Memory    │
└─────────────────┘  └─────────────┘  └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) (for local LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/Manju1303/Memora.git
cd Memora

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Ollama

```bash
# Download from https://ollama.ai, then:
ollama pull mistral
ollama serve
```

### Run Memora

```bash
streamlit run app.py
```

🎉 Open `http://localhost:8501` in your browser!

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
├── voice/                    # 🎤 Voice Features
│   ├── speech_to_text.py     # Whisper STT
│   └── text_to_speech.py     # pyttsx3 TTS
│
└── docs/
    └── PROJECT_DESCRIPTION.md # 📚 Full documentation
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
OLLAMA_MODEL = "mistral"       # LLM model (mistral, llama2, codellama)
SHORT_TERM_MEMORY_SIZE = 10    # Recent messages to keep
MEMORY_RETRIEVAL_COUNT = 5     # Memories per query
WHISPER_MODEL_SIZE = "base"    # tiny/base/small/medium/large
TTS_RATE = 150                 # Speech speed (WPM)
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - Agent framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Streamlit](https://streamlit.io) - UI framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition

---

## ⭐ Star History

If you find Memora useful, please consider giving it a star! ⭐

---

<p align="center">
  Made with ❤️ by the Memora Team
</p>
