# AI Assistant with Long-term Memory

## Complete Project Description (Exam-Ready Format)

---

## 1. Project Overview

This project builds an **AI assistant with long-term memory** that can remember information across sessions and provide personalized, context-aware responses. Unlike traditional chatbots that forget everything after each session, this assistant maintains persistent memory using vector databases.

---

## 2. Functional Capabilities

### 2.1 Text-based Conversation
The assistant can:
- Accept user queries as text
- Maintain multi-turn conversations
- Respond contextually using previous messages

### 2.2 Voice-based Interaction
The system can:
- Accept voice input using a microphone
- Convert speech to text (Whisper)
- Generate spoken responses using text-to-speech

### 2.3 Persistent Memory Across Sessions
The agent can:
- Remember important user information
- Recall past conversations after app restart
- Maintain continuity over time

### 2.4 Short-term Contextual Awareness
The system can:
- Track recent dialogue turns
- Maintain logical flow within a session
- Handle follow-up questions accurately

### 2.5 Semantic Memory Retrieval
The assistant can:
- Search past memories using meaning, not keywords
- Retrieve relevant past facts automatically
- Use retrieved memories while answering

### 2.6 Personalized Responses
The agent can:
- Adapt responses based on user history
- Remember user preferences and interests
- Provide tailored suggestions

### 2.7 Memory Summarization
The system can:
- Condense conversations into key facts
- Store only meaningful information
- Avoid redundant or noisy memory storage

### 2.8 Offline / Local Execution
The assistant can:
- Run without internet (local LLMs via Ollama)
- Store data locally
- Ensure user privacy

---

## 3. Frontend Tools

### 3.1 Streamlit (Python)
**Purpose:**
- Text input and output interface
- Voice input button (microphone recording)
- Display chat history
- Play AI voice responses
- Memory visualization panel

**Why Streamlit:**
- Fully free and open source
- Python-only stack (no JavaScript required)
- Fast development and prototyping
- Built-in session state management

---

## 4. Backend Tools

### 4.1 Python
Primary backend programming language.

**Purpose:**
- Agent logic implementation
- Memory handling
- AI orchestration
- File and data management

### 4.2 LangChain
Agent orchestration framework.

**Purpose:**
- Prompt construction and templating
- Tool chaining and agent execution
- Memory integration with LLMs
- RAG (Retrieval-Augmented Generation) implementation

### 4.3 Ollama
Open-source LLM runtime.

**Purpose:**
- Run large language models locally
- Generate natural language responses
- Privacy-preserving AI inference

**Models supported:** Mistral, Llama2, CodeLlama, etc.

### 4.4 Whisper (OpenAI)
Speech-to-text engine.

**Purpose:**
- Convert voice input to text
- Support multiple languages
- Works offline with local model

### 4.5 pyttsx3
Text-to-speech engine.

**Purpose:**
- Convert AI text response to voice
- Uses system's native TTS engine
- Works completely offline

---

## 5. Memory and Storage Tools

### 5.1 Short-term Memory (In-memory / Redis-ready)
**Purpose:**
- Store recent conversation messages
- Session-level context (last N messages)
- Fast access for immediate context

**Implementation:** Python deque with configurable size

### 5.2 ChromaDB
Vector database for long-term memory.

**Purpose:**
- Store text embeddings
- Semantic similarity search
- Persistent storage across sessions

**Why ChromaDB:**
- Easy to use with Python
- Built-in persistence
- Supports cosine similarity search

### 5.3 Sentence Transformers
Embedding generation.

**Purpose:**
- Convert text into numerical vectors
- Enable semantic comparison
- Model: all-MiniLM-L6-v2 (fast and efficient)

---

## 6. Algorithms Used

### 6.1 Transformer-based Language Model
**Used in:** Ollama LLMs

**Purpose:**
- Natural language understanding
- Text generation

**Core concept:** Self-attention mechanism

**Flow:**
```
Input Text → Tokenization → Self-Attention → Feed Forward → Output Text
```

### 6.2 Embedding Generation Algorithm
**Used for:** Memory storage and retrieval

**Purpose:**
- Convert text into numerical vectors (embeddings)
- Enable semantic comparison

**Implementation:** Sentence Transformers (MiniLM)

**Process:**
```
Text → Tokenize → Transformer Encoder → 384-dim Vector
```

### 6.3 Vector Similarity Search
**Used in:** ChromaDB memory retrieval

**Purpose:**
- Find memories similar to a query
- Retrieve relevant past information

**Algorithm:** Cosine Similarity + Approximate Nearest Neighbor (ANN)

**Formula:**
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

### 6.4 Conversation Summarization Algorithm
**Used before:** Storing to long-term memory

**Purpose:**
- Compress conversations into key facts
- Extract important information
- Reduce storage size

**Method:** LLM-based abstractive summarization

### 6.5 Retrieval-Augmented Generation (RAG)
**Core agent algorithm.**

**Purpose:**
- Combine retrieved memory with user query
- Improve response relevance and accuracy
- Add context to LLM prompts

**Flow:**
```
User Query → Retrieve Memories → Augment Prompt → Generate Response
```

**Implementation:**
```python
# Simplified RAG flow
memories = vector_db.search(query)
prompt = f"Context: {memories}\nQuery: {query}"
response = llm.generate(prompt)
```

### 6.6 Sliding Window Algorithm
**Used for:** Short-term memory

**Purpose:**
- Keep only last N messages
- Maintain recent context
- Efficient memory usage

**Implementation:**
```python
from collections import deque
messages = deque(maxlen=10)  # Keep last 10 messages
```

### 6.7 Speech Recognition Algorithm (Whisper)
**Used for:** Voice input

**Purpose:**
- Convert audio to text

**Technique:**
- Encoder-decoder neural network
- Log-mel spectrogram input
- Autoregressive text generation

### 6.8 Text-to-Speech Synthesis
**Used for:** Voice output

**Purpose:**
- Convert text to audio
- Enable spoken responses

**Technique:**
- Platform-native TTS engine (Windows SAPI, macOS Speech)

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                        (Streamlit Web App)                       │
│                                                                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│   │  Text Chat  │    │ Voice Input │    │  Memory Viewer      │ │
│   └──────┬──────┘    └──────┬──────┘    └─────────────────────┘ │
└──────────┼──────────────────┼────────────────────────────────────┘
           │                  │
           ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                         VOICE MODULE                              │
│                                                                   │
│   ┌─────────────────┐            ┌─────────────────────────────┐ │
│   │ Whisper (STT)   │            │ pyttsx3 (TTS)               │ │
│   │ Audio → Text    │            │ Text → Audio                │ │
│   └────────┬────────┘            └─────────────────────────────┘ │
└────────────┼─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AGENT MODULE                                │
│                                                                   │
│   ┌─────────────────────────────────────────────────────────────┐│
│   │                    MemoryAgent (RAG)                        ││
│   │                                                             ││
│   │  1. Receive query                                           ││
│   │  2. Retrieve relevant memories                              ││
│   │  3. Build augmented prompt                                  ││
│   │  4. Generate response via LLM                               ││
│   │  5. Store conversation in memory                            ││
│   └───────────────────────────┬─────────────────────────────────┘│
│                               │                                   │
│   ┌───────────────────────────▼─────────────────────────────────┐│
│   │                   Memory Manager                            ││
│   │                                                             ││
│   │  ┌─────────────────┐      ┌─────────────────────────────┐  ││
│   │  │ Short-term      │      │ Long-term                   │  ││
│   │  │ (Sliding Window)│      │ (Vector DB)                 │  ││
│   │  └─────────────────┘      └─────────────────────────────┘  ││
│   └─────────────────────────────────────────────────────────────┘│
└───────────────────────────────┬──────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   Ollama (LLM)    │ │  Python Dict    │ │   ChromaDB          │
│   Local Models    │ │  (Short-term)   │ │   (Long-term)       │
│   - Mistral       │ │  Recent msgs    │ │   Vector Storage    │
│   - Llama2        │ │                 │ │   Embeddings        │
└───────────────────┘ └─────────────────┘ └─────────────────────┘
```

---

## 8. Data Flow

### 8.1 User Message Processing
```
1. User types message or speaks
2. Voice is converted to text (if voice)
3. Message stored in short-term memory
4. Relevant memories retrieved from ChromaDB
5. Prompt constructed with memories + message
6. LLM generates response
7. Response stored in memory
8. Response displayed / spoken to user
```

### 8.2 Memory Storage Flow
```
1. Conversation accumulates
2. Every 10 messages, summarizer runs
3. Key facts extracted
4. Facts converted to embeddings
5. Embeddings stored in ChromaDB
6. Available for future retrieval
```

---

## 9. Use Cases

### 9.1 Academic Projects
- College mini project
- Final year project
- Research prototype

### 9.2 Practical Applications
- Personal productivity assistant
- Study companion
- Technical Q&A bot
- Customer support prototype

### 9.3 Learning Platform
- Learn AI/ML concepts hands-on
- Understand RAG architecture
- Explore vector databases

---

## 10. Technical Specifications

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.9+ |
| UI Framework | Streamlit | 1.28+ |
| Agent Framework | LangChain | 0.1+ |
| Vector Database | ChromaDB | 0.4+ |
| Embeddings | Sentence Transformers | 2.2+ |
| LLM Runtime | Ollama | Latest |
| STT | Whisper | Latest |
| TTS | pyttsx3 | 2.90+ |

---

## 11. Future Enhancements

1. **Multi-user support** with namespaced memories
2. **Redis integration** for distributed short-term memory
3. **Web deployment** on cloud platforms
4. **Multi-modal input** (images, documents)
5. **Fine-tuned models** for specific domains

---

## 12. Conclusion

This project demonstrates a **production-ready architecture** for building AI assistants with memory. By combining vector databases for semantic search with local LLMs for generation, the system achieves both **personalization** and **privacy**.

The modular design allows easy extension and customization for various use cases, making it suitable for both academic projects and practical applications.

---

*Document prepared in exam-ready format with clear sections, algorithms, and technical details.*
