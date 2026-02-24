# Orion

A fully local RAG (Retrieval-Augmented Generation) assistant with a modern chat interface, voice synthesis, and real-time knowledge base monitoring. Built for privacy-first personal knowledge management.

![Status](https://img.shields.io/badge/status-active%20development-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## Architecture

Orion consists of four main components:

| Component | Description | Tech Stack |
|-----------|-------------|------------|
| **RAG Pipeline** | Document ingestion, chunking, embedding, hybrid search, and reranking | ChromaDB, sentence-transformers, rank-bm25 |
| **Backend** | REST API and WebSocket server for chat, RAG queries, and system management | FastAPI, Uvicorn |
| **Frontend** | Modern chat interface based on HuggingFace's Chat-UI | SvelteKit, TailwindCSS |
| **Desktop App** | Native system tray application with window management | Tauri (Rust) - coming soon |

```
┌─────────────────────────────────────────────────────────────┐
│                      Orion Desktop                          │
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │ System Tray │  │         Svelte Frontend              │  │
│  │   (Tauri)   │  │   Chat UI / Settings / Ingestion     │  │
│  └─────────────┘  └──────────────────────────────────────┘  │
│                              │                              │
│                    REST API / WebSocket                     │
│                              │                              │
│  ┌───────────────────────────┴───────────────────────────┐  │
│  │                  FastAPI Backend                      │  │
│  │    RAG Pipeline │ Watchdog │ TTS │ LLM (Ollama)       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Retrieval Pipeline
- **Hybrid Search**: Combines semantic vector search with BM25 keyword matching
- **Reciprocal Rank Fusion (RRF)**: Industry-standard fusion algorithm for optimal result merging
- **MMR Diversity**: Maximal Marginal Relevance prevents redundant results
- **Cross-Encoder Reranking**: Two-stage retrieval with precision reranking
- **SQLite Embedding Cache**: Fast cached lookups with 10-100x improvement over file-based caching

### Document Processing
- **30+ File Formats**: PDF, DOCX, TXT, MD, JSON, CSV, YAML, XML, HTML, and source code files
- **Intelligent Chunking**: Recursive text splitting with configurable overlap
- **Automatic Deduplication**: Hash-based and similarity-based duplicate detection

### Real-Time Monitoring
- **File Watchdog**: Monitors knowledge base directories for changes
- **Incremental Ingestion**: Automatically processes new and modified files
- **Content Hash Tracking**: Skips re-ingestion when file content is unchanged
- **Debounced Events**: Consolidates rapid file system events

### Voice Synthesis
- **Piper TTS**: Fast, local neural text-to-speech with multiple voices
- **Qwen3-TTS**: Premium voice synthesis with voice design capabilities (requires GPU)
- **Streaming Audio**: Real-time audio generation during chat responses

### Chat Interface
- **Chat-UI Design**: Modern interface inspired by [HuggingFace Chat-UI](https://github.com/huggingface/chat-ui)
- **Conversation Modes**: Standard chat, RAG-enhanced, and conversation mode
- **Markdown Rendering**: Full markdown support with syntax highlighting
- **Source Citations**: Inline citations linking to source documents

### GPU Acceleration
- **CUDA Support**: 5-10x faster embeddings and reranking on NVIDIA GPUs
- **Automatic Fallback**: Seamlessly falls back to CPU when GPU unavailable
- **Configurable Batch Sizes**: Tune memory usage based on available VRAM

---

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) for local LLM inference
- Node.js 18+ (for frontend development)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Orion.git
cd Orion

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start the backend
python run.py
```

### GPU Acceleration (Recommended)

```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU detection
python check_gpu.py
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

---

## Configuration

Configuration is managed through environment variables or `src/utilities/config.py`:

```bash
# Core settings
ORION_GPU_ENABLED=true
ORION_KNOWLEDGE_BASE_PATHS=./documents,./notes

# Embedding settings
ORION_EMBEDDING_MODEL=all-MiniLM-L12-v2
ORION_EMBEDDING_BATCH_SIZE=64

# Retrieval settings
ORION_RETRIEVAL_DEFAULT_K=5
ORION_RETRIEVAL_ENABLE_RERANKING=true

# TTS settings
ORION_TTS_PROVIDER=piper
ORION_TTS_DEFAULT_VOICE=en_US-amy-medium
```

See [config.py](src/utilities/config.py) for all available options.

---

## Usage

### Ingesting Documents

```python
from src.core.ingest import IngestionPipeline

pipeline = IngestionPipeline()
stats = pipeline.ingest_knowledge_base("./documents")

print(f"Processed {stats.successful_files}/{stats.total_files} files")
print(f"Generated {stats.total_chunks} chunks")
```

### Querying the Knowledge Base

```python
from src.retrieval.retriever import Retriever

retriever = Retriever()
results = retriever.retrieve("How does the authentication system work?", k=5)

for result in results:
    print(f"[{result.score:.2f}] {result.content[:200]}...")
```

### Starting the Watchdog

```python
from src.core.ingest import ingest_with_watchdog

pipeline, watcher = ingest_with_watchdog(["./documents", "./notes"])
# Files are now automatically ingested when added or modified

# To stop
watcher.stop()
```

---

## Project Structure

```
Orion/
├── backend/              # FastAPI application
│   ├── api/              # REST endpoints
│   ├── models/           # Pydantic schemas
│   └── websockets/       # WebSocket handlers
├── frontend/             # SvelteKit application
│   ├── src/
│   │   ├── lib/          # Components and utilities
│   │   └── routes/       # Page routes
│   └── src-tauri/        # Tauri desktop wrapper (coming soon)
├── src/
│   ├── core/             # Ingestion and LLM integration
│   ├── generation/       # Prompt building and response generation
│   ├── retrieval/        # Search, embeddings, and reranking
│   └── utilities/        # Configuration and helpers
└── test/                 # Test suite
```

---

## Roadmap

- [x] Core RAG pipeline (embeddings, vector store, hybrid search)
- [x] Cross-encoder reranking
- [x] File watchdog with incremental ingestion
- [x] FastAPI backend with WebSocket support
- [x] Chat-UI frontend
- [x] Piper TTS integration
- [x] Qwen3-TTS voice synthesis
- [ ] Tauri desktop application
- [ ] System tray with quick actions
- [ ] Auto-update mechanism

See [Orion_Roadmap.md](Orion_Roadmap.md) for detailed planning.

---

## Acknowledgments

- [HuggingFace Chat-UI](https://github.com/huggingface/chat-ui) - Frontend design inspiration
- [Piper TTS](https://github.com/rhasspy/piper) - Local neural text-to-speech
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Advanced voice synthesis
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Ollama](https://ollama.ai) - Local LLM inference

---

## License

MIT License - See [LICENSE](LICENSE) for details.