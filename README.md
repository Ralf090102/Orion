# Orion

A fully local RAG (Retrieval-Augmented Generation) assistant with a modern chat interface, voice synthesis, and real-time knowledge base monitoring. Built for privacy-first personal knowledge management.

![Status](https://img.shields.io/badge/status-active%20development-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/4604a344-fabd-40e4-8376-9df9b4dc20fc" />

---

## Architecture

Orion consists of four main components:

| Component | Description | Tech Stack |
|-----------|-------------|------------|
| **RAG Pipeline** | Document ingestion, chunking, embedding, hybrid search, and reranking | ChromaDB, sentence-transformers, rank-bm25 |
| **Backend** | REST API and WebSocket server for chat, RAG queries, and system management | FastAPI, Uvicorn |
| **Frontend** | Modern chat interface based on HuggingFace's Chat-UI | SvelteKit, TailwindCSS |
| **Desktop App** | Native system tray application with window management | Tauri (Rust) |

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
git clone https://github.com/Ralf090102/Orion.git
cd Orion

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start the backend (FastAPI server)
python -m backend.app
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

### Desktop App (Tauri)

The desktop build wraps the SvelteKit frontend and the Python backend (auto-started as a sidecar process) in a native window with a system tray.

**Additional prerequisites (Windows):**
- [Rust](https://rustup.rs) (`winget install Rust.Rustup`)
- **Microsoft C++ Build Tools** — install via `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools"`, or through the Visual Studio Installer with the "Desktop development with C++" workload. Tauri's build needs the real MSVC `link.exe`; without this workload, Rust/Cargo either fails with `linker 'link.exe' not found`, or — if running from a Git Bash/MSYS shell — silently picks up MSYS's unrelated `link` (hard-link) utility instead and fails with a confusing `link: missing operand` error. Build from PowerShell/cmd, not Git Bash, to avoid the latter.

```bash
cd frontend
npx tauri dev    # dev mode, hot-reloads frontend + Rust
npx tauri build  # production installer
```

App launch itself should be fast (the backend just needs to bind its port) — it's the *first RAG query* that pays a one-time cost loading the embedding + reranker models, not app startup. (Prior to a 2026-09 fix, `backend/app.py` eagerly imported the entire ML stack — torch, sentence-transformers, chromadb — at module load, before the backend could even bind its port; that's what used to make launch itself slow. Those imports are now deferred to first actual use.)

**Building a distributable installer** (i.e. for a machine without this repo's `.venv`): `npx tauri build` needs a bundled, self-contained Python runtime — it doesn't ship one by default. Generate it once (or whenever `requirements.txt` changes) with:

```powershell
.\scripts\build_python_runtime.ps1
```

This downloads the official Python embeddable package into `python-runtime/` (gitignored — it's a build artifact, ~1.8GB with the CPU-only ML stack) and installs `requirements.txt` into it. `tauri.conf.json`'s `bundle.resources` then packages `python-runtime/`, `backend/`, and `src/` alongside the installer; at runtime, `backend.rs` prefers this bundled runtime and only falls back to a dev `.venv` (or system Python) when it isn't present.

**Verified end-to-end 2026-08-05**: `bundle.targets` is currently scoped to `"nsis"` (WiX/MSI has more historical friction with the large, deeply-nested file tree an ML stack produces — untested here, but a one-line config change to add back if wanted). A real release build (`npx tauri build`, no `--debug`/`--no-bundle`) produced `Orion_0.1.0_x64-setup.exe` (~626MB, compressed from the ~1.86GB bundled runtime) — roughly 7 minutes to compile the Rust release binary, then another ~15-20 minutes for NSIS to LZMA-compress the resource payload (single-threaded, genuinely slow on this much data but not stuck). Installed silently over a real pre-existing install, launched cleanly (backend healthy in ~5s), and a full RAG round-trip through the actual REST API returned a correctly grounded answer with real citations and sources — the same verification depth as dev mode, not just a smoke test. TTS, sessions, and the vector store all resolve correctly in the installed build (fixed 2026-08-04, re-confirmed 2026-08-05).

**Note (2026-09-01):** despite the ~5s figure above, backend startup on a real installed build was later found to take up to several minutes in practice, traced to `backend/app.py` eagerly importing the entire ML stack (torch, sentence-transformers, chromadb, PyMuPDF/langchain) at module load, before Uvicorn could even bind the port — likely compounded by Windows Defender scanning the ~20,000+ loose files those packages unpack to in the bundled `python-runtime`. Fixed by deferring those imports to the functions that actually use them (so the cost lands on first RAG query / first ingestion instead of app launch); the ~5s figure above predates that regression and hasn't been re-measured against the fix yet — needs a fresh install timing pass to replace this note with a real number.

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
│   └── src-tauri/        # Tauri desktop wrapper
├── src/
│   ├── core/             # Ingestion and LLM integration
│   ├── generation/       # Prompt building and response generation
│   ├── retrieval/        # Search, embeddings, and reranking
│   └── utilities/        # Configuration and helpers
└── test/                 # Test suite
```

---

## Testing

```bash
# Run the backend test suite (must use the project's own .venv --
# system Python is missing dependencies like soundfile)
.venv\Scripts\python.exe -m pytest test/ -v
```

Tests use FastAPI's `dependency_overrides` to fake out the ML stack (retriever, generator, session manager), so they run in seconds without needing Ollama or ChromaDB. See `test/conftest.py` for the fixtures.

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
