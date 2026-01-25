# Orion - Local Personal RAG Assistant

A fully local RAG (Retrieval-Augmented Generation) assistant that indexes and searches your personal knowledge base using state-of-the-art embeddings, hybrid search, and reranking.

## 🚀 Features

- **🔍 Advanced Retrieval Pipeline**
  - Hybrid search (semantic + BM25 keyword)
  - Reciprocal Rank Fusion (RRF) for optimal result merging
  - MMR (Maximal Marginal Relevance) for diversity
  - Cross-encoder reranking for precision

- **📁 Multi-Format Support**
  - 30+ file formats: PDF, DOCX, TXT, MD, code files, JSON, CSV, and more
  - Multi-directory knowledge base support
  - Automatic file type detection

- **⚡ GPU Acceleration**
  - CUDA support for 5-10x faster embeddings
  - GPU-accelerated reranking
  - Automatic CPU fallback

- **🔄 Real-time Monitoring**
  - File system watchdog for incremental updates
  - Auto-ingestion of new/modified files
  - Debounced event handling

- **🎛️ Highly Configurable**
  - Environment-based configuration
  - Multiple chunking strategies
  - Adjustable retrieval parameters
  - Flexible embedding models

## 📦 Installation

### Quick Start (CPU-only)

```bash
pip install -r requirements.txt
```

### GPU Acceleration (Recommended)

For NVIDIA GPUs with CUDA 12.8:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

**Check GPU availability:**

```bash
python check_gpu.py
```

See [GPU_INSTALLATION.md](GPU_INSTALLATION.md) for detailed GPU setup instructions.

## ⚙️ Configuration

GPU acceleration is controlled via environment variables or config:

```bash
# Enable GPU
export ORION_GPU_ENABLED=true

# Configure batch sizes (adjust based on GPU memory)
export ORION_EMBEDDING_BATCH_SIZE=64
export ORION_RERANKER_BATCH_SIZE=16
```

See [config.py](src/utilities/config.py) for all configuration options.

## 🎯 Quick Usage

```python
from src.core.ingest import ingest_documents
from src.retrieval.retriever import Retriever

# Ingest your knowledge base
stats = ingest_documents("path/to/your/documents")
print(f"Ingested {stats.total_chunks} chunks from {stats.successful_files} files")

# Search your knowledge base
retriever = Retriever()
results = retriever.retrieve("your query here", k=5)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}")
```

## 🔄 Auto-Ingestion with Watchdog

```python
from src.core.ingest import ingest_with_watchdog

# Start watching and auto-ingesting
ingestor, watcher = ingest_with_watchdog(["/path/to/kb"])

# Files are automatically ingested when added/modified
# Stop watching
watcher.stop()
```

## 📊 Project Status

Work in Progress - Retrieval pipeline complete, generation phase coming soon!

See [Orion_Roadmap.md](Orion_Roadmap.md) for the full development roadmap.