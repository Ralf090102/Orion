# 🌟 Orion - Local RAG Pipeline

A powerful, local Retrieval-Augmented Generation (RAG) system that allows you to query your documents using Large Language Models (LLMs) through Ollama.

## ✨ Features

- 📁 **Multi-format Document Support**: PDF, DOCX, Excel (.xlsx, .xls), and TXT files
- 🔍 **Intelligent Document Retrieval**: FAISS-powered similarity search with scoring
- 🤖 **Local LLM Integration**: Uses Ollama for private, offline AI processing
- 📊 **Rich CLI Interface**: Colorful logging and progress tracking
- ⚙️ **Configurable Parameters**: Customizable chunk sizes, retrieval settings, and more
- 🛡️ **Robust Error Handling**: Comprehensive validation and fallback mechanisms
- 🧪 **Testing Suite**: Unit tests for core functionality

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull Required Models**:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ralf090102/Orion.git
cd Orion

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Usage

#### 1. Ingest Documents

```bash
# Basic ingestion
python -m app.main ingest --path ./documents

# Advanced ingestion with custom settings
python -m app.main ingest --path ./documents --persist ./my_vectorstore --chunk-size 1500 --chunk-overlap 300
```

#### 2. Query Your Knowledge Base

```bash
# Simple query
python -m app.main query --question "What are the main findings?"

# Query with specific model and settings
python -m app.main query --question "Summarize the key points" --model llama2 --persist ./my_vectorstore -k 5
```

#### 3. List Available Models

```bash
python -m app.main models
```

## 📖 Detailed Usage

### Command Reference

#### Ingest Command
```bash
python -m app.main ingest [OPTIONS]

Options:
  --path TEXT           Path to folder containing documents [required]
  --persist TEXT        Path to save vectorstore (default: vectorstore)
  --chunk-size INTEGER  Text chunk size for splitting (default: 1000)
  --chunk-overlap INTEGER  Overlap between chunks (default: 200)
```

#### Query Command
```bash
python -m app.main query [OPTIONS]

Options:
  --question TEXT       Your question to ask the knowledgebase [required]
  --persist TEXT        Path to load vectorstore (default: vectorstore)
  --model TEXT          Ollama model to use (default: mistral)
  -k INTEGER           Number of relevant documents to retrieve (default: 3)
```

### Environment Configuration

Set environment variables to customize default behavior:

```bash
export ORION_EMBEDDING_MODEL="nomic-embed-text"
export ORION_LLM_MODEL="mistral"
export ORION_CHUNK_SIZE=1000
export ORION_CHUNK_OVERLAP=200
export ORION_RETRIEVAL_K=3
export ORION_TEMPERATURE=0.7
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Text Chunks    │    │  FAISS Vector   │
│  (PDF, DOCX,    │───▶│   (Processed     │───▶│    Store       │
│   Excel, TXT)   │    │    & Split)      │    │  (Embeddings)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                               ┌─────────────────────────┘
                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Response     │    │   Ollama LLM     │    │   Query +       │
│   (Generated    │◀───│   (Mistral,      │◀───│   Retrieved    │
│    Answer)      │    │   Llama, etc.)   │    │   Context       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Custom Configuration

Create a custom configuration by modifying `app/config.py`:

```python
from app.config import Config

# Create custom config
config = Config()
config.CHUNK_SIZE = 1500
config.RETRIEVAL_K = 5
config.TEMPERATURE = 0.8
```

### Supported File Types

- **PDF**: `.pdf`
- **Word Documents**: `.docx`
- **Excel Spreadsheets**: `.xlsx`, `.xls`
- **Text Files**: `.txt`

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_dummy.py
```

## 🛠️ Development

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code
black app/ tests/

# Lint code
ruff app/ tests/

# Type checking (if mypy is installed)
mypy app/
```

### Project Structure

```
Orion/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── ingest.py          # Document ingestion
│   ├── llm.py             # LLM interaction
│   ├── main.py            # CLI interface
│   ├── query.py           # Query processing
│   └── utils.py           # Utility functions
├── tests/
│   ├── test_dummy.py      # Test suite
│   └── ...
├── data/                  # Data directories
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
└── README.md
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Not Running**
   ```bash
   ollama serve
   ```

2. **Model Not Found**
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Empty Query Results**
   - Check if documents were properly ingested
   - Try reducing similarity threshold
   - Increase the number of retrieved documents (`-k`)

4. **Memory Issues**
   - Reduce chunk size
   - Process documents in smaller batches
   - Use a smaller embedding model

### Performance Tips

- **Chunk Size**: Larger chunks (1500-2000) for detailed answers, smaller chunks (500-1000) for specific facts
- **Overlap**: 10-20% of chunk size for good continuity
- **Retrieval K**: Start with 3-5, increase for more comprehensive answers

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM hosting
- [LangChain](https://langchain.com) for RAG components
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
