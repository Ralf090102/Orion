# Test Ingestion Samples

This directory contains various sample documents for testing the Orion RAG ingestion pipeline.

## Test Files Overview

| File | Type | Purpose |
|------|------|---------|
| `sample_text.txt` | Text | Plain text document about learning principles |
| `machine_learning_basics.md` | Markdown | Formatted document with tables, code, and headers |
| `employee_data.csv` | CSV | Tabular data for testing CSV loader |
| `config_sample.json` | JSON | JSON configuration for testing JSON loader |
| `vector_operations.py` | Python | Python code with docstrings and classes |
| `react_component.jsx` | JSX | React component for testing JavaScript loader |
| `webpage_sample.html` | HTML | HTML document with tables and styling |
| `api_config.yaml` | YAML | YAML configuration file |
| `database_schema.xml` | XML | Database schema in XML format |
| `algorithms.cpp` | C++ | C++ algorithms implementation |

## Testing Instructions

### Test Individual File
```bash
python run.py ingest Test_Ingest/sample_text.txt
```

### Test Entire Directory
```bash
python run.py ingest Test_Ingest
```

### Test with Watchdog (Auto-sync)
```bash
python run.py ingest Test_Ingest --watch
```

### Clear Before Ingesting
```bash
python run.py ingest Test_Ingest --clear
```

## Expected Results

After successful ingestion:
- 10 files processed
- Multiple chunks generated per file
- Embeddings created for each chunk
- Metadata stored with file type, source, and chunk info

## Query Testing

After ingestion, test retrieval with:

```bash
# General questions
python run.py query "What is machine learning?"
python run.py query "How to calculate cosine similarity?"

# Code-related queries
python run.py query "BM25 algorithm implementation"
python run.py query "React search component"

# Data queries
python run.py query "employee salary information"
python run.py query "API configuration settings"
```

## File Coverage

This test set covers:
- ✅ Plain text (TXT)
- ✅ Markdown with formatting (MD)
- ✅ Tabular data (CSV)
- ✅ Structured data (JSON, XML, YAML)
- ✅ Programming languages (Python, JavaScript, C++, HTML)
- ✅ Various content types (documentation, code, configs, data)

## Notes

- All files are small (< 10KB) for quick testing
- Content is meaningful and related to RAG/ML topics
- Files test different loaders and chunking strategies
- Good for validating the complete ingestion pipeline
