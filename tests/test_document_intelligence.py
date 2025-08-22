"""
Tests for Enhanced Document Intelligence system
"""

from datetime import datetime
from app.document_intelligence import (
    DocumentTypeDetector,
    MetadataEnricher,
    SmartChunker,
    DocumentType,
    DocumentMetadata,
)


class TestDocumentTypeDetector:
    """Tests for document type detection"""

    def setup_method(self):
        self.detector = DocumentTypeDetector()

    def test_python_file_detection(self):
        """Should detect Python files from extension"""
        assert self.detector.detect_type("script.py") == DocumentType.CODE_PYTHON
        assert self.detector.detect_type("main.py") == DocumentType.CODE_PYTHON

    def test_javascript_file_detection(self):
        """Should detect JavaScript/TypeScript files"""
        assert self.detector.detect_type("app.js") == DocumentType.CODE_JAVASCRIPT
        assert self.detector.detect_type("app.ts") == DocumentType.CODE_TYPESCRIPT
        assert (
            self.detector.detect_type("component.jsx") == DocumentType.CODE_JAVASCRIPT
        )
        assert (
            self.detector.detect_type("component.tsx") == DocumentType.CODE_TYPESCRIPT
        )

    def test_markdown_file_detection(self):
        """Should detect markdown files"""
        assert self.detector.detect_type("README.md") == DocumentType.MARKDOWN
        assert self.detector.detect_type("docs.markdown") == DocumentType.MARKDOWN

    def test_content_based_detection(self):
        """Should detect type from content when extension is unknown"""
        python_content = "def hello():\n    print('world')"
        assert (
            self.detector.detect_type("unknown_file", python_content)
            == DocumentType.CODE_PYTHON
        )

        js_content = "function hello() {\n  console.log('world');\n}"
        assert (
            self.detector.detect_type("unknown_file", js_content)
            == DocumentType.CODE_JAVASCRIPT
        )

        json_content = '{"key": "value", "number": 42}'
        assert (
            self.detector.detect_type("unknown_file", json_content)
            == DocumentType.CODE_JSON
        )

    def test_unknown_file_fallback(self):
        """Should return unknown for unrecognized files"""
        assert self.detector.detect_type("file.xyz") == DocumentType.UNKNOWN


class TestMetadataEnricher:
    """Tests for metadata enrichment"""

    def setup_method(self):
        self.enricher = MetadataEnricher()

    def test_basic_metadata_extraction(self):
        """Should extract basic file metadata"""
        content = "This is a test document with some content."
        metadata = self.enricher.extract_metadata("test.txt", content)

        assert metadata.filename == "test.txt"
        assert metadata.document_type == DocumentType.TEXT_PLAIN
        assert metadata.word_count == 8  # "This is a test document with some content"
        assert metadata.line_count == 1
        assert metadata.content_hash is not None

    def test_python_code_analysis(self):
        """Should analyze Python code structure"""
        python_code = """
import os
from datetime import datetime

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

def hello_world():
    print("Hello!")

def goodbye():
    print("Goodbye!")
"""
        metadata = self.enricher.extract_metadata("calc.py", python_code)

        assert metadata.document_type == DocumentType.CODE_PYTHON
        assert metadata.programming_language == "Python"
        assert "Calculator" in metadata.classes_detected
        assert "hello_world" in metadata.functions_detected
        assert "goodbye" in metadata.functions_detected
        assert "os" in metadata.imports_detected
        assert "datetime" in metadata.imports_detected

    def test_javascript_code_analysis(self):
        """Should analyze JavaScript code structure"""
        js_code = """
import React from 'react';
import { useState } from 'react';

function MyComponent() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}

const helper = () => {
    return "helper";
};

export default MyComponent;
"""
        metadata = self.enricher.extract_metadata("component.js", js_code)

        assert metadata.document_type == DocumentType.CODE_JAVASCRIPT
        assert metadata.programming_language == "JavaScript"
        assert "MyComponent" in metadata.functions_detected
        assert "helper" in metadata.functions_detected
        assert "react" in metadata.imports_detected

    def test_markdown_analysis(self):
        """Should analyze markdown structure and extract headings"""
        markdown_content = """
# Main Title

This is the introduction section.

## Installation

Steps to install the software.

## Usage

How to use the software.

### Advanced Usage

More complex examples.
"""
        metadata = self.enricher.extract_metadata("README.md", markdown_content)

        assert metadata.document_type == DocumentType.MARKDOWN
        assert metadata.title == "Main Title"
        assert "Installation" in metadata.topics
        assert "Usage" in metadata.topics
        assert "Advanced Usage" in metadata.topics

    def test_keyword_extraction(self):
        """Should extract relevant technical keywords"""
        technical_content = """
This document discusses machine learning algorithms and neural networks.
We'll cover how to build APIs using Python and integrate with databases.
The microservice architecture uses Docker containers deployed on AWS.
"""
        metadata = self.enricher.extract_metadata("tech.txt", technical_content)

        # Should find some technical keywords
        keywords_found = any(
            kw in metadata.keywords
            for kw in [
                "machine learning",
                "neural network",
                "api",
                "python",
                "database",
                "docker",
                "aws",
            ]
        )
        assert keywords_found, f"No technical keywords found in: {metadata.keywords}"


class TestSmartChunker:
    """Tests for smart document chunking"""

    def setup_method(self):
        self.chunker = SmartChunker(chunk_size=200, overlap=50)

    def test_markdown_chunking_by_sections(self):
        """Should chunk markdown by sections/headings"""
        markdown_content = """
# Introduction

This is the introduction section with some content.

## Method

This section describes the methodology used.

## Results

Here are the results of the study.

### Detailed Analysis

More detailed analysis of the results.
"""
        metadata = DocumentMetadata(
            filename="test.md",
            filepath="test.md",
            file_size=100,
            created_date=None,
            modified_date=None,
            document_type=DocumentType.MARKDOWN,
            title="Test Document",
        )

        chunks = self.chunker.chunk_document(markdown_content, metadata)

        # Should create separate chunks for each section
        assert len(chunks) >= 4  # Introduction, Method, Results, Detailed Analysis

        # Check that chunks have section headings
        headings = [chunk["metadata"].get("section_heading") for chunk in chunks]
        assert "Introduction" in headings
        assert "Method" in headings
        assert "Results" in headings

    def test_python_code_chunking(self):
        """Should chunk Python code by functions and classes"""
        python_code = """
import os
import sys

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        return item.upper()

def helper_function():
    return "helper"

def main():
    processor = DataProcessor()
    result = processor.process("test")
    print(result)

if __name__ == "__main__":
    main()
"""
        metadata = DocumentMetadata(
            filename="processor.py",
            filepath="processor.py",
            file_size=100,
            created_date=None,
            modified_date=None,
            document_type=DocumentType.CODE_PYTHON,
            programming_language="Python",
        )

        chunks = self.chunker.chunk_document(python_code, metadata)

        # Should create chunks for different code blocks
        assert len(chunks) >= 1

        # Check chunk types
        chunk_types = [chunk["metadata"]["chunk_type"] for chunk in chunks]
        assert any(
            chunk_type in ["code_block", "function", "class", "declarations"]
            for chunk_type in chunk_types
        )

    def test_generic_text_chunking_with_overlap(self):
        """Should chunk generic text with proper overlap"""
        long_text = "This is a long text. " * 50  # Create text longer than chunk_size

        metadata = DocumentMetadata(
            filename="long.txt",
            filepath="long.txt",
            file_size=len(long_text),
            created_date=None,
            modified_date=None,
            document_type=DocumentType.TEXT_PLAIN,
        )

        chunks = self.chunker.chunk_document(long_text, metadata)

        # Should create multiple chunks for long text
        assert len(chunks) > 1

        # Check that chunks have overlap (if more than one chunk)
        if len(chunks) > 1:
            # Last part of first chunk should appear in second chunk
            first_chunk_end = chunks[0]["text"][-50:]
            second_chunk_start = chunks[1]["text"][:50:]
            # Should have some overlapping words
            first_words = set(first_chunk_end.split())
            second_words = set(second_chunk_start.split())
            overlap = first_words & second_words
            assert len(overlap) > 0, "Chunks should have overlapping content"

    def test_chunk_metadata_preservation(self):
        """Should preserve document metadata in chunk metadata"""
        content = "Simple test content for chunking."
        metadata = DocumentMetadata(
            filename="test.txt",
            filepath="/path/to/test.txt",
            file_size=100,
            created_date=datetime.now(),
            modified_date=datetime.now(),
            document_type=DocumentType.TEXT_PLAIN,
            title="Test Document",
        )

        chunks = self.chunker.chunk_document(content, metadata)

        assert len(chunks) == 1
        chunk_meta = chunks[0]["metadata"]

        # Original metadata should be preserved
        assert chunk_meta["filename"] == metadata.filename
        assert chunk_meta["filepath"] == metadata.filepath
        assert chunk_meta["document_type"] == metadata.document_type
        assert chunk_meta["title"] == metadata.title

        # Chunk-specific metadata should be added
        assert "chunk_index" in chunk_meta
        assert "chunk_type" in chunk_meta


class TestDocumentIntelligenceIntegration:
    """Integration tests for the complete document intelligence pipeline"""

    def test_complete_pipeline(self):
        """Should process a document through the complete pipeline"""
        # Test with a Python file
        python_content = """
#!/usr/bin/env python3
'''
A simple calculator module for mathematical operations.
'''

import math
from typing import Union

class Calculator:
    '''A basic calculator class'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        '''Add two numbers'''
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        '''Multiply two numbers'''
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def main():
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.multiply(4, 5))

if __name__ == "__main__":
    main()
"""

        # Step 1: Detect document type
        detector = DocumentTypeDetector()
        doc_type = detector.detect_type("calculator.py", python_content)
        assert doc_type == DocumentType.CODE_PYTHON

        # Step 2: Extract metadata
        enricher = MetadataEnricher()
        metadata = enricher.extract_metadata("calculator.py", python_content)

        assert metadata.document_type == DocumentType.CODE_PYTHON
        assert metadata.programming_language == "Python"
        assert "Calculator" in metadata.classes_detected
        assert "add" in metadata.functions_detected
        assert "multiply" in metadata.functions_detected
        assert "main" in metadata.functions_detected

        # Step 3: Create smart chunks
        chunker = SmartChunker()
        chunks = chunker.chunk_document(python_content, metadata)

        assert len(chunks) >= 1

        # All chunks should have proper metadata
        for chunk in chunks:
            chunk_meta = chunk["metadata"]
            assert chunk_meta["document_type"] == DocumentType.CODE_PYTHON
            assert chunk_meta["programming_language"] == "Python"
            assert "chunk_index" in chunk_meta
            assert len(chunk["text"]) > 0
