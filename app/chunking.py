"""
Advanced chunking strategies for better RAG performance.
"""
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class SemanticChunker:
    """
    Chunks documents based on semantic boundaries rather than just character count.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def chunk_by_headers(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk by markdown headers or document structure."""
        # Detect headers (markdown style)
        header_pattern = r'^(#{1,6}\s+.+)$'
        lines = text.split('\n')
        
        chunks = []
        current_chunk = []
        current_header = None
        
        for line in lines:
            if re.match(header_pattern, line):
                # Save previous chunk if it exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                **metadata,
                                'header': current_header,
                                'chunk_type': 'semantic_header'
                            }
                        })
                
                # Start new chunk
                current_header = line
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'header': current_header,
                        'chunk_type': 'semantic_header'
                    }
                })
        
        return chunks
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Apply semantic chunking strategy based on document type.
        """
        source = metadata.get('source', '')
        
        # Try semantic chunking first
        if any(indicator in text.lower() for indicator in ['#', '##', '###']):
            semantic_chunks = self.chunk_by_headers(text, metadata)
            if semantic_chunks:
                return semantic_chunks
        
        # Fallback to standard chunking
        base_chunks = self.base_splitter.split_text(text)
        return [{
            'text': chunk,
            'metadata': {
                **metadata,
                'chunk_type': 'standard'
            }
        } for chunk in base_chunks]
