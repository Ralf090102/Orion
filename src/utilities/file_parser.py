"""
File Parser Utility

Parse uploaded files and extract text content for temporary context injection.
Does NOT add files to vector database - used for one-time file analysis.
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

from src.core.ingest import DocumentProcessor
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)


def parse_uploaded_file(
    file_data: dict,
    config: Optional[OrionConfig] = None
) -> tuple[str, dict]:
    """
    Parse an uploaded file and extract text content.
    
    Args:
        file_data: Dictionary with 'value' (base64), 'mime', 'name'
        config: Optional Orion configuration
        
    Returns:
        Tuple of (extracted_text, file_metadata)
        
    Raises:
        ValueError: If file format not supported or parsing fails
    """
    try:
        # Decode base64 file content
        if file_data.get("type") != "base64":
            raise ValueError("Only base64 encoded files are supported")
        
        file_content = base64.b64decode(file_data["value"])
        file_name = file_data.get("name", "uploaded_file")
        mime_type = file_data.get("mime", "")
        
        # Determine file extension from name or MIME type
        file_path = Path(file_name)
        file_ext = file_path.suffix.lower()
        
        if not file_ext:
            # Try to infer from MIME type
            mime_to_ext = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/msword": ".doc",
                "text/plain": ".txt",
                "text/markdown": ".md",
                "application/json": ".json",
                "text/csv": ".csv",
            }
            file_ext = mime_to_ext.get(mime_type, ".txt")
            file_name = file_path.stem + file_ext
        
        logger.info(f"Parsing uploaded file: {file_name} ({mime_type}, {len(file_content)} bytes)")
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=file_ext,
            delete=False
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = Path(temp_file.name)
        
        try:
            # Use existing document processor to extract text
            processor = DocumentProcessor(config=config)
            documents = processor.load_document(temp_path)
            
            # Combine all document chunks into single text
            extracted_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Build metadata
            metadata = {
                "file_name": file_name,
                "file_size": len(file_content),
                "mime_type": mime_type,
                "file_type": file_ext[1:] if file_ext.startswith(".") else file_ext,
                "chunks_extracted": len(documents),
                "text_length": len(extracted_text),
            }
            
            logger.info(
                f"Successfully extracted {len(extracted_text)} chars "
                f"from {file_name} ({len(documents)} chunks)"
            )
            
            return extracted_text, metadata
            
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"Failed to parse uploaded file {file_data.get('name')}: {e}")
        raise ValueError(f"Failed to parse file: {str(e)}")


def format_file_context(text: str, metadata: dict, max_length: int = 10000) -> str:
    """
    Format extracted file text for context injection.
    
    Args:
        text: Extracted file text
        metadata: File metadata
        max_length: Maximum text length to include
        
    Returns:
        Formatted context string
    """
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... truncated {len(text) - max_length} characters ...]"
    
    formatted = f"""
**📎 Attached File: {metadata.get('file_name', 'Unknown')}**
File Type: {metadata.get('file_type', 'Unknown')}
Size: {metadata.get('file_size', 0)} bytes

**Content:**
{text}

---
""".strip()
    
    return formatted


def parse_multiple_files(
    files: list[dict],
    config: Optional[OrionConfig] = None,
    max_per_file: int = 5000
) -> tuple[str, list[dict]]:
    """
    Parse multiple uploaded files and combine their content.
    
    Args:
        files: List of file data dictionaries
        config: Optional Orion configuration
        max_per_file: Max characters per file
        
    Returns:
        Tuple of (combined_context, list of metadata dicts)
    """
    all_contexts = []
    all_metadata = []
    
    for file_data in files:
        try:
            text, metadata = parse_uploaded_file(file_data, config)
            formatted_context = format_file_context(text, metadata, max_per_file)
            all_contexts.append(formatted_context)
            all_metadata.append(metadata)
        except Exception as e:
            logger.warning(f"Skipping file {file_data.get('name')}: {e}")
            # Add error metadata
            all_metadata.append({
                "file_name": file_data.get("name", "Unknown"),
                "error": str(e),
                "status": "failed"
            })
    
    combined_context = "\n\n".join(all_contexts)
    
    return combined_context, all_metadata
