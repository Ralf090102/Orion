"""
Vector operations and similarity metrics for RAG systems.

This module provides common vector operations used in retrieval-augmented generation,
including cosine similarity, dot product, and Euclidean distance calculations.
"""

import numpy as np
from typing import List, Union


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Distance value (lower is more similar)
    """
    return np.linalg.norm(vec1 - vec2)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def batch_cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query and multiple documents.
    
    Args:
        query: Query vector of shape (d,)
        documents: Document vectors of shape (n, d)
        
    Returns:
        Similarity scores of shape (n,)
    """
    # Normalize vectors
    query_norm = normalize_vector(query)
    docs_norm = np.apply_along_axis(normalize_vector, 1, documents)
    
    # Compute dot product
    similarities = np.dot(docs_norm, query_norm)
    
    return similarities


class VectorIndex:
    """Simple in-memory vector index for testing."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
    
    def add(self, vector: np.ndarray, metadata: dict):
        """Add a vector with metadata to the index."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        self.vectors.append(normalize_vector(vector))
        self.metadata.append(metadata)
    
    def search(self, query: np.ndarray, k: int = 5) -> List[tuple]:
        """Search for k most similar vectors."""
        if len(self.vectors) == 0:
            return []
        
        vectors_array = np.array(self.vectors)
        similarities = batch_cosine_similarity(query, vectors_array)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = [
            (self.metadata[i], similarities[i])
            for i in top_k_indices
        ]
        
        return results
