"""
Similarity search module using FAISS for finding similar models.
Real implementation using FAISS index from artifacts.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from src.core.logger import logger


class SimilaritySearch:
    """
    FAISS-based similarity search for model recommendations.
    
    Uses FAISS index for fast similarity search.
    """
    
    def __init__(self, faiss_index: Optional[faiss.Index] = None, embeddings: Optional[np.ndarray] = None):
        """
        Initialize similarity search.
        
        Args:
            faiss_index: FAISS index for similarity search
            embeddings: Model embeddings matrix (fallback if FAISS not available)
        """
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.num_models = faiss_index.ntotal if faiss_index is not None else (embeddings.shape[0] if embeddings is not None else 0)
        
        if faiss_index is not None:
            logger.info(f"Initialized SimilaritySearch with FAISS index: {self.num_models} models")
        elif embeddings is not None:
            logger.info(f"Initialized SimilaritySearch with embeddings: {self.num_models} models")
        else:
            logger.warning("SimilaritySearch initialized without index or embeddings")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for similar models given a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity score threshold (not used with FAISS distance)
            
        Returns:
            List of (model_index, similarity_score) tuples, sorted by score descending
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded. Cannot perform similarity search.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure query embedding matches index dimension
        if query_embedding.shape[1] != self.faiss_index.d:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} "
                f"does not match FAISS index dimension {self.faiss_index.d}"
            )
        
        # Search FAISS
        k = top_k if top_k else min(10, self.num_models)
        distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)
        
        # Convert distances to similarities (inverse of distance)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = float(1 / (1 + dist)) if dist >= 0 else 0.0
            if threshold is None or similarity >= threshold:
                results.append((int(idx), similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Found {len(results)} similar models")
        return results
    
    def find_similar_models(
        self,
        model_id: str,
        model_id_mapping: Dict[str, str],
        features_df: Any,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar models for a given model ID.
        
        Args:
            model_id: Model ID to find similar models for
            model_id_mapping: Mapping from index to model_id
            features_df: DataFrame with model features
            k: Number of similar models to return
            
        Returns:
            List of similar model dictionaries with model_id and similarity
        """
        # Reverse mapping: model_id -> index
        index_to_id = {v: int(k) for k, v in model_id_mapping.items()}
        
        if model_id not in index_to_id:
            logger.warning(f"Model {model_id} not found in mapping")
            return []
        
        model_idx = index_to_id[model_id]
        
        # Get embedding for this model
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded. Cannot find similar models.")
        
        if model_idx >= len(self.embeddings):
            logger.error(f"Model index {model_idx} out of range")
            return []
        
        query_embedding = self.embeddings[model_idx:model_idx+1]
        
        # Search for similar models
        similar = self.search(query_embedding, top_k=k + 1)  # +1 to exclude self
        
        # Filter out the query model itself and format results
        results = []
        for idx, similarity in similar:
            if idx == model_idx:
                continue  # Skip the query model itself
            
            # Get model_id from index
            model_id_str = str(idx)
            if model_id_str in model_id_mapping:
                similar_model_id = model_id_mapping[model_id_str]
                
                # Get model info from features_df if available
                model_info = {}
                if features_df is not None and 'model_id' in features_df.columns:
                    model_data = features_df[features_df['model_id'] == similar_model_id]
                    if len(model_data) > 0:
                        model_info = model_data.iloc[0].to_dict()
                
                results.append({
                    "model_id": similar_model_id,
                    "similarity": similarity,
                    **model_info
                })
        
        return results[:k]  # Return top k (excluding self)
    
    def get_model_embedding(self, model_index: int) -> np.ndarray:
        """
        Get embedding vector for a specific model.
        
        Args:
            model_index: Model index
            
        Returns:
            Model embedding vector
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded.")
        
        if model_index < 0 or model_index >= len(self.embeddings):
            raise IndexError(f"Model index {model_index} out of range [0, {len(self.embeddings)})")
        
        return self.embeddings[model_index]
    
    def batch_search(
        self,
        query_embeddings: List[np.ndarray],
        top_k: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Search for multiple queries.
        
        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of results per query
            
        Returns:
            List of search results, one per query
        """
        results = []
        for query_emb in query_embeddings:
            query_results = self.search(query_emb, top_k=top_k)
            results.append(query_results)
        
        return results
