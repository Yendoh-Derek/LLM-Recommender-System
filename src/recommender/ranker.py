"""
Ranking module for scoring and ranking model recommendations.
Combines RL agent scores with similarity scores and other factors.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.core.logger import logger


class Ranker:
    """
    Ranking engine that combines multiple scoring signals.
    
    Combines:
    - RL agent scores (from SAC policy)
    - Similarity scores (from embedding search)
    - User preference matching
    - Model metadata signals (popularity, quality indicators)
    
    TODO: Replace with actual ranking algorithm when artifacts are available.
    """
    
    def __init__(
        self,
        rl_weight: float = 0.6,
        similarity_weight: float = 0.3,
        metadata_weight: float = 0.1
    ):
        """
        Initialize the ranker with scoring weights.
        
        Args:
            rl_weight: Weight for RL agent scores (default: 0.6)
            similarity_weight: Weight for similarity scores (default: 0.3)
            metadata_weight: Weight for metadata signals (default: 0.1)
        """
        self.rl_weight = rl_weight
        self.similarity_weight = similarity_weight
        self.metadata_weight = metadata_weight
        
        # Ensure weights sum to 1
        total_weight = rl_weight + similarity_weight + metadata_weight
        if total_weight > 0:
            self.rl_weight /= total_weight
            self.similarity_weight /= total_weight
            self.metadata_weight /= total_weight
        
        logger.info(
            f"Initialized Ranker with weights: "
            f"RL={self.rl_weight:.2f}, Similarity={self.similarity_weight:.2f}, "
            f"Metadata={self.metadata_weight:.2f}"
        )
    
    def compute_metadata_score(
        self,
        model_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute score based on model metadata and user preferences.
        
        Args:
            model_metadata: Model metadata dictionary
            user_context: Optional user context with preferences
            
        Returns:
            Metadata-based score [0, 1]
        """
        score = 0.0
        
        # Popularity signals
        downloads = model_metadata.get("downloads", 0)
        likes = model_metadata.get("likes", 0)
        
        # Normalize downloads (assuming max ~10M downloads)
        download_score = min(downloads / 10_000_000.0, 1.0)
        like_score = min(likes / 100_000.0, 1.0)
        
        score += 0.4 * download_score + 0.2 * like_score
        
        # User preference matching
        if user_context:
            preferences = user_context.get("preferences", {})
            constraints = user_context.get("constraints", {})
            
            # Task matching
            if "task" in preferences:
                if model_metadata.get("task") == preferences["task"]:
                    score += 0.2
            
            # License matching
            if "license" in preferences:
                if model_metadata.get("license") == preferences["license"]:
                    score += 0.1
            
            # Size constraint matching
            if "max_model_size_gb" in constraints:
                max_size = constraints["max_model_size_gb"]
                model_size = model_metadata.get("size_on_disk_gb", 0)
                if model_size <= max_size:
                    score += 0.1
        
        return min(score, 1.0)
    
    def combine_scores(
        self,
        rl_scores: List[float],
        similarity_scores: List[float],
        metadata_scores: List[float]
    ) -> List[float]:
        """
        Combine multiple scoring signals into final scores.
        
        Args:
            rl_scores: RL agent scores
            similarity_scores: Similarity search scores
            metadata_scores: Metadata-based scores
            
        Returns:
            Combined final scores
        """
        if len(rl_scores) != len(similarity_scores) or len(rl_scores) != len(metadata_scores):
            raise ValueError("All score lists must have the same length")
        
        # Normalize each score list to [0, 1]
        rl_norm = self._normalize_scores(rl_scores)
        sim_norm = self._normalize_scores(similarity_scores)
        meta_norm = self._normalize_scores(metadata_scores)
        
        # Weighted combination
        combined = [
            self.rl_weight * rl + self.similarity_weight * sim + self.metadata_weight * meta
            for rl, sim, meta in zip(rl_norm, sim_norm, meta_norm)
        ]
        
        return combined
    
    def rank_models(
        self,
        model_indices: List[int],
        rl_scores: List[float],
        similarity_scores: List[float],
        model_metadata_list: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank models using combined scoring signals.
        
        Args:
            model_indices: List of model indices
            rl_scores: RL agent scores for each model
            similarity_scores: Similarity scores for each model
            model_metadata_list: List of model metadata dictionaries
            user_context: Optional user context
            top_k: Optional limit on number of results
            
        Returns:
            List of (model_index, final_score) tuples, sorted by score descending
        """
        if len(model_indices) != len(rl_scores) or len(model_indices) != len(similarity_scores):
            raise ValueError("All input lists must have the same length")
        
        # Compute metadata scores
        metadata_scores = [
            self.compute_metadata_score(meta, user_context)
            for meta in model_metadata_list
        ]
        
        # Combine scores
        final_scores = self.combine_scores(rl_scores, similarity_scores, metadata_scores)
        
        # Create (index, score) pairs
        ranked = list(zip(model_indices, final_scores))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k:
            ranked = ranked[:top_k]
        
        logger.debug(f"Ranked {len(ranked)} models")
        return ranked
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

