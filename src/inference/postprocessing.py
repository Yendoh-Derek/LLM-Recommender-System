"""
Postprocessing module for formatting RL agent outputs into recommendations.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.core.logger import logger
from src.api.schemas.response_schema import ModelRecommendation, ModelMetadata


class Postprocessor:
    """
    Postprocessing for RL agent outputs.
    
    This class handles:
    - Converting RL scores to recommendation format
    - Generating explanations
    - Formatting metadata
    - Ranking and sorting
    """
    
    def __init__(self):
        """Initialize the postprocessor."""
        logger.info("Initialized Postprocessor")
    
    def format_recommendations(
        self,
        model_scores: List[Tuple[int, float]],
        model_catalog: Dict[str, Any],
        query_text: str,
        top_k: Optional[int] = None
    ) -> List[ModelRecommendation]:
        """
        Format RL agent scores into recommendation objects.
        
        Args:
            model_scores: List of (model_index, score) tuples from RL agent
            model_catalog: Dictionary mapping model indices to model metadata
            query_text: Original query text for explanation generation
            top_k: Optional limit on number of recommendations
            
        Returns:
            List of ModelRecommendation objects
        """
        recommendations = []
        
        # Limit to top_k if specified
        if top_k:
            model_scores = model_scores[:top_k]
        
        for rank, (model_idx, score) in enumerate(model_scores, start=1):
            # Get model metadata from catalog
            model_data = model_catalog.get(str(model_idx), {})
            
            # Generate explanation
            explanation = self._generate_explanation(
                model_data=model_data,
                score=score,
                query_text=query_text
            )
            
            # Create model metadata
            model_metadata = self._create_model_metadata(model_data)
            
            # Create recommendation
            recommendation = ModelRecommendation(
                model=model_metadata,
                score=float(score),
                explanation=explanation,
                rank=rank
            )
            
            recommendations.append(recommendation)
        
        logger.debug(f"Formatted {len(recommendations)} recommendations")
        return recommendations
    
    def _generate_explanation(
        self,
        model_data: Dict[str, Any],
        score: float,
        query_text: str
    ) -> str:
        """
        Generate explanation for why a model was recommended.
        
        Args:
            model_data: Model metadata dictionary
            score: Recommendation score
            query_text: Original query text
            
        Returns:
            Explanation string
        """
        # TODO: Enhance explanation generation with:
        # - Query matching analysis
        # - Feature importance
        # - Model strengths
        
        model_name = model_data.get("model_name", model_data.get("model_id", "this model"))
        task = model_data.get("task", "")
        downloads = model_data.get("downloads", 0)
        
        explanation_parts = [
            f"This model is recommended (score: {score:.3f}) because it matches your query.",
        ]
        
        if task:
            explanation_parts.append(f"It specializes in {task} tasks.")
        
        if downloads > 0:
            explanation_parts.append(f"It has {downloads:,} downloads, indicating popularity and reliability.")
        
        # Add query-specific reasoning if available
        if "creative" in query_text.lower() or "writing" in query_text.lower():
            explanation_parts.append("It is well-suited for creative text generation tasks.")
        
        return " ".join(explanation_parts)
    
    def _create_model_metadata(self, model_data: Dict[str, Any]) -> ModelMetadata:
        """
        Create ModelMetadata object from raw model data.
        
        Args:
            model_data: Raw model data dictionary
            
        Returns:
            ModelMetadata object
        """
        # Ensure all required fields are present
        metadata_dict = {
            "model_id": model_data.get("model_id", ""),
            "model_name": model_data.get("model_name", model_data.get("model_id", "Unknown Model")),
            "description": model_data.get("description", ""),
            "author": model_data.get("author", ""),
            "downloads": model_data.get("downloads", 0),
            "likes": model_data.get("likes", 0),
            "tags": model_data.get("tags", []),
            "task": model_data.get("task", ""),
            "library_name": model_data.get("library_name", ""),
            "pipeline_tag": model_data.get("pipeline_tag"),
            "base_model": model_data.get("base_model"),
            "model_type": model_data.get("model_type"),
            "size_on_disk_gb": model_data.get("size_on_disk_gb", 0.0),
            "parameters": model_data.get("parameters"),
            "quantization": model_data.get("quantization"),
            "license": model_data.get("license"),
            "huggingface_url": model_data.get(
                "huggingface_url",
                f"https://huggingface.co/{model_data.get('model_id', '')}"
            ),
        }
        
        return ModelMetadata(**metadata_dict)
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
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
    
    def rank_by_score(
        self,
        model_scores: List[Tuple[int, float]],
        descending: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Rank model scores.
        
        Args:
            model_scores: List of (model_index, score) tuples
            descending: If True, sort descending (highest first)
            
        Returns:
            Sorted list of (model_index, score) tuples
        """
        sorted_scores = sorted(
            model_scores,
            key=lambda x: x[1],
            reverse=descending
        )
        return sorted_scores

