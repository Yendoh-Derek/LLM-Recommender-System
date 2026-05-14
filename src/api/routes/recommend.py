"""
Recommendation endpoint for getting LLM recommendations.
Real implementation using trained SAC agent.
"""
from fastapi import APIRouter, HTTPException
from typing import List
import time
import numpy as np
import pandas as pd

from src.api.schemas.request_schema import RecommendationRequest
from src.api.schemas.response_schema import RecommendationResponse, ModelRecommendation, ModelMetadata
from src.core.config import settings
from src.core.logger import logger
from src.inference.load_artifacts import get_artifact_loader
from src.inference.preprocessing import Preprocessor

router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get LLM recommendations based on user query and context.
    
    This endpoint accepts:
    - query_text: Description of what the user needs
    - user_context: Optional user preferences and constraints
    - top_k: Number of recommendations (defaults to config value)
    
    Returns ranked list of model recommendations with scores and explanations.
    """
    try:
        start_time = time.time()
        logger.info(f"Recommendation request received: query='{request.query_text[:50]}...'")
        
        # Get artifact loader
        loader = get_artifact_loader()
        if not loader or not loader.artifacts_loaded:
            raise HTTPException(
                status_code=503,
                detail="Artifacts not loaded. Please wait for the service to initialize."
            )
        
        # Determine top_k
        top_k = request.top_k or settings.DEFAULT_TOP_K
        top_k = min(top_k, settings.MAX_TOP_K)  # Enforce max limit
        
        # Create preprocessor
        preprocessor = Preprocessor(
            feature_scaler=loader.feature_scaler,
            feature_columns=['cost_score', 'popularity_score', 'recency_score', 
                           'diversity_score', 'performance_score']
        )
        
        # Create state vector from features
        state = preprocessor.preprocess_request(loader.features_df, step_progress=0.0)
        
        # Get action (scores) from SAC agent
        action_scores = loader.rl_policy.predict(state)
        
        # Select top-K models based on action scores
        top_k_indices = np.argsort(action_scores)[-top_k:][::-1]
        
        # Build recommendations
        recommendations = []
        for rank, idx in enumerate(top_k_indices, start=1):
            model_info = loader.features_df.iloc[idx]
            score = float(action_scores[idx])
            
            # Get model_id from mapping if available
            model_id = str(idx)
            if loader.model_id_mapping and model_id in loader.model_id_mapping:
                model_id = loader.model_id_mapping[model_id]
            elif 'model_id' in model_info:
                model_id = model_info['model_id']
            
            # Create model metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_info.get('model_name', model_id),
                description=model_info.get('description', ''),
                author=model_info.get('author', ''),
                downloads=int(model_info.get('downloads', 0)) if pd.notna(model_info.get('downloads')) else 0,
                likes=int(model_info.get('likes', 0)) if pd.notna(model_info.get('likes')) else 0,
                tags=model_info.get('tags', []) if isinstance(model_info.get('tags'), list) else [],
                task=model_info.get('task', ''),
                library_name=model_info.get('library_name', ''),
                pipeline_tag=model_info.get('pipeline_tag'),
                base_model=model_info.get('base_model'),
                model_type=model_info.get('model_type'),
                size_on_disk_gb=float(model_info.get('size_on_disk_gb', 0.0)) if pd.notna(model_info.get('size_on_disk_gb')) else 0.0,
                parameters=model_info.get('parameters'),
                quantization=model_info.get('quantization'),
                license=model_info.get('license'),
                huggingface_url=model_info.get('huggingface_url', f"https://huggingface.co/{model_id}")
            )
            
            # Generate explanation
            explanation = (
                f"This model is recommended (score: {score:.3f}) because it matches your query. "
                f"It has a performance score of {model_info.get('performance_score', 0):.2f} "
                f"and cost score of {model_info.get('cost_score', 0):.2f}."
            )
            
            recommendation = ModelRecommendation(
                model=model_metadata,
                score=score,
                explanation=explanation,
                rank=rank
            )
            recommendations.append(recommendation)
        
        inference_time = (time.time() - start_time) * 1000
        logger.info(f"Generated {len(recommendations)} recommendations in {inference_time:.2f}ms")
        
        return RecommendationResponse(
            recommendations=recommendations,
            query_text=request.query_text,
            total_results=len(recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating recommendations"
        )
