"""
Similar models endpoint for finding similar models using FAISS.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.inference.load_artifacts import get_artifact_loader
from src.recommender.similarity_search import SimilaritySearch

router = APIRouter(prefix="/similar", tags=["similarity"])


class SimilarModelsRequest(BaseModel):
    """Request schema for finding similar models."""
    model_id: str = Field(..., description="Model ID to find similar models for")
    num_similar: int = Field(default=5, ge=1, le=20, description="Number of similar models to return")


@router.post("")
@router.post("/")
async def find_similar(request: SimilarModelsRequest) -> Dict[str, Any]:
    """
    Find similar models using FAISS similarity search.
    
    Args:
        request: SimilarModelsRequest with model_id and num_similar
        
    Returns:
        Dictionary with list of similar models
    """
    try:
        loader = get_artifact_loader()
        if not loader or not loader.artifacts_loaded:
            raise HTTPException(
                status_code=503,
                detail="Artifacts not loaded. Please wait for the service to initialize."
            )
        
        if loader.faiss_index is None:
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Similarity search is not enabled."
            )
        
        if loader.model_id_mapping is None:
            raise HTTPException(
                status_code=500,
                detail="Model ID mapping not loaded"
            )
        
        # Create similarity search instance
        similarity_search = SimilaritySearch(
            faiss_index=loader.faiss_index,
            embeddings=loader.embeddings
        )
        
        # Find similar models
        similar = similarity_search.find_similar_models(
            model_id=request.model_id,
            model_id_mapping=loader.model_id_mapping,
            features_df=loader.features_df,
            k=request.num_similar
        )
        
        if not similar:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found or no similar models available"
            )
        
        return {"similar_models": similar}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while finding similar models"
        )
