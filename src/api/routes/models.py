"""
Model endpoints for listing and getting model details.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from src.core.logger import logger
from src.inference.load_artifacts import get_artifact_loader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
@router.get("/")
async def list_models() -> Dict[str, Any]:
    """
    List all available models.
    
    Returns:
        Dictionary with list of models and total count
    """
    try:
        loader = get_artifact_loader()
        if not loader or not loader.artifacts_loaded or loader.features_df is None:
            raise HTTPException(
                status_code=503,
                detail="Artifacts not loaded. Please wait for the service to initialize."
            )
        
        # Get model information
        columns_to_return = ['model_id', 'author', 'downloads', 'likes', 'task', 'library_name']
        available_columns = [col for col in columns_to_return if col in loader.features_df.columns]
        
        models = loader.features_df[available_columns].to_dict('records')
        
        return {
            "models": models,
            "total": len(models)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while listing models"
        )


@router.get("/{model_id}")
async def get_model_detail(model_id: str) -> Dict[str, Any]:
    """
    Get details for a specific model.
    
    Args:
        model_id: Model ID to get details for
        
    Returns:
        Dictionary with model_id and features
    """
    try:
        loader = get_artifact_loader()
        if not loader or not loader.artifacts_loaded or loader.features_df is None:
            raise HTTPException(
                status_code=503,
                detail="Artifacts not loaded. Please wait for the service to initialize."
            )
        
        # Find model in features DataFrame
        if 'model_id' not in loader.features_df.columns:
            raise HTTPException(
                status_code=500,
                detail="Model ID column not found in features"
            )
        
        model_data = loader.features_df[loader.features_df['model_id'] == model_id]
        
        if len(model_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        model_info = model_data.iloc[0].to_dict()
        
        return {
            "model_id": model_id,
            "features": model_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while getting model details"
        )
