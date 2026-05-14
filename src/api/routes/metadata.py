"""
Metadata endpoint for getting API and model catalog information.
"""
from fastapi import APIRouter

from src.api.schemas.response_schema import MetadataResponse
from src.core.config import settings
from src.core.logger import logger
from src.core.constants import LLM_TASKS, MODEL_METADATA_FIELDS
from src.inference.load_artifacts import get_artifact_loader

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("", response_model=MetadataResponse)
@router.get("/", response_model=MetadataResponse)
async def get_metadata() -> MetadataResponse:
    """
    Get metadata about available models, features, and API capabilities.
    
    Returns information about:
    - Number of available models
    - Supported tasks
    - Available features for filtering
    - API version and catalog info
    """
    try:
        logger.info("Metadata request received")
        
        loader = get_artifact_loader()
        
        # Get model count from features DataFrame if available
        available_models_count = 0
        if loader and loader.artifacts_loaded and loader.features_df is not None:
            available_models_count = len(loader.features_df)
        
        # Get supported tasks from deployment config if available
        supported_tasks = LLM_TASKS
        if loader and loader.deployment_config:
            data_info = loader.deployment_config.get('data_info', {})
            if 'benchmarks' in data_info:
                supported_tasks = data_info['benchmarks']
        
        # Get available features from features DataFrame if available
        available_features = MODEL_METADATA_FIELDS
        if loader and loader.artifacts_loaded and loader.features_df is not None:
            available_features = list(loader.features_df.columns)
        
        # Build catalog info
        model_catalog_info = {
            "status": "loaded" if (loader and loader.artifacts_loaded) else "not_loaded",
            "artifacts_loaded": loader.artifacts_loaded if loader else False,
            "faiss_available": loader.faiss_index is not None if loader else False,
            "scaler_available": loader.feature_scaler is not None if loader else False,
        }
        
        if loader and loader.deployment_config:
            model_catalog_info.update({
                "performance": loader.deployment_config.get('performance', {}),
                "model_info": loader.deployment_config.get('model_info', {}),
            })
        
        return MetadataResponse(
            available_models_count=available_models_count,
            supported_tasks=supported_tasks,
            available_features=available_features,
            model_catalog_info=model_catalog_info,
            api_version=settings.API_VERSION
        )
        
    except Exception as e:
        logger.error(f"Error retrieving metadata: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving metadata"
        )
