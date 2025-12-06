"""
Health check endpoint for monitoring API status.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any

from src.core.config import settings
from src.core.logger import logger
from src.inference.load_artifacts import get_artifact_loader

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=Dict[str, Any])
@router.get("/", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with API status, version, and timestamp
    """
    logger.info("Health check requested")
    
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.API_TITLE,
    }


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    Checks if artifacts are loaded and service is ready.
    
    Returns:
        Dictionary with readiness status
    """
    logger.info("Readiness check requested")
    
    loader = get_artifact_loader()
    artifacts_loaded = loader is not None and loader.artifacts_loaded
    
    status_info = {
        "ready": artifacts_loaded,
        "status": "ready" if artifacts_loaded else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if loader:
        status_info["artifacts"] = {
            "sac_agent": loader.sac_agent is not None,
            "features": loader.features_df is not None,
            "embeddings": loader.embeddings is not None,
            "faiss_index": loader.faiss_index is not None,
            "model_mapping": loader.model_id_mapping is not None,
            "feature_scaler": loader.feature_scaler is not None,
        }
        
        if loader.features_df is not None:
            status_info["artifacts"]["num_models"] = len(loader.features_df)
    else:
        status_info["message"] = "Artifacts not yet loaded"
    
    return status_info
