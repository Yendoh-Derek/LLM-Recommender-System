"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from src.core.config import settings, ensure_directories
from src.core.logger import logger
from src.api.routes import health, recommend, metadata, models, similar
from src.inference.load_artifacts import load_all_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting LLM Recommender API")
    ensure_directories()
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    
    # Load artifacts
    logger.info("Loading artifacts...")
    try:
        loader = load_all_artifacts()
        if loader.artifacts_loaded:
            logger.info("All artifacts loaded successfully")
        else:
            logger.error("Failed to load critical artifacts")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}", exc_info=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Recommender API")


# Create FastAPI application instance
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
        }
    )


# Register routes
app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(metadata.router)
app.include_router(models.router)
app.include_router(similar.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Recommender API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/health/ready",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
