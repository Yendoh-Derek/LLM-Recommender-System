"""
fastapi_app.py
FastAPI app for RL-powered LLM recommender system.
"""
from fastapi import FastAPI
from .routers.health import router as health_router
from .routers.recommend import router as recommend_router
from .routers.feedback import router as feedback_router

app = FastAPI(
    title="LLM Recommender System",
    description="A lightweight API for RL-driven model recommendations.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(feedback_router)
