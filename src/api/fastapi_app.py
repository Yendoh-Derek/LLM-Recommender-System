"""
fastapi_app.py
FastAPI app for RL-powered LLM recommender system.
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

# Add other endpoints as needed
