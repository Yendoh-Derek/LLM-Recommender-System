"""
ranking.py
Ranking utilities for recommendations.
"""
from typing import Any, Dict, List


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort candidate recommendations by descending score."""
    return sorted(candidates, key=lambda candidate: candidate.get("score", 0.0), reverse=True)
