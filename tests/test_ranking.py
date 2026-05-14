from src.recommender.ranking import rank_candidates


def test_rank_candidates_orders_by_score():
    candidates = [
        {"model_id": "a", "score": 0.1},
        {"model_id": "b", "score": 0.9},
        {"model_id": "c", "score": 0.5},
    ]

    ranked = rank_candidates(candidates)
    assert ranked[0]["model_id"] == "b"
    assert ranked[-1]["model_id"] == "a"
