"""
Tests for recommender modules (ranker, similarity search, recipe generator).
"""
import pytest
import numpy as np
from typing import List, Dict, Any

from src.recommender.ranker import Ranker, MockRanker
from src.recommender.similarity_search import SimilaritySearch, MockSimilaritySearch
from src.recommender.recipe_generator import RecipeGenerator, MockRecipeGenerator
from tests.conftest import sample_model_metadata, sample_model_metadata_list, sample_user_context


class TestRanker:
    """Tests for the Ranker class."""
    
    def test_ranker_init(self):
        """Test ranker initialization."""
        ranker = Ranker(rl_weight=0.6, similarity_weight=0.3, metadata_weight=0.1)
        # Use approximate equality due to floating-point precision
        assert abs(ranker.rl_weight - 0.6) < 0.01
        assert abs(ranker.similarity_weight - 0.3) < 0.01
        assert abs(ranker.metadata_weight - 0.1) < 0.01
    
    def test_ranker_init_normalizes_weights(self):
        """Test that ranker normalizes weights to sum to 1."""
        ranker = Ranker(rl_weight=0.6, similarity_weight=0.3, metadata_weight=0.1)
        total = ranker.rl_weight + ranker.similarity_weight + ranker.metadata_weight
        assert abs(total - 1.0) < 0.01
    
    def test_compute_metadata_score(self, sample_model_metadata, sample_user_context):
        """Test metadata score computation."""
        ranker = Ranker()
        score = ranker.compute_metadata_score(sample_model_metadata, sample_user_context)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_compute_metadata_score_no_context(self, sample_model_metadata):
        """Test metadata score computation without user context."""
        ranker = Ranker()
        score = ranker.compute_metadata_score(sample_model_metadata, None)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_combine_scores(self):
        """Test score combination."""
        ranker = Ranker(rl_weight=0.5, similarity_weight=0.3, metadata_weight=0.2)
        
        rl_scores = [0.8, 0.6, 0.4]
        similarity_scores = [0.7, 0.5, 0.3]
        metadata_scores = [0.9, 0.7, 0.5]
        
        combined = ranker.combine_scores(rl_scores, similarity_scores, metadata_scores)
        
        assert len(combined) == len(rl_scores)
        assert all(0.0 <= score <= 1.0 for score in combined)
    
    def test_rank_models(self, sample_model_metadata_list, sample_user_context):
        """Test model ranking."""
        ranker = MockRanker()
        
        model_indices = [0, 1]
        rl_scores = [0.9, 0.7]
        similarity_scores = [0.8, 0.6]
        
        ranked = ranker.rank_models(
            model_indices=model_indices,
            rl_scores=rl_scores,
            similarity_scores=similarity_scores,
            model_metadata_list=sample_model_metadata_list,
            user_context=sample_user_context,
            top_k=2
        )
        
        assert len(ranked) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)
        # Check that scores are sorted descending
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_models_with_top_k(self, sample_model_metadata_list):
        """Test model ranking with top_k limit."""
        ranker = MockRanker()
        
        model_indices = list(range(5))
        rl_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        similarity_scores = [0.8, 0.7, 0.6, 0.5, 0.4]
        metadata_list = sample_model_metadata_list * 3  # Extend for 5 models
        
        ranked = ranker.rank_models(
            model_indices=model_indices,
            rl_scores=rl_scores,
            similarity_scores=similarity_scores,
            model_metadata_list=metadata_list,
            top_k=3
        )
        
        assert len(ranked) == 3


class TestSimilaritySearch:
    """Tests for the SimilaritySearch class."""
    
    def test_mock_similarity_search_init(self):
        """Test mock similarity search initialization."""
        search = MockSimilaritySearch(num_models=10, seed=42)
        assert search.num_models == 10
        assert search.embeddings is not None
        assert search.embeddings.shape[0] == 10
    
    def test_search(self):
        """Test similarity search."""
        search = MockSimilaritySearch(num_models=5, seed=42)
        query_embedding = np.random.rand(128).astype(np.float32)
        
        results = search.search(query_embedding, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        # Check that scores are sorted descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_threshold(self):
        """Test similarity search with threshold."""
        search = MockSimilaritySearch(num_models=5, seed=42)
        query_embedding = np.random.rand(128).astype(np.float32)
        
        results = search.search(query_embedding, threshold=0.5)
        
        # All results should have score >= threshold
        assert all(score >= 0.5 for _, score in results)
    
    def test_get_model_embedding(self):
        """Test getting model embedding."""
        search = MockSimilaritySearch(num_models=5, seed=42)
        
        embedding = search.get_model_embedding(0)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 128
    
    def test_batch_search(self):
        """Test batch similarity search."""
        search = MockSimilaritySearch(num_models=5, seed=42)
        query_embeddings = [
            np.random.rand(128).astype(np.float32) for _ in range(3)
        ]
        
        results = search.batch_search(query_embeddings, top_k=2)
        
        assert len(results) == 3
        assert all(len(result) == 2 for result in results)


class TestRecipeGenerator:
    """Tests for the RecipeGenerator class."""
    
    def test_recipe_generator_init(self):
        """Test recipe generator initialization."""
        generator = RecipeGenerator()
        assert generator is not None
    
    def test_generate_recipe(self, sample_model_metadata, sample_user_context):
        """Test recipe generation."""
        generator = MockRecipeGenerator()
        
        recipe = generator.generate_recipe(sample_model_metadata, sample_user_context)
        
        assert recipe.model_id == sample_model_metadata["model_id"]
        assert recipe.model_name == sample_model_metadata["model_name"]
        assert "configuration" in recipe.__dict__
        assert "usage_instructions" in recipe.__dict__
        assert isinstance(recipe.configuration, dict)
        assert isinstance(recipe.usage_instructions, str)
        assert len(recipe.usage_instructions) > 0
    
    def test_generate_recipe_no_context(self, sample_model_metadata):
        """Test recipe generation without user context."""
        generator = MockRecipeGenerator()
        
        recipe = generator.generate_recipe(sample_model_metadata, None)
        
        assert recipe.model_id == sample_model_metadata["model_id"]
        assert recipe.configuration is not None
    
    def test_generate_batch_recipes(self, sample_model_metadata_list):
        """Test batch recipe generation."""
        generator = MockRecipeGenerator()
        
        recipes = generator.generate_batch_recipes(sample_model_metadata_list)
        
        assert len(recipes) == len(sample_model_metadata_list)
        assert all(recipe.model_id in [m["model_id"] for m in sample_model_metadata_list] 
                  for recipe in recipes)
    
    def test_recipe_has_example_code(self, sample_model_metadata):
        """Test that recipe includes example code."""
        generator = RecipeGenerator()
        
        recipe = generator.generate_recipe(sample_model_metadata)
        
        assert recipe.example_code is not None
        assert isinstance(recipe.example_code, str)
        assert len(recipe.example_code) > 0
    
    def test_recipe_has_requirements(self, sample_model_metadata):
        """Test that recipe includes requirements."""
        generator = RecipeGenerator()
        
        recipe = generator.generate_recipe(sample_model_metadata)
        
        assert recipe.requirements is not None
        assert isinstance(recipe.requirements, list)
        assert len(recipe.requirements) > 0
        assert "transformers" in recipe.requirements

