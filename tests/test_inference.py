"""
Tests for inference modules (preprocessing, RL policy, postprocessing, artifact loading).
"""
import pytest
import numpy as np
from typing import Dict, Any

from src.inference.preprocessing import Preprocessor, MockPreprocessor
from src.inference.rl_policy import RLPolicy, MockRLPolicy
from src.inference.postprocessing import Postprocessor
from src.inference.load_artifacts import ArtifactLoader
from tests.conftest import sample_model_metadata, sample_user_context


class TestPreprocessor:
    """Tests for the Preprocessor class."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = MockPreprocessor()
        assert preprocessor.config is not None
        assert "normalize" in preprocessor.config
    
    def test_encode_query(self):
        """Test query encoding."""
        preprocessor = MockPreprocessor()
        query = "I need a model for text generation"
        embedding = preprocessor.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) > 0
    
    def test_extract_user_context_features(self, sample_user_context):
        """Test user context feature extraction."""
        preprocessor = MockPreprocessor()
        features = preprocessor.extract_user_context_features(sample_user_context)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0
    
    def test_extract_user_context_features_none(self):
        """Test user context feature extraction with None."""
        preprocessor = MockPreprocessor()
        features = preprocessor.extract_user_context_features(None)
        
        assert isinstance(features, np.ndarray)
        assert len(features) >= 0
    
    def test_construct_state_vector(self):
        """Test state vector construction."""
        preprocessor = MockPreprocessor()
        query_features = np.random.rand(128).astype(np.float32)
        context_features = np.random.rand(32).astype(np.float32)
        
        state = preprocessor.construct_state_vector(query_features, context_features)
        
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) == len(query_features) + len(context_features)
    
    def test_preprocess_request(self, sample_user_context):
        """Test full preprocessing pipeline."""
        preprocessor = MockPreprocessor()
        query = "I need a model for creative writing"
        
        state = preprocessor.preprocess_request(query, sample_user_context)
        
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) > 0


class TestRLPolicy:
    """Tests for the RLPolicy class."""
    
    def test_mock_rl_policy_init(self):
        """Test mock RL policy initialization."""
        policy = MockRLPolicy(seed=42)
        assert policy.is_loaded
        assert policy.model == "mock_model"
    
    def test_mock_rl_policy_select_action(self):
        """Test mock RL policy action selection."""
        policy = MockRLPolicy(seed=42)
        state = np.random.rand(128).astype(np.float32)
        
        action = policy.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert action >= 0
    
    def test_mock_rl_policy_predict(self):
        """Test mock RL policy prediction."""
        policy = MockRLPolicy(seed=42)
        state = np.random.rand(128).astype(np.float32)
        
        scores = policy.predict(state)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0
        # Check that scores sum to approximately 1 (probabilities)
        assert abs(scores.sum() - 1.0) < 0.01
    
    def test_mock_rl_policy_get_recommendation_scores(self):
        """Test mock RL policy recommendation scores."""
        policy = MockRLPolicy(seed=42)
        states = [
            np.random.rand(128).astype(np.float32) for _ in range(5)
        ]
        
        scores = policy.get_recommendation_scores(states, top_k=3)
        
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in scores)
        # Check that scores are sorted descending
        score_values = [score for _, score in scores]
        assert score_values == sorted(score_values, reverse=True)


class TestPostprocessor:
    """Tests for the Postprocessor class."""
    
    def test_postprocessor_init(self):
        """Test postprocessor initialization."""
        postprocessor = Postprocessor()
        assert postprocessor is not None
    
    def test_format_recommendations(self, sample_model_metadata):
        """Test recommendation formatting."""
        postprocessor = Postprocessor()
        
        model_scores = [(0, 0.9), (1, 0.7)]
        model_catalog = {
            "0": sample_model_metadata,
            "1": {**sample_model_metadata, "model_id": "model2", "model_name": "Model 2"}
        }
        query_text = "I need a text generation model"
        
        recommendations = postprocessor.format_recommendations(
            model_scores, model_catalog, query_text, top_k=2
        )
        
        assert len(recommendations) == 2
        assert recommendations[0].rank == 1
        assert recommendations[1].rank == 2
        assert recommendations[0].score == 0.9
        assert recommendations[1].score == 0.7
    
    def test_generate_explanation(self, sample_model_metadata):
        """Test explanation generation."""
        postprocessor = Postprocessor()
        
        explanation = postprocessor._generate_explanation(
            model_data=sample_model_metadata,
            score=0.85,
            query_text="I need a creative writing model"
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "0.85" in explanation or "recommended" in explanation.lower()
    
    def test_create_model_metadata(self, sample_model_metadata):
        """Test model metadata creation."""
        postprocessor = Postprocessor()
        
        metadata = postprocessor._create_model_metadata(sample_model_metadata)
        
        assert metadata.model_id == sample_model_metadata["model_id"]
        assert metadata.model_name == sample_model_metadata["model_name"]
        assert metadata.huggingface_url == sample_model_metadata["huggingface_url"]
    
    def test_normalize_scores(self):
        """Test score normalization."""
        postprocessor = Postprocessor()
        
        scores = [0.1, 0.5, 0.9]
        normalized = postprocessor.normalize_scores(scores)
        
        assert len(normalized) == len(scores)
        assert min(normalized) >= 0.0
        assert max(normalized) <= 1.0
        assert normalized[0] == 0.0  # Min should be 0
        assert normalized[2] == 1.0  # Max should be 1
    
    def test_rank_by_score(self):
        """Test ranking by score."""
        postprocessor = Postprocessor()
        
        model_scores = [(0, 0.3), (1, 0.9), (2, 0.5)]
        ranked = postprocessor.rank_by_score(model_scores, descending=True)
        
        assert ranked[0][1] == 0.9  # Highest score first
        assert ranked[1][1] == 0.5
        assert ranked[2][1] == 0.3


class TestArtifactLoader:
    """Tests for the ArtifactLoader class."""
    
    def test_artifact_loader_init(self):
        """Test artifact loader initialization."""
        loader = ArtifactLoader()
        assert loader.sac_model is None
        assert loader.embeddings is None
        assert loader.preprocessing_config is None
        assert loader.model_catalog is None
        assert not loader.artifacts_loaded
    
    def test_load_all_mock(self):
        """Test loading all artifacts (will use mocks since artifacts don't exist)."""
        loader = ArtifactLoader()
        result = loader.load_all()
        
        # Should return False since artifacts don't exist, but should not raise error
        assert isinstance(result, bool)
    
    def test_load_sac_model_mock(self):
        """Test loading SAC model (will use mock)."""
        loader = ArtifactLoader()
        result = loader.load_sac_model()
        
        # Should return False since model doesn't exist, but should set mock
        assert isinstance(result, bool)
        # Mock should be set
        assert loader.sac_model is not None
    
    def test_load_embeddings_mock(self):
        """Test loading embeddings (will use mock)."""
        loader = ArtifactLoader()
        result = loader.load_embeddings()
        
        assert isinstance(result, bool)
        assert loader.embeddings is not None
    
    def test_load_preprocessing_config_mock(self):
        """Test loading preprocessing config (will use default)."""
        loader = ArtifactLoader()
        result = loader.load_preprocessing_config()
        
        assert isinstance(result, bool)
        assert loader.preprocessing_config is not None
        assert isinstance(loader.preprocessing_config, dict)

