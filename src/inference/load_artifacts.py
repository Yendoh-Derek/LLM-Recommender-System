"""
Artifact loading module for trained models, embeddings, and preprocessing configs.
Real implementation loading Colab artifacts.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import json
import pickle

import numpy as np
import pandas as pd
import faiss
import torch

from src.core.config import settings
from src.core.logger import logger
from src.inference.rl_policy import SACAgent, RLPolicy


class ArtifactLoader:
    """
    Class for loading trained models and preprocessing artifacts.
    Loads real artifacts from Colab training.
    """
    
    def __init__(self):
        """Initialize the artifact loader."""
        self.sac_agent = None
        self.rl_policy = None
        self.features_df = None
        self.embeddings = None
        self.faiss_index = None
        self.model_id_mapping = None
        self.feature_scaler = None
        self.sac_config = None
        self.deployment_config = None
        self.artifacts_loaded = False
    
    def load_sac_model(self, model_path: Optional[Path] = None, config_path: Optional[Path] = None) -> bool:
        """
        Load the trained SAC agent model.
        
        Args:
            model_path: Path to the SAC model file. If None, uses config path.
            config_path: Path to SAC config file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if model_path is None:
            model_path = settings.MODELS_DIR / "sac_best.pt"
        if config_path is None:
            config_path = settings.ARTIFACTS_DIR / "config" / "sac_config.json"
        
        if not model_path.exists():
            logger.error(f"SAC model not found at {model_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"SAC config not found at {config_path}")
            return False
        
        try:
            # Load config to get architecture parameters
            with open(config_path, 'r') as f:
                self.sac_config = json.load(f)
            
            state_dim = self.sac_config['architecture']['state_dim']
            action_dim = self.sac_config['architecture']['action_dim']
            
            # Create and load SAC agent
            self.sac_agent = SACAgent(state_dim, action_dim, device='cpu')
            self.sac_agent.load(str(model_path))
            
            # Create RL policy wrapper
            self.rl_policy = RLPolicy(self.sac_agent)
            
            logger.info(f"Loaded SAC agent: state_dim={state_dim}, action_dim={action_dim}")
            return True
        except Exception as e:
            logger.error(f"Error loading SAC model: {e}", exc_info=True)
            return False
    
    def load_embeddings(self, embeddings_path: Optional[Path] = None) -> bool:
        """
        Load model embeddings.
        
        Args:
            embeddings_path: Path to embeddings file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if embeddings_path is None:
            embeddings_path = settings.EMBEDDINGS_DIR / "model_embeddings.npy"
        
        if not embeddings_path.exists():
            logger.error(f"Embeddings not found at {embeddings_path}")
            return False
        
        try:
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings: shape {self.embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}", exc_info=True)
            return False
    
    def load_faiss_index(self, faiss_path: Optional[Path] = None) -> bool:
        """
        Load FAISS index for similarity search.
        
        Args:
            faiss_path: Path to FAISS index file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if faiss_path is None:
            faiss_path = settings.EMBEDDINGS_DIR / "model_embeddings.faiss"
        
        if not faiss_path.exists():
            logger.warning(f"FAISS index not found at {faiss_path}. Similarity search will not be available.")
            return False
        
        try:
            self.faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}", exc_info=True)
            return False
    
    def load_model_id_mapping(self, mapping_path: Optional[Path] = None) -> bool:
        """
        Load model ID to index mapping.
        
        Args:
            mapping_path: Path to mapping file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if mapping_path is None:
            mapping_path = settings.EMBEDDINGS_DIR / "model_id_mapping.json"
        
        if not mapping_path.exists():
            logger.error(f"Model ID mapping not found at {mapping_path}")
            return False
        
        try:
            with open(mapping_path, 'r') as f:
                self.model_id_mapping = json.load(f)
            logger.info(f"Loaded model ID mapping: {len(self.model_id_mapping)} models")
            return True
        except Exception as e:
            logger.error(f"Error loading model ID mapping: {e}", exc_info=True)
            return False
    
    def load_features(self, features_path: Optional[Path] = None) -> bool:
        """
        Load model features DataFrame.
        
        Args:
            features_path: Path to features parquet file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if features_path is None:
            features_path = settings.ARTIFACTS_DIR / "features" / "features_with_scores.parquet"
        
        if not features_path.exists():
            logger.error(f"Features file not found at {features_path}")
            return False
        
        try:
            self.features_df = pd.read_parquet(features_path)
            logger.info(f"Loaded features: {len(self.features_df)} models, {len(self.features_df.columns)} features")
            return True
        except Exception as e:
            logger.error(f"Error loading features: {e}", exc_info=True)
            return False
    
    def load_feature_scaler(self, scaler_path: Optional[Path] = None) -> bool:
        """
        Load feature scaler for preprocessing.
        
        Args:
            scaler_path: Path to scaler pickle file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if scaler_path is None:
            scaler_path = settings.PREPROCESSING_DIR / "feature_scaler.pkl"
        
        if not scaler_path.exists():
            logger.warning(f"Feature scaler not found at {scaler_path}. Will proceed without scaling.")
            return False
        
        try:
            with open(scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
            logger.info("Loaded feature scaler")
            return True
        except Exception as e:
            logger.warning(f"Error loading feature scaler: {e}. Will proceed without scaling.")
            return False
    
    def load_deployment_config(self, config_path: Optional[Path] = None) -> bool:
        """
        Load deployment configuration.
        
        Args:
            config_path: Path to deployment config file. If None, uses config path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if config_path is None:
            config_path = settings.ARTIFACTS_DIR / "config" / "deployment_config.json"
        
        if not config_path.exists():
            logger.warning(f"Deployment config not found at {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                self.deployment_config = json.load(f)
            logger.info("Loaded deployment config")
            return True
        except Exception as e:
            logger.warning(f"Error loading deployment config: {e}")
            return False
    
    def load_all(self) -> bool:
        """
        Load all artifacts.
        
        Returns:
            True if all critical artifacts loaded successfully, False otherwise.
        """
        logger.info("Loading all artifacts...")
        
        # Load critical artifacts
        sac_loaded = self.load_sac_model()
        features_loaded = self.load_features()
        embeddings_loaded = self.load_embeddings()
        mapping_loaded = self.load_model_id_mapping()
        
        # Load optional artifacts
        faiss_loaded = self.load_faiss_index()
        scaler_loaded = self.load_feature_scaler()
        config_loaded = self.load_deployment_config()
        
        # Critical artifacts must be loaded
        self.artifacts_loaded = all([sac_loaded, features_loaded, embeddings_loaded, mapping_loaded])
        
        if self.artifacts_loaded:
            logger.info("All critical artifacts loaded successfully")
            if faiss_loaded:
                logger.info("FAISS index loaded - similarity search available")
            if scaler_loaded:
                logger.info("Feature scaler loaded - normalization available")
        else:
            logger.error("Failed to load critical artifacts")
        
        return self.artifacts_loaded


# Global artifact loader instance
_artifact_loader: Optional[ArtifactLoader] = None


def load_all_artifacts() -> ArtifactLoader:
    """
    Load all artifacts and return the loader instance.
    This is a singleton pattern - artifacts are loaded once.
    
    Returns:
        ArtifactLoader instance with loaded artifacts
    """
    global _artifact_loader
    
    if _artifact_loader is None:
        _artifact_loader = ArtifactLoader()
        _artifact_loader.load_all()
    
    return _artifact_loader


def get_artifact_loader() -> Optional[ArtifactLoader]:
    """
    Get the current artifact loader instance without loading.
    
    Returns:
        ArtifactLoader instance if loaded, None otherwise
    """
    return _artifact_loader
