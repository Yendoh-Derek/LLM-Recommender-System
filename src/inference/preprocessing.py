"""
Preprocessing module for feature transformation.
Real implementation using feature scaler and actual feature columns.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.core.logger import logger


class Preprocessor:
    """
    Feature preprocessing for RL agent input.
    
    Creates state vectors from model features for the SAC agent.
    """
    
    def __init__(self, feature_scaler=None, feature_columns=None):
        """
        Initialize the preprocessor.
        
        Args:
            feature_scaler: Scikit-learn scaler for feature normalization
            feature_columns: List of feature column names to use
        """
        self.feature_scaler = feature_scaler
        self.feature_columns = feature_columns or [
            'cost_score',
            'popularity_score',
            'recency_score',
            'diversity_score',
            'performance_score'
        ]
        logger.info(f"Initialized Preprocessor with {len(self.feature_columns)} feature columns")
    
    def create_state_from_features(
        self,
        features_df: pd.DataFrame,
        step_progress: float = 0.0
    ) -> np.ndarray:
        """
        Create state vector from features DataFrame.
        
        Args:
            features_df: DataFrame with model features
            step_progress: Step progress value (0.0 for inference)
            
        Returns:
            State vector for SAC agent
        """
        # Get mean of feature columns across all models
        available_cols = [col for col in self.feature_columns if col in features_df.columns]
        if not available_cols:
            logger.warning(f"None of the feature columns found. Using default values.")
            state = np.array([0.5] * len(self.feature_columns), dtype=np.float32)
        else:
            state = features_df[available_cols].fillna(0.5).mean().values
        
        # Ensure we have the right number of features
        if len(state) < len(self.feature_columns):
            # Pad with default values
            padding = np.array([0.5] * (len(self.feature_columns) - len(state)), dtype=np.float32)
            state = np.concatenate([state, padding])
        elif len(state) > len(self.feature_columns):
            # Truncate if needed
            state = state[:len(self.feature_columns)]
        
        # Add step progress
        state = np.append(state, step_progress)
        
        # Apply scaling if available
        if self.feature_scaler is not None:
            try:
                state = self.feature_scaler.transform(state.reshape(1, -1))[0]
            except Exception as e:
                logger.warning(f"Error applying feature scaler: {e}. Using unscaled features.")
        
        return state.astype(np.float32)
    
    def preprocess_request(
        self,
        features_df: pd.DataFrame,
        step_progress: float = 0.0
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for a recommendation request.
        
        Args:
            features_df: DataFrame with model features
            step_progress: Step progress value (0.0 for inference)
            
        Returns:
            Preprocessed state vector ready for RL agent
        """
        logger.debug("Preprocessing request...")
        
        state = self.create_state_from_features(features_df, step_progress)
        
        logger.debug(f"Preprocessed state vector shape: {state.shape}")
        return state
