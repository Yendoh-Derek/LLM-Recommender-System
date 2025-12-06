"""
Leaderboard client for fetching and parsing LLM benchmark scores.
Supports Open LLM Leaderboard and other public benchmarks.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import requests
from tqdm.auto import tqdm


@dataclass
class BenchmarkScore:
    """Individual benchmark score."""
    model_id: str
    benchmark_name: str
    score: float
    metadata: Optional[Dict] = None


@dataclass
class ModelScores:
    """Complete scores for a model across all benchmarks."""
    model_id: str
    average_score: Optional[float]
    # NEW benchmarks (2024+ leaderboard)
    ifeval: Optional[float]
    bbh: Optional[float]
    math: Optional[float]
    gpqa: Optional[float]
    musr: Optional[float]
    mmlu_pro: Optional[float]
    # Legacy benchmarks (kept for compatibility)
    mmlu: Optional[float] = None
    gsm8k: Optional[float] = None
    truthfulqa: Optional[float] = None
    hellaswag: Optional[float] = None
    arc: Optional[float] = None
    winogrande: Optional[float] = None
    benchmark_count: int = 0
    raw_data: Optional[Dict] = None


class LeaderboardClient:
    """
    Client for fetching LLM benchmark scores from Open LLM Leaderboard.
    """
    
    # Open LLM Leaderboard dataset on HF
    LEADERBOARD_DATASET = "open-llm-leaderboard/contents"
    LEADERBOARD_API = "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    
    def __init__(
        self,
        cache_file: Path,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize leaderboard client.
        
        Args:
            cache_file: Path to cache file for leaderboard data
            rate_limit_delay: Seconds to wait between API calls
        """
        self.cache_file = cache_file
        self.rate_limit_delay = rate_limit_delay
        self._cache = {}
        
        # Load cache if exists
        if cache_file.exists():
            self._load_cache()
    
    def _load_cache(self):
        """Load cached leaderboard data."""
        try:
            with open(self.cache_file, 'r') as f:
                self._cache = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load cache: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save leaderboard data to cache."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    def fetch_leaderboard_data(self) -> pd.DataFrame:
        """
        Fetch full leaderboard data from HuggingFace.
        
        Returns:
            DataFrame with leaderboard scores
        """
        print("Fetching Open LLM Leaderboard data...")
        
        try:
            # Try to fetch from HF Datasets
            from datasets import load_dataset
            
            print("Loading leaderboard dataset from HuggingFace...")
            dataset = load_dataset(
                "open-llm-leaderboard/contents",
                split="train"
            )
            
            # Convert to pandas
            df = dataset.to_pandas()
            
            print(f"✅ Loaded {len(df)} leaderboard entries")
            
            # Debug: Print available columns
            print(f"\nAvailable columns: {list(df.columns)[:10]}...")  # Show first 10
            
            return df
            
        except Exception as e:
            print(f"⚠️  Could not fetch leaderboard: {e}")
            print("Returning empty DataFrame")
            return pd.DataFrame()
    
    def parse_model_scores(
        self,
        leaderboard_df: pd.DataFrame,
        model_id: str
    ) -> ModelScores:
        """
        Parse scores for a specific model from leaderboard data.
        
        Args:
            leaderboard_df: Full leaderboard DataFrame
            model_id: Model identifier to look up
            
        Returns:
            ModelScores object
        """
        # Handle empty dataframe
        if len(leaderboard_df) == 0:
            return self._empty_scores(model_id)
        
        # Find the model column (could be 'model', 'Model', 'model_name', etc.)
        model_col = self._find_model_column(leaderboard_df)
        
        if model_col is None:
            print(f"⚠️  Could not find model column in leaderboard")
            return self._empty_scores(model_id)
        
        # Try exact match first
        model_data = leaderboard_df[leaderboard_df[model_col] == model_id]
        
        # If no exact match, try partial match (handle different naming)
        if len(model_data) == 0:
            # Try matching on base name (e.g., "gpt2" matches "openai-community/gpt2")
            base_name = model_id.split('/')[-1]
            model_data = leaderboard_df[
                leaderboard_df[model_col].str.contains(base_name, case=False, na=False)
            ]
        
        if len(model_data) == 0:
            # No scores found
            return self._empty_scores(model_id)
        
        # Take first match if multiple found
        row = model_data.iloc[0]
        
        # Extract scores (column names may vary)
        scores = self._extract_scores_from_row(row)
        
        # Calculate average from available scores
        available_scores = [s for s in scores.values() if s is not None]
        avg_score = sum(available_scores) / len(available_scores) if available_scores else None
        
        return ModelScores(
            model_id=model_id,
            average_score=avg_score,
            ifeval=scores.get('ifeval'),
            bbh=scores.get('bbh'),
            math=scores.get('math'),
            gpqa=scores.get('gpqa'),
            musr=scores.get('musr'),
            mmlu_pro=scores.get('mmlu_pro'),
            # Legacy (likely None for new leaderboard)
            mmlu=scores.get('mmlu'),
            gsm8k=scores.get('gsm8k'),
            truthfulqa=scores.get('truthfulqa'),
            hellaswag=scores.get('hellaswag'),
            arc=scores.get('arc'),
            winogrande=scores.get('winogrande'),
            benchmark_count=len(available_scores),
            raw_data=row.to_dict()
        )
    
    def _find_model_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column that contains model identifiers.
        
        Args:
            df: Leaderboard DataFrame
            
        Returns:
            Column name or None
        """
        # Prioritize 'fullname' for new leaderboard
        if 'fullname' in df.columns:
            return 'fullname'
        
        possible_names = [
            'model', 'Model', 'model_name', 'Model Name', 'model_id', 
            'Model ID', 'name', 'Name', 'model_name_for_query',
            'Base Model'
        ]
        
        for name in possible_names:
            if name in df.columns:
                # Check if it contains HTML (like the 'Model' column)
                sample = df[name].iloc[0] if len(df) > 0 else ""
                if isinstance(sample, str) and '<' not in sample:
                    return name
        
        # If not found, check for columns containing 'model'
        for col in df.columns:
            if 'model' in col.lower():
                sample = df[col].iloc[0] if len(df) > 0 else ""
                if isinstance(sample, str) and '<' not in sample:
                    return col
        
        return None
    
    def _empty_scores(self, model_id: str) -> ModelScores:
        """Return ModelScores with all None values."""
        return ModelScores(
            model_id=model_id,
            average_score=None,
            ifeval=None,
            bbh=None,
            math=None,
            gpqa=None,
            musr=None,
            mmlu_pro=None,
            mmlu=None,
            gsm8k=None,
            truthfulqa=None,
            hellaswag=None,
            arc=None,
            winogrande=None,
            benchmark_count=0,
            raw_data=None
        )
    
    def _extract_scores_from_row(self, row: pd.Series) -> Dict[str, Optional[float]]:
        """
        Extract benchmark scores from a leaderboard row.
        Handles NEW Open LLM Leaderboard benchmarks (2024+).
        
        Args:
            row: DataFrame row with leaderboard data
            
        Returns:
            Dictionary of benchmark_name -> score
        """
        scores = {}
        
        # NEW leaderboard benchmark mappings (as of late 2024)
        benchmark_mappings = {
            'ifeval': ['IFEval', 'ifeval'],
            'bbh': ['BBH', 'bbh'],
            'math': ['MATH Lvl 5', 'MATH_Lvl_5', 'math'],
            'gpqa': ['GPQA', 'gpqa'],
            'musr': ['MUSR', 'musr'],
            'mmlu_pro': ['MMLU-PRO', 'MMLU_PRO', 'mmlu-pro', 'mmlu_pro'],
            
            # Legacy benchmarks (may not exist in new leaderboard)
            'mmlu': ['mmlu', 'MMLU', 'Average'],
            'gsm8k': ['gsm8k', 'GSM8K', 'GSM-8K'],
            'truthfulqa': ['truthfulqa', 'TruthfulQA', 'truthful_qa'],
            'hellaswag': ['hellaswag', 'HellaSwag', 'hella_swag'],
            'arc': ['arc', 'ARC', 'arc_challenge', 'ARC-Challenge'],
            'winogrande': ['winogrande', 'Winogrande', 'WinoGrande']
        }
        
        for benchmark, possible_cols in benchmark_mappings.items():
            score = None
            for col in possible_cols:
                if col in row.index:
                    try:
                        val = row[col]
                        if val is not None and not pd.isna(val):
                            score = float(val)
                            break
                    except (ValueError, TypeError):
                        continue
            scores[benchmark] = score
        
        return scores
    
    def fetch_batch_scores(
        self,
        model_ids: List[str],
        use_cache: bool = True
    ) -> Dict[str, ModelScores]:
        """
        Fetch scores for multiple models.
        
        Args:
            model_ids: List of model IDs
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping model_id -> ModelScores
        """
        # Check cache first
        if use_cache and self._cache:
            print("Using cached leaderboard data...")
            results = {}
            for model_id in model_ids:
                if model_id in self._cache:
                    cached = self._cache[model_id]
                    results[model_id] = ModelScores(**cached)
                else:
                    results[model_id] = ModelScores(
                        model_id=model_id,
                        average_score=None,
                        mmlu=None,
                        gsm8k=None,
                        truthfulqa=None,
                        hellaswag=None,
                        arc=None,
                        winogrande=None,
                        benchmark_count=0
                    )
            return results
        
        # Fetch fresh data
        leaderboard_df = self.fetch_leaderboard_data()
        
        if len(leaderboard_df) == 0:
            print("⚠️  No leaderboard data available, returning empty scores")
            return {
                model_id: ModelScores(
                    model_id=model_id,
                    average_score=None,
                    mmlu=None,
                    gsm8k=None,
                    truthfulqa=None,
                    hellaswag=None,
                    arc=None,
                    winogrande=None,
                    benchmark_count=0
                )
                for model_id in model_ids
            }
        
        # Parse scores for each model
        results = {}
        for model_id in tqdm(model_ids, desc="Parsing leaderboard scores"):
            scores = self.parse_model_scores(leaderboard_df, model_id)
            results[model_id] = scores
            
            # Cache the result (NEW benchmarks)
            self._cache[model_id] = {
                'model_id': scores.model_id,
                'average_score': scores.average_score,
                'ifeval': scores.ifeval,
                'bbh': scores.bbh,
                'math': scores.math,
                'gpqa': scores.gpqa,
                'musr': scores.musr,
                'mmlu_pro': scores.mmlu_pro,
                'mmlu': scores.mmlu,
                'gsm8k': scores.gsm8k,
                'truthfulqa': scores.truthfulqa,
                'hellaswag': scores.hellaswag,
                'arc': scores.arc,
                'winogrande': scores.winogrande,
                'benchmark_count': scores.benchmark_count
            }
        
        # Save cache
        self._save_cache()
        
        return results


class PerformanceScorer:
    """
    Calculate composite performance scores from benchmark results.
    """
    
    # Benchmark weights (updated for NEW leaderboard)
    DEFAULT_WEIGHTS = {
        # NEW benchmarks (2024+ leaderboard)
        'ifeval': 0.20,     # Instruction following
        'bbh': 0.20,        # Big-Bench Hard (reasoning)
        'math': 0.20,       # Math Level 5
        'gpqa': 0.15,       # Graduate-level Q&A
        'musr': 0.15,       # Multi-step reasoning
        'mmlu_pro': 0.10,   # MMLU-PRO
        
        # Legacy benchmarks (kept for compatibility, but likely unused)
        'mmlu': 0.0,
        'gsm8k': 0.0,
        'truthfulqa': 0.0,
        'hellaswag': 0.0,
        'arc': 0.0,
        'winogrande': 0.0
    }
    
    @classmethod
    def calculate_composite_score(
        cls,
        scores: ModelScores,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted composite performance score.
        
        Args:
            scores: ModelScores object
            weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)
            
        Returns:
            Composite score (0-100)
        """
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS
        
        # Collect available scores (NEW benchmarks)
        available = {}
        if scores.ifeval is not None:
            available['ifeval'] = scores.ifeval
        if scores.bbh is not None:
            available['bbh'] = scores.bbh
        if scores.math is not None:
            available['math'] = scores.math
        if scores.gpqa is not None:
            available['gpqa'] = scores.gpqa
        if scores.musr is not None:
            available['musr'] = scores.musr
        if scores.mmlu_pro is not None:
            available['mmlu_pro'] = scores.mmlu_pro
        
        # Legacy benchmarks (if available)
        if scores.mmlu is not None:
            available['mmlu'] = scores.mmlu
        if scores.gsm8k is not None:
            available['gsm8k'] = scores.gsm8k
        if scores.truthfulqa is not None:
            available['truthfulqa'] = scores.truthfulqa
        if scores.hellaswag is not None:
            available['hellaswag'] = scores.hellaswag
        if scores.arc is not None:
            available['arc'] = scores.arc
        if scores.winogrande is not None:
            available['winogrande'] = scores.winogrande
        
        if not available:
            return None
        
        # Calculate weighted average using only available scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        for benchmark, score in available.items():
            weight = weights.get(benchmark, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        return weighted_sum / total_weight
    
    @staticmethod
    def normalize_performance_scores(scores: pd.Series) -> pd.Series:
        """
        Normalize performance scores to [0, 1] range.
        
        Args:
            scores: Series of raw performance scores
            
        Returns:
            Normalized scores
        """
        valid_scores = scores.dropna()
        
        if len(valid_scores) == 0:
            return pd.Series([None] * len(scores), index=scores.index)
        
        min_score = valid_scores.min()
        max_score = valid_scores.max()
        
        if max_score == min_score:
            return pd.Series([0.5] * len(scores), index=scores.index)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized