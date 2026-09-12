"""
double_ml_engine.py

Implements the Double Machine Learning (DML) engine for causal source apportionment.
This engine is responsible for "residualizing" out the effects of weather (confounders)
so we can isolate the true emissions impact of our proxy ratios (treatments) on PM2.5 (outcome).

Architecture:
    - Stage 1: Cross-fitted residualization using LightGBM.
        - Regress PM2.5 on weather (Y ~ W) to get Y_res.
        - Regress each proxy on weather (T_k ~ W) to get T_res_k.
    - Stage 2: Orthogonal Linear Regression.
        - Regress Y_res on T_res (with an intercept) to obtain the unconfounded weights (beta_k).

The engine returns a BetaWeights object (from src.models.counterfactual) containing
the fitted causal coefficients, which can then be fed into the PolicySimulator.

Usage
-----
    import pandas as pd
    from src.models.double_ml_engine import CausalSourceApportionmentEngine

    df = pd.read_csv("training_data.csv")
    engine = CausalSourceApportionmentEngine()
    weights = engine.fit(df)
    
    # weights is a BetaWeights object, pass to PolicySimulator...
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from typing import List, Optional
import logging

from src.models.counterfactual import BetaWeights

logger = logging.getLogger(__name__)

class CausalSourceApportionmentEngine:
    """
    Double ML Engine for estimating causal effects of chemical proxy ratios on PM2.5,
    controlling for meteorological confounders.
    """
    def __init__(
        self, 
        n_splits: int = 5,
        treatment_cols: Optional[List[str]] = None,
        confounder_cols: Optional[List[str]] = None,
        target_col: str = "PM2.5",
        random_state: int = 42
    ):
        """
        Initialize the Double ML Engine.

        Args:
            n_splits: Number of CV folds for cross-fitting.
            treatment_cols: List of proxy feature columns (e.g., ['traffic_ratio', ...]).
            confounder_cols: List of weather/confounder columns (e.g., ['temp', 'wind_u', ...]).
            target_col: The outcome column (default PM2.5).
            random_state: Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.target_col = target_col
        self.random_state = random_state
        
        # Defaults based on our pipeline so far
        self.treatment_cols = treatment_cols or [
            'traffic_ratio', 'industry_ratio', 'dust_ratio', 'biomass_ratio'
        ]
        self.confounder_cols = confounder_cols or [
            'temp', 'humidity', 'wind_u', 'wind_v', 'pblh'
        ]

    def fit(self, df: pd.DataFrame) -> BetaWeights:
        """
        Run the Double ML estimation pipeline.

        Args:
            df: DataFrame containing treatments, confounders, and the target.

        Returns:
            BetaWeights: An object containing the fitted causal coefficients.
        """
        # Ensure all columns exist, drop NaNs (DML needs clean numeric inputs)
        required_cols = self.treatment_cols + self.confounder_cols + [self.target_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for DML: {missing}")
            
        clean_df = df[required_cols].dropna()
        if len(clean_df) < len(df):
            logger.warning(f"Dropped {len(df) - len(clean_df)} rows containing NaNs before DML fit.")

        W = clean_df[self.confounder_cols].values
        Y = clean_df[self.target_col].values
        T = clean_df[self.treatment_cols].values

        logger.info(f"Starting Stage 1: Cross-fitted residualization ({self.n_splits}-fold CV)")
        
        # Arrays to hold the out-of-fold residuals
        Y_res = np.zeros(len(Y))
        T_res = np.zeros_like(T)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Stage 1: Nuisance Models (Regress out weather W)
        for fold, (train_idx, test_idx) in enumerate(kf.split(clean_df), 1):
            # Residualize Outcome Y
            model_y = LGBMRegressor(
                n_estimators=100, 
                max_depth=5, 
                random_state=self.random_state,
                verbose=-1
            )
            model_y.fit(W[train_idx], Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - model_y.predict(W[test_idx])

            # Residualize Treatments T
            for k in range(T.shape[1]):
                model_t = LGBMRegressor(
                    n_estimators=100, 
                    max_depth=5, 
                    random_state=self.random_state,
                    verbose=-1
                )
                model_t.fit(W[train_idx], T[train_idx, k])
                T_res[test_idx, k] = T[test_idx, k] - model_t.predict(W[test_idx])

        logger.info("Starting Stage 2: Orthogonal Estimation")
        
        # Stage 2: Orthogonal Linear Regression (Y_res ~ T_res)
        # We use fit_intercept=True because the theoretical DML allows an intercept 
        # to absorb global mean shifts, although theoretically it should be near 0.
        # We use positive=True to constrain causal effects to be physically non-negative.
        second_stage = LinearRegression(fit_intercept=True, positive=True)
        second_stage.fit(T_res, Y_res)

        # Build coefficients dictionary
        coefs = {col: coef for col, coef in zip(self.treatment_cols, second_stage.coef_)}
        intercept = float(second_stage.intercept_)

        logger.info(f"DML Fit Complete. Intercept: {intercept:.2f}")
        for k, v in coefs.items():
            logger.info(f"  {k}: {v:.4f}")

        # Return the BetaWeights container designed by the simulator module
        return BetaWeights(
            intercept=intercept,
            coefficients=coefs,
            target=self.target_col
        )

# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    n = 1000
    
    # Synthetic Data
    df = pd.DataFrame({
        'temp': rng.uniform(10, 35, n),
        'humidity': rng.uniform(30, 90, n),
        'wind_u': rng.normal(0, 5, n),
        'wind_v': rng.normal(0, 5, n),
        'pblh': rng.uniform(500, 2000, n),
        
        'traffic_ratio': rng.uniform(0.1, 2.0, n),
        'industry_ratio': rng.uniform(0.01, 0.5, n),
        'dust_ratio': rng.uniform(0.1, 0.8, n),
        'biomass_ratio': rng.uniform(0.05, 0.3, n),
    })
    
    # Generate PM2.5 with known causal effects plus some weather confounding
    df['PM2.5'] = (
        10.0 + # intercept
        50.0 * df['traffic_ratio'] + 
        30.0 * df['industry_ratio'] + 
        20.0 * df['dust_ratio'] + 
        15.0 * df['biomass_ratio'] + 
        2.0 * df['temp'] - 1.5 * df['wind_u'] # confounding
        + rng.normal(0, 2, n) # noise
    )

    engine = CausalSourceApportionmentEngine()
    weights = engine.fit(df)
    
    print("\nRecovered Weights:")
    print(f"Intercept: {weights.intercept:.2f}")
    for k, v in weights.coefficients.items():
        print(f"{k}: {v:.2f}")
