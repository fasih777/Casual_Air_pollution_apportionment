"""
counterfactual.py

A policy simulator that uses trained linear-model coefficients (beta
weights) over the chemical-signature ratio proxies -- see
`src/features/chemical_ratios.py` -- to answer "what-if" questions such
as:

    "If we cut traffic proxy emissions by 30%, what would the new
     PM2.5 level be?"

Model assumption
-----------------
The simulator assumes a (possibly log-linear) additive model of the form

    PM2.5 = intercept
            + beta_traffic  * traffic_ratio
            + beta_industry * industry_ratio
            + beta_dust     * dust_ratio
            + beta_biomass  * biomass_ratio
            + [other covariates, held fixed]

i.e. the coefficients come from a regression already fit elsewhere
(OLS/Ridge/etc. via statsmodels or sklearn) on the ratio features
produced by `chemical_ratios.compute_all_ratios`. This module does not
retrain that model -- it only *applies* trained weights to simulate
interventions.

A "what-if" intervention is expressed as a fractional change to one or
more input features (e.g. -0.30 for "cut by 30%", +0.15 for "increase by
15%"). The simulator is a ceteris-paribus tool: it holds every feature
not named in the intervention fixed at its observed value and asks what
the linear model would predict if the named feature(s) had been
different. It does not model second-order effects between proxies
(e.g. a traffic cut indirectly changing dust); if those interactions
matter for your use case, they should be encoded as additional,
explicit interventions.

Usage
-----
    import pandas as pd
    from src.models.counterfactual import BetaWeights, PolicySimulator

    weights = BetaWeights(
        intercept=8.2,
        coefficients={
            "traffic_ratio": 0.42,
            "industry_ratio": 0.35,
            "dust_ratio": 0.20,
            "biomass_ratio": 0.18,
        },
    )
    sim = PolicySimulator(weights)

    # df must already contain the ratio columns (see chemical_ratios.py)
    result = sim.what_if(df, {"traffic_ratio": -0.30})
    print(result[["PM2.5_baseline", "PM2.5_counterfactual", "pm25_delta"]])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight container
# ---------------------------------------------------------------------------


@dataclass
class BetaWeights:
    """
    Trained linear-model weights over the chemical-ratio proxy features.

    Attributes
    ----------
    intercept : float
        Model intercept term.
    coefficients : dict[str, float]
        Mapping of feature column name -> fitted coefficient, e.g.
        {"traffic_ratio": 0.42, "industry_ratio": 0.35, ...}.
    target : str
        Name of the predicted/target column (default "PM2.5").
    """

    intercept: float
    coefficients: dict[str, float] = field(default_factory=dict)
    target: str = "PM2.5"

    @property
    def feature_names(self) -> list[str]:
        return list(self.coefficients.keys())

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BetaWeights":
        """
        Load weights from a JSON file of the form:

            {
                "intercept": 8.2,
                "coefficients": {
                    "traffic_ratio": 0.42,
                    "industry_ratio": 0.35,
                    "dust_ratio": 0.20,
                    "biomass_ratio": 0.18
                },
                "target": "PM2.5"
            }
        """
        path = Path(path)
        with path.open("r") as f:
            payload = json.load(f)
        return cls(
            intercept=payload["intercept"],
            coefficients=payload.get("coefficients", {}),
            target=payload.get("target", "PM2.5"),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open("w") as f:
            json.dump(
                {
                    "intercept": self.intercept,
                    "coefficients": self.coefficients,
                    "target": self.target,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_statsmodels(cls, fitted_result, target: str = "PM2.5") -> "BetaWeights":
        """
        Convenience constructor from a fitted statsmodels OLS/GLM result.
        Assumes an 'Intercept' (or 'const') term is present in params.
        """
        params = fitted_result.params.to_dict()
        intercept = params.pop("Intercept", params.pop("const", 0.0))
        return cls(intercept=intercept, coefficients=params, target=target)

    @classmethod
    def from_sklearn(
        cls, fitted_estimator, feature_names: list[str], target: str = "PM2.5"
    ) -> "BetaWeights":
        """
        Convenience constructor from a fitted sklearn linear model
        (LinearRegression, Ridge, Lasso, ...), given the ordered list of
        feature names the estimator was trained on.
        """
        coefs = dict(zip(feature_names, np.ravel(fitted_estimator.coef_)))
        intercept = float(np.ravel(fitted_estimator.intercept_)[0])
        return cls(intercept=intercept, coefficients=coefs, target=target)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class PolicySimulator:
    """
    Applies trained BetaWeights to simulate "what-if" policy interventions
    on the chemical-ratio proxy features.
    """

    def __init__(self, weights: BetaWeights):
        self.weights = weights

    # -- core scoring -------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the target (e.g. PM2.5) from the model's feature columns
        present in `df`. Missing feature columns are treated as
        contributing zero (with a warning), so a partially-specified
        dataframe degrades gracefully rather than raising.
        """
        pred = pd.Series(self.weights.intercept, index=df.index, dtype=float)
        for feat, beta in self.weights.coefficients.items():
            if feat not in df.columns:
                logger.warning(
                    "predict: feature '%s' not found in dataframe; "
                    "treating its contribution as 0",
                    feat,
                )
                continue
            pred = pred + beta * df[feat]
        return pred.rename(f"{self.weights.target}_predicted")

    # -- counterfactual interventions ----------------------------------

    def apply_intervention(
        self,
        df: pd.DataFrame,
        interventions: Mapping[str, float],
        clip_negative: bool = True,
    ) -> pd.DataFrame:
        """
        Return a copy of `df` with the named feature columns scaled by
        (1 + pct_change). E.g. {"traffic_ratio": -0.30} multiplies
        traffic_ratio by 0.70 everywhere.

        Parameters
        ----------
        interventions : mapping of feature name -> fractional change
            (e.g. -0.30 for "cut by 30%", +0.15 for "raise by 15%").
        clip_negative : bool
            If True (default), floors resulting feature values at 0,
            since these ratios aren't physically negative.
        """
        unknown = [f for f in interventions if f not in self.weights.coefficients]
        if unknown:
            logger.warning(
                "apply_intervention: %s not in model coefficients; "
                "the model won't respond to changes in these",
                unknown,
            )

        missing = [f for f in interventions if f not in df.columns]
        if missing:
            raise KeyError(
                f"Cannot apply intervention: columns not found in dataframe: {missing}"
            )

        out = df.copy()
        for feat, pct_change in interventions.items():
            scaled = out[feat] * (1.0 + pct_change)
            if clip_negative:
                scaled = scaled.clip(lower=0)
            out[feat] = scaled
        return out

    def what_if(
        self,
        df: pd.DataFrame,
        interventions: Mapping[str, float],
        clip_negative: bool = True,
    ) -> pd.DataFrame:
        """
        Run a full what-if scenario: predict baseline PM2.5, apply the
        intervention(s), predict counterfactual PM2.5, and report deltas.

        Returns a dataframe with columns:
            - one column per intervened feature's baseline value
              ("<feat>_baseline") and post-intervention value
              ("<feat>_counterfactual")
            - "<target>_baseline"
            - "<target>_counterfactual"
            - "pm25_delta" (counterfactual - baseline; negative = improvement)
            - "pm25_pct_change" (delta / baseline, as a fraction)

        Example
        -------
            sim.what_if(df, {"traffic_ratio": -0.30})
        """
        target = self.weights.target

        baseline_pred = self.predict(df)
        counterfactual_df = self.apply_intervention(
            df, interventions, clip_negative=clip_negative
        )
        counterfactual_pred = self.predict(counterfactual_df)

        result = pd.DataFrame(index=df.index)
        for feat in interventions:
            result[f"{feat}_baseline"] = df[feat]
            result[f"{feat}_counterfactual"] = counterfactual_df[feat]

        result[f"{target}_baseline"] = baseline_pred
        result[f"{target}_counterfactual"] = counterfactual_pred
        result["pm25_delta"] = counterfactual_pred - baseline_pred
        result["pm25_pct_change"] = _safe_pct_change(counterfactual_pred, baseline_pred)

        return result

    def scenario_summary(
        self,
        df: pd.DataFrame,
        interventions: Mapping[str, float],
        clip_negative: bool = True,
    ) -> dict:
        """
        Aggregate a what_if scenario over the whole dataframe into a
        single human-readable summary, e.g. for a dashboard or report.
        """
        detail = self.what_if(df, interventions, clip_negative=clip_negative)
        target = self.weights.target

        summary = {
            "interventions": dict(interventions),
            f"mean_{target}_baseline": float(detail[f"{target}_baseline"].mean()),
            f"mean_{target}_counterfactual": float(
                detail[f"{target}_counterfactual"].mean()
            ),
            "mean_pm25_delta": float(detail["pm25_delta"].mean()),
            "mean_pm25_pct_change": float(detail["pm25_pct_change"].mean()),
            "n_rows": int(len(detail)),
        }
        return summary


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_pct_change(new: pd.Series, old: pd.Series, eps: float = 1e-9) -> pd.Series:
    denom = old.where(old.abs() >= eps, np.nan)
    return (new - old) / denom


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example trained weights (stand-ins for a real regression fit).
    weights = BetaWeights(
        intercept=8.5,
        coefficients={
            "traffic_ratio": 0.35,
            "industry_ratio": 18.0,
            "dust_ratio": 40.0,
            "biomass_ratio": 60.0,
        },
        target="PM2.5",
    )
    sim = PolicySimulator(weights)

    rng = np.random.default_rng(0)
    n = 5
    demo_df = pd.DataFrame(
        {
            "traffic_ratio": rng.uniform(20, 100, n),
            "industry_ratio": rng.uniform(0.1, 1.0, n),
            "dust_ratio": rng.uniform(0.1, 0.6, n),
            "biomass_ratio": rng.uniform(0.02, 0.15, n),
        }
    )

    print("Baseline prediction:")
    print(sim.predict(demo_df))

    print("\nWhat-if: cut traffic proxy by 30%")
    print(sim.what_if(demo_df, {"traffic_ratio": -0.30}))

    print("\nWhat-if: cut traffic 30% AND industry 20%, scenario summary:")
    print(
        sim.scenario_summary(
            demo_df, {"traffic_ratio": -0.30, "industry_ratio": -0.20}
        )
    )
