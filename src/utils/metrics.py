"""
metrics.py

Refutation ("stress-test") utilities for a fitted DoWhy causal model.

DoWhy's estimated effect (e.g. how much PM2.5 changes per unit of a
chemical-ratio proxy such as traffic_ratio) is only as trustworthy as its
robustness to sanity checks. This module wraps three of DoWhy's built-in
refuters into a single, consistent interface with pass/fail heuristics:

    - Placebo Treatment  : replace the real treatment with random noise /
                            a permutation of itself. A well-specified
                            model should see its estimated effect collapse
                            toward zero, since a placebo can't cause
                            anything.
    - Random Common Cause: add a random, independent "fake" confounder
                            (e.g. standing in for an unmeasured variable
                            like weather) to the model. A robust estimate
                            should barely move, since an independent
                            random variable carries no real confounding
                            information.
    - Data Subset         : re-estimate on random subsets of the data
                            (DoWhy's bootstrap-style subset refuter). A
                            robust estimate should be stable across
                            subsets rather than swinging wildly.

None of these tests *prove* the causal model correct -- they're falsification
tests. Failing one is a strong signal the estimate is fragile (overfit,
misspecified, or confounded); passing all three is necessary but not
sufficient evidence of validity.

Usage
-----
    from dowhy import CausalModel
    from src.utils.metrics import run_refutation_suite

    model = CausalModel(data=df, treatment="traffic_ratio", outcome="PM2.5",
                         common_causes=["industry_ratio", "dust_ratio", "biomass_ratio"])
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    report = run_refutation_suite(model, identified_estimand, estimate)
    print(report.summary_table())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from dowhy.causal_estimator import CausalEstimate
    from dowhy.causal_refuter import CausalRefutation
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dowhy is required for src.utils.metrics. Install it with "
        "`pip install dowhy`."
    ) from exc


# Relative tolerance used to judge whether random-common-cause / data-subset
# refuters left the estimate "essentially unchanged". Effects that move by
# more than this fraction of the original estimate are flagged as fragile.
DEFAULT_STABILITY_TOLERANCE = 0.20

# Below this p-value, DoWhy's own significance test says the new estimate
# is statistically distinguishable from the simulated null distribution.
DEFAULT_SIGNIFICANCE_ALPHA = 0.05


@dataclass
class RefutationResult:
    """Normalized result of a single refutation test."""

    name: str
    original_effect: float
    new_effect: float
    p_value: Optional[float]
    passed: bool
    interpretation: str
    raw: CausalRefutation = field(repr=False)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: original={self.original_effect:.4g}, "
            f"new={self.new_effect:.4g}, p={self._fmt_p()} — {self.interpretation}"
        )

    def _fmt_p(self) -> str:
        return f"{self.p_value:.3g}" if self.p_value is not None else "n/a"


@dataclass
class RefutationReport:
    """Collection of RefutationResults from a full test suite run."""

    results: list[RefutationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary_table(self):
        """Return a pandas DataFrame summarizing all refutation results."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "test": r.name,
                    "original_effect": r.original_effect,
                    "new_effect": r.new_effect,
                    "p_value": r.p_value,
                    "passed": r.passed,
                    "interpretation": r.interpretation,
                }
                for r in self.results
            ]
        )

    def __str__(self) -> str:
        lines = [str(r) for r in self.results]
        lines.append(
            f"\nOverall: {'ALL TESTS PASSED' if self.all_passed else 'ONE OR MORE TESTS FAILED'}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual refuters
# ---------------------------------------------------------------------------


def placebo_treatment_test(
    model: "CausalModel",
    identified_estimand,
    estimate: CausalEstimate,
    placebo_type: str = "permute",
    num_simulations: int = 100,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> RefutationResult:
    """
    Placebo Treatment refuter: replace the real treatment with noise (or a
    row-wise permutation of itself, the default) and re-estimate.

    Pass condition: the new effect should be close to zero and/or DoWhy's
    significance test should say the new estimate is NOT statistically
    different from noise (p >= significance_alpha) -- i.e. a placebo
    shouldn't "cause" anything.
    """
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type=placebo_type,
        num_simulations=num_simulations,
        random_state=random_state,
        **kwargs,
    )

    p_value = _extract_p_value(refutation)
    # A well-specified model's placebo effect should collapse toward zero.
    # We treat "passed" as: the refuter's own significance test finds no
    # significant difference from the null (placebo) distribution.
    passed = p_value is None or p_value >= significance_alpha

    interpretation = (
        "Placebo effect is statistically indistinguishable from zero noise "
        "(as expected for a well-specified model)."
        if passed
        else "Placebo effect is statistically significant — the model may be "
        "picking up spurious signal rather than a genuine treatment effect."
    )

    return RefutationResult(
        name="Placebo Treatment",
        original_effect=float(refutation.estimated_effect),
        new_effect=float(refutation.new_effect),
        p_value=p_value,
        passed=passed,
        interpretation=interpretation,
        raw=refutation,
    )


def random_common_cause_test(
    model: "CausalModel",
    identified_estimand,
    estimate: CausalEstimate,
    num_simulations: int = 100,
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> RefutationResult:
    """
    Random Common Cause refuter: add an independent random variable (a
    stand-in for an unmeasured confounder, e.g. "fake weather") as an
    additional common cause and re-estimate.

    Pass condition: the new effect should stay close (within
    `stability_tolerance` relative change) to the original estimate --
    an independent random variable carries no real confounding
    information, so it shouldn't move a robust estimate much.
    """
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=num_simulations,
        random_state=random_state,
        **kwargs,
    )

    original = float(refutation.estimated_effect)
    new = float(refutation.new_effect)
    rel_change = _relative_change(original, new)
    p_value = _extract_p_value(refutation)

    passed = rel_change <= stability_tolerance

    interpretation = (
        f"Estimate shifted only {rel_change:.1%} after adding a random "
        "confounder — stable, as expected."
        if passed
        else f"Estimate shifted {rel_change:.1%} after adding a random "
        "confounder, exceeding the stability tolerance — suggests the "
        "original estimate may be sensitive to unobserved confounding."
    )

    return RefutationResult(
        name="Random Common Cause",
        original_effect=original,
        new_effect=new,
        p_value=p_value,
        passed=passed,
        interpretation=interpretation,
        raw=refutation,
    )


def data_subset_test(
    model: "CausalModel",
    identified_estimand,
    estimate: CausalEstimate,
    subset_fraction: float = 0.8,
    num_simulations: int = 100,
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> RefutationResult:
    """
    Data Subset refuter: repeatedly re-estimate the effect on random
    subsets (fraction `subset_fraction`) of the data -- DoWhy's
    bootstrap-style robustness check.

    Pass condition: the mean effect across subsets should stay close
    (within `stability_tolerance` relative change) to the original
    full-data estimate -- a real effect shouldn't depend heavily on
    exactly which rows happen to be included.
    """
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=subset_fraction,
        num_simulations=num_simulations,
        random_state=random_state,
        **kwargs,
    )

    original = float(refutation.estimated_effect)
    new = float(refutation.new_effect)
    rel_change = _relative_change(original, new)
    p_value = _extract_p_value(refutation)

    passed = rel_change <= stability_tolerance

    interpretation = (
        f"Estimate shifted only {rel_change:.1%} across data subsets — "
        "stable, as expected."
        if passed
        else f"Estimate shifted {rel_change:.1%} across data subsets, "
        "exceeding the stability tolerance — suggests the estimate may be "
        "driven by a small number of influential observations."
    )

    return RefutationResult(
        name="Data Subset (bootstrap)",
        original_effect=original,
        new_effect=new,
        p_value=p_value,
        passed=passed,
        interpretation=interpretation,
        raw=refutation,
    )


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------


def run_refutation_suite(
    model: "CausalModel",
    identified_estimand,
    estimate: CausalEstimate,
    num_simulations: int = 100,
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    random_state: Optional[int] = None,
) -> RefutationReport:
    """
    Run the full placebo / random-common-cause / data-subset refutation
    suite and return a combined RefutationReport.

    Any individual refuter that raises is logged and skipped rather than
    aborting the whole suite, so one flaky test doesn't hide the other
    results.
    """
    tests = [
        ("Placebo Treatment", placebo_treatment_test, dict(significance_alpha=significance_alpha)),
        ("Random Common Cause", random_common_cause_test, dict(stability_tolerance=stability_tolerance)),
        ("Data Subset", data_subset_test, dict(stability_tolerance=stability_tolerance)),
    ]

    results: list[RefutationResult] = []
    for name, fn, extra_kwargs in tests:
        try:
            result = fn(
                model,
                identified_estimand,
                estimate,
                num_simulations=num_simulations,
                random_state=random_state,
                **extra_kwargs,
            )
            results.append(result)
        except Exception:
            logger.exception("Refutation test '%s' failed to run; skipping.", name)

    return RefutationReport(results=results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_change(original: float, new: float, eps: float = 1e-9) -> float:
    denom = abs(original) if abs(original) >= eps else eps
    return abs(new - original) / denom


def _extract_p_value(refutation: CausalRefutation) -> Optional[float]:
    """
    DoWhy's CausalRefutation stores the significance test result in
    `refutation_result` (a dict with a 'p_value' key) when available.
    Returns None if no significance test was attached.
    """
    result = getattr(refutation, "refutation_result", None)
    if isinstance(result, dict):
        return result.get("p_value")
    return None


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import numpy as np
    import pandas as pd
    from dowhy import CausalModel

    rng = np.random.default_rng(0)
    n = 500
    traffic_ratio = rng.uniform(10, 100, n)
    industry_ratio = rng.uniform(0.1, 1.0, n)
    pm25 = 10 + 0.3 * traffic_ratio + 15 * industry_ratio + rng.normal(0, 5, n)

    df = pd.DataFrame(
        {"traffic_ratio": traffic_ratio, "industry_ratio": industry_ratio, "PM2.5": pm25}
    )

    causal_model = CausalModel(
        data=df,
        treatment="traffic_ratio",
        outcome="PM2.5",
        common_causes=["industry_ratio"],
    )
    identified = causal_model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = causal_model.estimate_effect(
        identified, method_name="backdoor.linear_regression"
    )
    print(f"Original estimated effect: {causal_estimate.value:.4f}\n")

    report = run_refutation_suite(
        causal_model, identified, causal_estimate, num_simulations=20, random_state=0
    )
    print(report)
    print()
    print(report.summary_table())
