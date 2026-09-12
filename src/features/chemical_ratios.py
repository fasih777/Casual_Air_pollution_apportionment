"""
chemical_ratios.py

Computes chemical-signature ratio features used as proxy signals for
attributing air-quality observations to likely emission sources:

    - Traffic   : NO2 / CO
    - Industry  : SO2 / PM2.5
    - Dust      : coarse particulate fraction, (PM10 - PM2.5) / PM10
    - Biomass   : K / PM2.5   (falls back to OC/EC if K is unavailable)

These are standard proxies used in source-apportionment / receptor
modeling literature:

    * NO2/CO   - NO2 decays faster than CO downwind of a source, so a
                 high, "fresh" ratio flags direct vehicular exhaust.
    * SO2/PM2.5 - SO2 has few common urban sources other than fuel/oil
                 combustion (power plants, industrial boilers, shipping),
                 so an elevated ratio relative to fine PM flags industrial
                 plumes.
    * (PM10-PM2.5)/PM10 - Dust and other mechanically-generated
                 particulates are overwhelmingly coarse-mode, so isolating
                 the coarse fraction of PM10 is a simple, robust dust proxy.
    * K/PM2.5  - Potassium (K+) is the classical biomass/crop burning
                 tracer; where K isn't measured, OC/EC ratio is a common
                 substitute (biomass smoke is OC-rich relative to EC).

Notes on assumptions
---------------------
This module doesn't know your pipeline's exact schema, so column names
are configurable via keyword arguments (with sensible defaults) and every
function is defensive about missing columns / zero denominators. Adjust
the `*_col` defaults below to match your actual dataframe once wired in.

Usage
-----
    import pandas as pd
    from src.features.chemical_ratios import compute_all_ratios

    df = pd.read_csv("station_data.csv")
    df = compute_all_ratios(df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    min_denominator: float = 1e-6,
) -> pd.Series:
    """
    Elementwise numerator/denominator with guards against divide-by-zero
    and negative/near-zero denominators (common in noisy sensor data).

    Values where the denominator is below `min_denominator` are set to NaN
    rather than raising or producing inf, so downstream stats/models don't
    silently blow up on a bad reading.
    """
    denom = denominator.where(denominator.abs() >= min_denominator, np.nan)
    return numerator / denom


def _require_columns(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    """Return the name of the first missing column, or None if all present."""
    for c in cols:
        if c not in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Individual proxy signals
# ---------------------------------------------------------------------------


def traffic_ratio(
    df: pd.DataFrame,
    no2_col: str = "NO2",
    co_col: str = "CO",
) -> pd.Series:
    """
    Traffic proxy: NO2 / CO.

    Elevated values indicate fresh vehicular exhaust (NO2 is depleted
    faster than CO as a plume ages/disperses).
    """
    missing = _require_columns(df, [no2_col, co_col])
    if missing:
        logger.warning("traffic_ratio: missing column '%s'; returning NaNs", missing)
        return pd.Series(np.nan, index=df.index, name="traffic_ratio")

    ratio = _safe_ratio(df[no2_col], df[co_col])
    return ratio.rename("traffic_ratio")


def industry_ratio(
    df: pd.DataFrame,
    so2_col: str = "SO2",
    pm25_col: str = "PM2.5",
) -> pd.Series:
    """
    Industry proxy: SO2 / PM2.5.

    SO2 has few common non-industrial urban sources (power generation,
    industrial boilers, refineries, shipping fuel), so a high ratio
    relative to fine particulate mass flags industrial influence.
    """
    missing = _require_columns(df, [so2_col, pm25_col])
    if missing:
        logger.warning("industry_ratio: missing column '%s'; returning NaNs", missing)
        return pd.Series(np.nan, index=df.index, name="industry_ratio")

    ratio = _safe_ratio(df[so2_col], df[pm25_col])
    return ratio.rename("industry_ratio")


def dust_ratio(
    df: pd.DataFrame,
    pm10_col: str = "PM10",
    pm25_col: str = "PM2.5",
) -> pd.Series:
    """
    Dust proxy: coarse particulate fraction, (PM10 - PM2.5) / PM10.

    Windblown/road/construction dust is predominantly coarse-mode
    (aerodynamic diameter 2.5-10 um), so isolating the coarse share of
    PM10 gives a simple dust-influence signal. Clipped to [0, 1] to guard
    against sensor noise producing PM2.5 > PM10.
    """
    missing = _require_columns(df, [pm10_col, pm25_col])
    if missing:
        logger.warning("dust_ratio: missing column '%s'; returning NaNs", missing)
        return pd.Series(np.nan, index=df.index, name="dust_ratio")

    coarse = df[pm10_col] - df[pm25_col]
    ratio = _safe_ratio(coarse, df[pm10_col]).clip(lower=0, upper=1)
    return ratio.rename("dust_ratio")


def biomass_ratio(
    df: pd.DataFrame,
    k_col: str = "K",
    pm25_col: str = "PM2.5",
    oc_col: str = "OC",
    ec_col: str = "EC",
) -> pd.Series:
    """
    Biomass-burning proxy: K / PM2.5 (potassium is the classical biomass
    smoke tracer). If potassium isn't available, falls back to OC/EC
    (biomass smoke is OC-rich relative to EC compared to traffic exhaust).

    If neither the primary (K, PM2.5) nor fallback (OC, EC) inputs are
    available, returns NaNs.
    """
    has_k_inputs = _require_columns(df, [k_col, pm25_col]) is None
    if has_k_inputs:
        ratio = _safe_ratio(df[k_col], df[pm25_col])
        return ratio.rename("biomass_ratio")

    has_oc_ec_inputs = _require_columns(df, [oc_col, ec_col]) is None
    if has_oc_ec_inputs:
        logger.info(
            "biomass_ratio: '%s'/'%s' unavailable, falling back to %s/%s",
            k_col, pm25_col, oc_col, ec_col,
        )
        ratio = _safe_ratio(df[oc_col], df[ec_col])
        return ratio.rename("biomass_ratio")

    logger.warning(
        "biomass_ratio: neither (%s, %s) nor (%s, %s) available; returning NaNs",
        k_col, pm25_col, oc_col, ec_col,
    )
    return pd.Series(np.nan, index=df.index, name="biomass_ratio")


# ---------------------------------------------------------------------------
# Convenience: compute everything at once
# ---------------------------------------------------------------------------


def compute_all_ratios(
    df: pd.DataFrame,
    no2_col: str = "NO2",
    co_col: str = "CO",
    so2_col: str = "SO2",
    pm25_col: str = "PM2.5",
    pm10_col: str = "PM10",
    k_col: str = "K",
    oc_col: str = "OC",
    ec_col: str = "EC",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute all four chemical-signature ratio proxies and attach them as
    new columns: 'traffic_ratio', 'industry_ratio', 'dust_ratio',
    'biomass_ratio'.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations. Must contain whichever source columns are
        needed for each ratio; missing inputs produce NaN columns with a
        logged warning rather than raising.
    inplace : bool
        If True, mutate and return `df` directly. Otherwise (default)
        return a copy, leaving the input untouched.

    Returns
    -------
    pd.DataFrame
        `df` with the four ratio columns added.
    """
    out = df if inplace else df.copy()

    out["traffic_ratio"] = traffic_ratio(out, no2_col=no2_col, co_col=co_col)
    out["industry_ratio"] = industry_ratio(out, so2_col=so2_col, pm25_col=pm25_col)
    out["dust_ratio"] = dust_ratio(out, pm10_col=pm10_col, pm25_col=pm25_col)
    out["biomass_ratio"] = biomass_ratio(
        out, k_col=k_col, pm25_col=pm25_col, oc_col=oc_col, ec_col=ec_col
    )

    return out


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    n = 8
    demo_df = pd.DataFrame(
        {
            "NO2": rng.uniform(10, 80, n),
            "CO": rng.uniform(0.2, 2.0, n),
            "SO2": rng.uniform(2, 40, n),
            "PM2.5": rng.uniform(15, 120, n),
            "PM10": rng.uniform(30, 200, n),
            "K": rng.uniform(0.1, 5.0, n),
        }
    )
    # Ensure PM10 >= PM2.5 for a realistic demo
    demo_df["PM10"] = demo_df[["PM10", "PM2.5"]].max(axis=1) + 5

    result = compute_all_ratios(demo_df)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(result)
