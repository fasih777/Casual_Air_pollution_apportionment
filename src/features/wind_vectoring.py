"""
wind_vectoring.py

Computes orthogonal spatial wind projections (U and V vectors) from scalar
wind speed and direction.

Wind vectors are necessary for the causal model to understand the upstream
influences of pollution sources (e.g., separating southerly winds from
easterly winds).

Formulas used:
    U = -v * sin(pi * theta / 180)
    V = -v * cos(pi * theta / 180)

Where `v` is the scalar wind speed, and `theta` is the wind direction in
meteorological degrees (0 = North, 90 = East, 180 = South, 270 = West).
The negative sign ensures that the vectors point towards where the wind
is blowing *to*, which is standard meteorological convention for U and V
components.

Usage
-----
    import pandas as pd
    from src.features.wind_vectoring import compute_wind_vectors

    df = pd.read_csv("weather_data.csv")
    df = compute_wind_vectors(df, speed_col="wind_speed", dir_col="wind_direction")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _require_columns(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    """Return the name of the first missing column, or None if all present."""
    for c in cols:
        if c not in df.columns:
            return c
    return None

def compute_wind_vectors(
    df: pd.DataFrame,
    speed_col: str = "wind_speed",
    dir_col: str = "wind_direction",
    u_col_name: str = "wind_u",
    v_col_name: str = "wind_v",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Project scalar wind speed and direction into orthogonal U (East-West)
    and V (North-South) components.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing wind speed and direction.
    speed_col : str
        Column name for scalar wind speed.
    dir_col : str
        Column name for wind direction in degrees (0-360).
    u_col_name : str
        Desired name for the output U-vector column.
    v_col_name : str
        Desired name for the output V-vector column.
    inplace : bool
        If True, mutate and return `df` directly. Otherwise (default)
        return a copy, leaving the input untouched.

    Returns
    -------
    pd.DataFrame
        DataFrame with the U and V vector columns added.
    """
    out = df if inplace else df.copy()

    missing = _require_columns(out, [speed_col, dir_col])
    if missing:
        logger.warning(
            "compute_wind_vectors: missing column '%s'; returning NaNs for wind vectors", 
            missing
        )
        out[u_col_name] = np.nan
        out[v_col_name] = np.nan
        return out

    # Convert degrees to radians
    theta_rad = np.pi * out[dir_col] / 180.0

    # Calculate U and V components
    # U: East-West vector (positive = blowing to East)
    # V: North-South vector (positive = blowing to North)
    out[u_col_name] = -out[speed_col] * np.sin(theta_rad)
    out[v_col_name] = -out[speed_col] * np.cos(theta_rad)

    return out

# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_df = pd.DataFrame({
        "wind_speed": [10.0, 5.0, 15.0, 0.0],
        "wind_direction": [90.0, 180.0, 225.0, 0.0]  # East, South, South-West, North
    })
    
    # 90 deg (East wind): blowing FROM east TO west. U should be negative, V approx 0
    # 180 deg (South wind): blowing FROM south TO north. U approx 0, V should be positive
    # 225 deg (SW wind): blowing FROM SW TO NE. U positive, V positive.

    result = compute_wind_vectors(demo_df)
    print("Wind Vector Projections:")
    print(result)
