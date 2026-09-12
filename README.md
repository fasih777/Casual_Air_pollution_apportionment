# Causal Air Apportionment

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A modern, causal-inference-driven approach to Micro-Locality Source Apportionment for Air Quality.

Traditional source apportionment relies on expensive chemical mass balance (CMB) or Positive Matrix Factorization (PMF) which requires physical filters and laboratory analysis. This project uses **Double Machine Learning (DML)** and **Causal Inference (DoWhy)** to dynamically apportion PM2.5 to its sources (Traffic, Industry, Dust, Biomass) using real-time proxy sensor data, while rigorously stripping out meteorological confounders.

## 🏗️ Project Architecture

```text
causal-air-apportionment/
├── data/
│   ├── raw/                 # Raw API JSONs & CSVs
│   └── processed/           # Cleaned, merged time-series tables
├── notebooks/
│   └── 02_causal_dag_identification.ipynb # DoWhy DAG Visualization
├── src/
│   ├── ingestion/
│   │   ├── weather_fetcher.py      # Open-Meteo API (Confounders)
│   │   ├── aqi_fetcher.py          # OpenAQ API (Pollutants)
│   │   └── satellite_proxy.py      # NASA FIRMS API (Biomass Proxy)
│   ├── features/
│   │   ├── chemical_ratios.py      # Computes proxy ratios (e.g. NO2/CO)
│   │   └── wind_vectoring.py       # Upwind U/V trigonometric projections
│   ├── models/
│   │   ├── causal_dag.py           # DAG structure definition (DoWhy)
│   │   ├── double_ml_engine.py     # Stage-1 residualizer & Stage-2 estimator
│   │   └── counterfactual.py       # What-if scenario policy simulator
│   └── utils/
│       └── metrics.py              # (Upcoming) Refutation tests
├── app/
│   ├── dashboard.py                # (Upcoming) Streamlit UI
│   └── components/
├── requirements.txt
└── README.md
```

## 🚀 Key Features

1. **Proxy-Based Attribution**: 
   - **Traffic**: $\text{NO}_2/\text{CO}$
   - **Industry**: $\text{SO}_2/\text{PM}_{2.5}$
   - **Dust**: Coarse Fraction $(\text{PM}_{10} - \text{PM}_{2.5}) / \text{PM}_{10}$
   - **Biomass**: $\text{K}/\text{PM}_{2.5}$ or $\text{OC}/\text{EC}$
2. **Double Machine Learning**: Uses `LightGBM` in Stage 1 to regress out weather effects (temp, humidity, PBLH, wind vectors), followed by orthogonal linear estimation in Stage 2 to extract the unconfounded causal weights.
3. **Policy Simulator**: Includes a robust Counterfactual Engine to run "What-If" scenarios (e.g., "If we enforce a 30% cut on traffic proxy emissions, what is the expected drop in PM2.5?").

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fasih777/Casual_Air_pollution_apportionment.git
   cd Casual_Air_pollution_apportionment
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

**1. Generate Features:**
```python
import pandas as pd
from src.features.chemical_ratios import compute_all_ratios
from src.features.wind_vectoring import compute_wind_vectors

df = pd.read_csv("data.csv")
df = compute_all_ratios(df)
df = compute_wind_vectors(df)
```

**2. Fit the Causal Model:**
```python
from src.models.double_ml_engine import CausalSourceApportionmentEngine

engine = CausalSourceApportionmentEngine()
# Returns BetaWeights containing the true causal coefficients
weights = engine.fit(df) 
```

**3. Run Counterfactual Scenarios:**
```python
from src.models.counterfactual import PolicySimulator

sim = PolicySimulator(weights)
# Simulate a 30% reduction in traffic emissions and 20% in industry
result = sim.scenario_summary(df, {"traffic_ratio": -0.30, "industry_ratio": -0.20})
print(result)
```

## 🗺️ Roadmap
- [x] **Week 1**: Data Ingestion & Causal DAG Design
- [x] **Week 2**: Feature Engineering (Chemical Ratios & Wind Vectoring)
- [x] **Week 3**: Causal Engine & Double ML Pipeline
- [ ] **Week 4**: Refutation Tests (DoWhy) & Streamlit Interactive Dashboard
