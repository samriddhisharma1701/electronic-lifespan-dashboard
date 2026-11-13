# Reliability Studio — Electronic Component Lifespan Dashboard

A complete Streamlit website dashboard for educational Weibull reliability analysis.

## Included files

- `app.py` — complete interactive website and Weibull calculations
- `requirements.txt` — Python dependencies
- `sample_data.csv` — reproducible simulated failure-time observations

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Upload your own data

CSV needs two columns:

```csv
component,failure_time_hours
LED Bulb,850
LED Bulb,920
Capacitor,1200
```

Column names are case-insensitive. `component_name`, `failure_time`, `time`, `hours`, and `operating_hours` are also recognised aliases.

Each row is treated as an observed failure. This version does not handle right-censored units. Use positive operating times in hours and provide at least three observations per component for fitting.

## Model

Two-parameter Weibull model, with location fixed at zero:

- Reliability: `R(t) = exp(-(t / eta) ** beta)`
- Cumulative failure probability: `F(t) = 1 - R(t)`
- Characteristic life: `eta`
- Shape parameter: `beta`

The demo CSV is simulated solely to demonstrate the dashboard. Do not describe it as manufacturer or field-test data. Replace it with verified test observations for any real engineering study.

## Notes

The dashboard is an educational project, not a certified reliability tool. Validate data, assumptions and results before using them for engineering decisions.
