# Solar Supply Forecast in South Australia

This is my first attempt at exploring Energy and Environmental topic, which may yield useful information to anticipate potential grid volatility. A forecast on energy generation using Python, with a narrowed scoop of solar energy in South Australia.

A simple Steamlit visualisation is presented [here](https://shuuheialb-solar-supply-forecast-script-model-cakycc.streamlit.app/).

## How to Run

Run `etl.py`, then run `model.py`. The process may take some time (~0.5 min for ETL and 7-8 mins for modelling respectively).

## Data

Data is collected with [OpenNEM API](https://opennem.org.au/) and [Open Meteo API](https://open-meteo.com/), made available under [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Methodology

These models are considered:

1. Benchmark model: naive, 7-day window average, Triple Exponential Smoothing

2. Statistical models:

a. Seasonal Autoregressive Integrated Moving Average with exogenous variables (SARIMAx), i.e. temperature (averaged) and solar irradiance

b. Croston's method

3. Machine learning models: LightGBM, XGBoost

These models are not considered:

1. LSTM. Its associated Python library `TensorFlow` ideally requires an additional CuDA GUI requirement which is currently unavailable in Author's device.


Python library [StatsForecast and MLForecast by Nixtla](https://nixtlaverse.nixtla.io/) is borrowed for these functions.

The current cross validation implementation is a simple forward chain, with 30-day steps (i.e. monthly) for shorter series and 90-day steps (i.e. quarter yearly) for longer series.



More ideas can be read at [the Jupyter's version here](https://nbviewer.org/github/ShuuheiAlb/solar-supply-forecast/blob/main/tmp/nb.ipynb).

## Other tools used

VS Code Notebook, pipreqs