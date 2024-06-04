# Solar Supply Forecast in South Australia

This is my first attempt at exploring Energy and Environmental topic, which may yield useful information to anticipate potential grid volatility. A forecast on energy generation using Python, with a narrowed scoop of solar energy in South Australia.

A simple Steamlit visualisation is presented [here](https://shuuheialb-solar-supply-forecast-script-model-cakycc.streamlit.app/).

## How to Run

Run `script_etl.py` then run `script_model.py`. The ETL and modelling may take some time (~0.5 min and 7-8 mins respectively).

## Data

Data is collected with [OpenNEM API](https://opennem.org.au/) and [Open Meteo API](https://open-meteo.com/), made available under [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Methodology

Data is modelled under:

1. Benchmark model: naive, 7-days-window average, Exponential Smoothing

2. SARIMAx with exogenous variables such as temperature and solar irrandiance. Also Croston's method.

3. LightGBM with temporal features  (still on the way)

Python library [Nixtla's StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/) is borrowed for these functions.

The current cross validation implementation is a simple forward chain, with 30-day steps (i.e. monthly) for shorter series and 90-day steps (i.e. quarter yearly) for longer series.

More ideas can be read at [the Jupyter's version here](https://nbviewer.org/github/ShuuheiAlb/solar-supply-forecast/blob/main/nb.ipynb).

## Other tools used

VS Code Notebook, pipreqs