# Solar Supply Forecast in South Australia

This is my first attempt at exploring Energy and Environmental topic, which may yield useful information to anticipate potential grid volatility. A forecast on energy generation using Python, with a narrowed scoop of solar energy in South Australia.

A simple Steamlit visualisation is presented [here](https://shuuheialb-solar-supply-forecast-script-model-cakycc.streamlit.app/).

## How to Run

Run `etl.py`, then run `model.py`, then `streamlit run main.py`. The ETL and modelling process may take some time (~0.5 min and few mins respectively).

## Data

Data is collected with [OpenNEM API](https://opennem.org.au/) and [Open Meteo API](https://open-meteo.com/), made available under [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Methodology

The basic model developed is a linear regression model of multiple variables including previous temporal values, rolling max/min, and exogenous variables. This highly relies on assumption that when each row is available, other data points at previous time will be accessible.

As a result, cross validation are not dependant on typical time-series limitation, provided that feature generations have been completed before the CV process.

Classical statistical modelling are also considered such as Triple Exponential Smoothing, Autoregressive Integrated Moving Average (ARIMA), and Croston's method. Previous sketch can read at [the Jupyter's version here](https://nbviewer.org/github/ShuuheiAlb/solar-supply-forecast/blob/main/tmp/nb.ipynb). A plan to generate these as linear modelling features is underway.

Machine learning such as LightGBM may be considered but not a priority.

## Future Improvement

Currently there is a lot of processes that are separately looped for each times series. There might be a mor efficient way.

## Other tools used

VS Code Notebook, pipreqs