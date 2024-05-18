#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import isfile
import pickle
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse

# Import data, set aside 10% based on forward-chain
#DATA = pd.read_csv("sol_data.csv")
#TEST = DATA[-round(0.1*len(DATA)):] # watch out for the missing exog data
#sol = DATA[:-round(0.1*len(DATA))]
sol = pd.read_csv("sol_data.csv")

# Select the data from establishment dates (actually should be in ETL)
for name in sol["Name"].unique():
    start_date_idx = sol[sol["Name"] == name].index[0]
    est_date_idx = sol[(sol["Name"] == name) & (sol["Energy"] > 0)].index[0]
    sol.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)
# Preview plot
for name in sol["Name"].unique():
    plt.close()
    df = sol[sol["Name"] == name]
    df.loc[:, 'Date'] = pd.to_datetime(df["Date"], yearfirst=True).dt.strftime('%m/%y') # only change format for this loop
    df.plot(x='Date', y='Energy', label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()

#%%

# === Model

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, RandomWalkWithDrift, WindowAverage, \
                                 AutoTheta, AutoETS, AutoARIMA, CrostonOptimized
from mlforecast import MLForecast
from lightgbm import LGBMRegressor # potentially?

h = 3

# Change header for modelling library
sol_sf = sol[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })

models = [Naive(), SeasonalNaive(365), RandomWalkWithDrift(), WindowAverage(7), \
          AutoTheta(), AutoETS(), AutoARIMA(), CrostonOptimized()]
sf = StatsForecast(models, freq="D", df=sol_sf)

#%%

# Potentially an alternative basic model: "stabilised" seasonal avg
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2


# %%

# === Cross-validation

# Separated based on short vs long time series
# For simplicity, load the objects from Pickle
#
# Note:
# 1. BNGSF1, BNGSF2, TBSF are very regular. Model: ARIMA
# 2. MWPS, PAREPW, HVWW; MBPS2 and MAPS2 are somewhat regular with a bit volatility
# 3. BOLIVAR, ADP; TB2SF and CBWWBA are like, wtf. Model: CrostonClassic

ds_counts = sol_sf.groupby("unique_id").size()
ds_count_limit = 500
sol_sf_filter = sol_sf["unique_id"].isin(ds_counts[ds_counts > ds_count_limit].index)
sol_sf_long = sol_sf[sol_sf_filter]
sol_sf_short = sol_sf[-sol_sf_filter]
if not isfile("save_model.pkl"):
    # combined cv results
    cv_sol_sf_long = sf.cross_validation(df=sol_sf_long, h=h, n_windows=5, step_size = 100) \
                        .reset_index()
    cv_sol_sf_short = sf.cross_validation(df=sol_sf_short, h=h, n_windows=5, step_size = 10) \
                        .reset_index()
    cv_sol_sf = pd.concat([cv_sol_sf_long, cv_sol_sf_short])
    with open('save_model.pkl', 'wb') as outp:
        pickle.dump(cv_sol_sf, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('save_model.pkl', 'rb') as inp:
        cv_sol_sf = pickle.load(inp)

# Ensemble result
cv_sol_sf["h"] = (cv_sol_sf["ds"] - cv_sol_sf["cutoff"]).dt.days
cv_sol_sf["EnsembleBaseline"] = cv_sol_sf.loc[:, "Naive":"WindowAverage"].mean(axis=1)
cv_sol_sf["EnsembleAll"] = cv_sol_sf.loc[:, "Naive":"CrostonOptimized"].mean(axis=1)

print(cv_sol_sf)

#%%

# === Accuracy

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mape, mse

def evaluate_cross_validation(df, metrics): #Simplify soon?
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y', 'h']).columns.tolist()
    evals = []
    for cutoff in df['cutoff'].unique():
        df_smp = df[df['cutoff'] == cutoff]
        for h in df_smp['h'].unique():
            df_smp_smp = df_smp[df_smp['h'] == h]
            eval_ = evaluate(df_smp_smp, metrics=metrics, models=models)
            eval_['h'] = h
            evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby(["unique_id", "metric", "h"]).mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals.reset_index()

# AIC, BIC soon
error_sol_sf = evaluate_cross_validation(cv_sol_sf, [mape, mse]) # ignore SeasonalNaive when 0
print(error_sol_sf)

#%%

# Collect final predictions: STILLSOMESSY

final = cv_sol_sf[cv_sol_sf["cutoff"] == max(cv_sol_sf["cutoff"].unique())]
final = final.merge(error_sol_sf[error_sol_sf['metric'] == "mse"][["unique_id", "h", "best_model"]], \
                    on=["unique_id", "h"], how="outer")
final = final.melt(id_vars = ["unique_id", "ds", "best_model"])
preds = final[final["best_model"] == final["variable"]] \
            .drop(columns="variable") \
            .sort_values(by = ["unique_id", "ds"]) \
            .rename(columns = {"value": "y"})
hists = sol_sf

print(preds)

# %%

# === Basic visualisation

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"

def sol_points(unique_id):
    curr_hists = hists[hists["unique_id"] == unique_id]
    curr_preds = preds[preds["unique_id"] == unique_id]
    return(dict(x = [curr_hists["ds"][-60:-h], curr_hists["ds"][-(h+1):], pd.concat([curr_hists["ds"][-(h+1):-h], curr_preds["ds"]])],
                y = [curr_hists["y"][-60:-h], curr_hists["y"][-(h+1):], pd.concat([curr_hists["y"][-(h+1):-h], curr_preds["y"]])],
                visible = True))

curr_loc = "ADP"
curr_sol_points = sol_points(curr_loc)
fig = go.Figure()
fig.add_trace(go.Scatter(x = curr_sol_points["x"][0], y = curr_sol_points["y"][0], mode='lines', name = "Historic"))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][1], y = curr_sol_points["y"][1], mode='lines', name = "Actual"))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][2], y = curr_sol_points["y"][2], mode='lines', name = "Forecast"))
fig.update_layout(barmode = 'overlay', template = "plotly_white")
fig.update_layout(
    updatemenus = [dict(direction = "down",
                        buttons = [dict(args=[sol_points(loc)],
                                        label=loc,
                                        method="restyle") for loc in sol["Name"].unique()],
                        pad = {"r": 10, "t": 10}, showactive =True,
                        x = 0.11, xanchor="left", y = 1.1, yanchor = "top")]
)

# %%

import streamlit as st

st.plotly_chart(fig, use_container_width=True)
# %%
