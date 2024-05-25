#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import isfile
import pickle

# Import data, no set aside test data (let the real data shows it)
sol = pd.read_csv("data/etl_out.csv")

# Select the data from establishment dates (actually should be in ETL)
for name in sol["Name"].unique():
    start_date_idx = sol[sol["Name"] == name].index[0]
    est_date_idx = sol[(sol["Name"] == name) & (sol["Energy"] > 0)].index[0]
    sol.drop(index = list(range(start_date_idx, est_date_idx)), inplace=True)

# Preview plot
_ = """for name in sol["Name"].unique():
    plt.close()
    df = sol[sol["Name"] == name]
    df.loc[:, 'Date'] = pd.to_datetime(df["Date"], yearfirst=True).dt.strftime('%m/%y') # only change format for this loop
    df.plot(x='Date', y='Energy', label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()"""

#%%

# === Model

from statsforecast import StatsForecast
from statsforecast.models import Naive, WindowAverage, AutoETS, AutoARIMA, CrostonOptimized
from mlforecast import MLForecast # LGBM potentially?
import lightgbm as lgb
from mlforecast.target_transforms import Differences

# Change header for modelling library
sol_sf = sol[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })

models = [Naive(), WindowAverage(7), AutoETS(), AutoARIMA(), CrostonOptimized()]
ml_models = [lgb.LGBMRegressor(verbosity=-1)]

sf = StatsForecast(models, freq="D", df=sol_sf)
mlf = mlf = MLForecast(ml_models, freq="D") #Difference lag SOOON


# %%

# === Cross-validation
# For simplicity, load the objects from Pickle
# Hypothesis:
# * BNGSF1, BNGSF2, TBSF has lots of data and forms seasonality. Model: ARIMA, Ensemble
# * HVWW, MWPS, PAREPW is a bit shorter but also lots of data. Model: ARIMA, Ensemble
# * BOLIVAR, ADP: scale-dependent; MBPS2, MAPS2, TB2SF are pretty new
# * CBWWBA: wtf

h = 7
ds_limit = 500
if not isfile("data/cv_obj.pkl"):
    # Nixtla's results
    cvs = []
    for unique_id in sol_sf["unique_id"].unique():
        sol_ = sol_sf[sol_sf["unique_id"] == unique_id]
        # Separated based on short vs long time series
        step = 90 if len(sol_) > ds_limit else 30
        n_windows = min(np.floor(len(sol_)/step).astype(int), 10)
        cv_ = sf.cross_validation(df=sol_, h=h, n_windows=n_windows, step_size=step).reset_index()
        cvs.append(cv_)
    cvs = pd.concat(cvs)
    cvs["h"] = (cvs["ds"] - cvs["cutoff"]).dt.days
    # Ensemble result
    cvs["EnsembleFreaky"] = (cvs["AutoARIMA"] + cvs["WindowAverage"])/2
    cvs["EnsembleAll"] = cvs.loc[:, "Naive":"CrostonOptimized"].mean(axis=1)

    with open('data/cv_obj.pkl', 'wb') as outp:
        pickle.dump(cvs, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('data/cv_obj.pkl', 'rb') as inp:
        cvs = pickle.load(inp)

#print(cvs)

#%%

# === Accuracy

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mape, mse #mase soon

def evaluate_cv(df, metrics): #Simplify soon?
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y', 'h']).columns.tolist()
    evals = []
    for cutoff in df['cutoff'].unique():
        df_smp = df[df['cutoff'] == cutoff]
        for h in df_smp['h'].unique():
            df_smp_smp = df_smp[df_smp['h'] == h]
            eval_ = evaluate(df_smp_smp, metrics=metrics, models=models)
            eval_['h'] = h
            evals.append(eval_)
    # SOON: weight based on number of training samples?
    evals = pd.concat(evals)
    evals = evals.groupby(["unique_id", "metric", "h"]).mean(numeric_only=True)
    evals['best_model'] = evals.idxmin(axis=1)
    return evals.reset_index()

# The Auto function has been based on grid search using AIC so AIC/BIC not included
errors = evaluate_cv(cvs.drop('SeasonalNaive', axis=1), [mape, mse])
#print(errors)

#%%

# Collect model
# ! The best model for unique_id, h should be saved and then prediction remade
# But the specifics of the model like p,q should also be recorded
#   not sure if Nixtla saves them from

hists = sol_sf

# Wait, the last cutoff is like for each unique_id
final = cvs[cvs["cutoff"] == cvs.groupby("unique_id")["cutoff"].transform(max)]
final = final.merge(errors[errors['metric'] == "mse"][["unique_id", "best_model", "h"]], \
                    on=["unique_id", "h"], how="outer")
final = final.melt(id_vars = ["unique_id", "ds", "best_model"])
preds = final[final["best_model"] == final["variable"]] \
            .drop(columns="variable") \
            .sort_values(by = ["unique_id", "ds"]) \
            .rename(columns = {"value": "y"})
#print(preds)



# %%

# === Basic visualisation

import streamlit as st
import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default = "plotly_mimetype+notebook_connected"

st.title("Solar Supply Forecast in South Australia")

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

curr_loc = st.selectbox(
   "Location code",
   sorted(sol["Name"].unique()),
   index=0,
   placeholder="Select"
)
def sol_points(unique_id):
    curr_hists = hists[hists["unique_id"] == unique_id]
    curr_preds = preds[preds["unique_id"] == unique_id]
    return(dict(x = [curr_hists["ds"][-60:-h], curr_hists["ds"][-(h+1):], pd.concat([curr_hists["ds"][-(h+1):-h], curr_preds["ds"]])],
                y = [curr_hists["y"][-60:-h], curr_hists["y"][-(h+1):], pd.concat([curr_hists["y"][-(h+1):-h], curr_preds["y"]])],
                visible = True))
curr_sol_points = sol_points(curr_loc)

YELLOW = "#e6ba72"
GREEN = "#c1c87a"
fig = go.Figure()
fig.add_trace(go.Scatter(x = curr_sol_points["x"][0], y = curr_sol_points["y"][0], mode='lines', name = "Historic", line={"color": YELLOW}))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][1], y = curr_sol_points["y"][1], mode='lines', name = "Actual", line={"color": YELLOW, "dash": 'dot'}))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][2], y = curr_sol_points["y"][2], mode='lines', name = "Forecast", line={"color": GREEN}))
fig.update_layout(barmode = 'overlay', template = "plotly_white", yaxis_title = "Energy (MW)")

st.plotly_chart(fig, use_container_width=True)
# %%
