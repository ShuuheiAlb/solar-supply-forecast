#%%

import lib
import numpy as np
import pandas as pd

# Import data, no set aside test data (let the real data shows it)
sol = pd.read_csv(lib.etl_out_path)

# Convert date string to date object
sol["Date"] = pd.to_datetime(sol["Date"])

# Separate data: questioned data, meaningful historical data
hist_sub_sols = []
quest_sub_sols = []
for name in sol["Name"].unique():
    sub_sol = sol[sol["Name"] == name].sort_values(by="Date")
    quest_sub_sol =  sub_sol.iloc[-lib.h:]
    quest_sub_sols.append(quest_sub_sol)
    estb_time_idx = sub_sol[sub_sol["Energy"] > 0].index[0]
    hist_sub_sol = sub_sol.loc[estb_time_idx:].iloc[:-lib.h]
    hist_sub_sols.append(hist_sub_sol)
    print(name, len(hist_sub_sol))
sol_hist = pd.concat(hist_sub_sols)
sol_quest = pd.concat(hist_sub_sols)

#%%

# === Exploratory Plot

#import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Change header for modelling library
sol_nixtla = sol_hist.replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
StatsForecast.plot(sol_nixtla)

#df = sol_nixtla[sol_nixtla["unique_id"] == "BNGSF1", "y"]
#plot_acf(df, lags=90) # find total lags: mostly less than 60
#plot_pacf(df, lags=50) # find differences

#%%

# FEATURE ENGINEERING

# Roll own cross-validation. Nixtla is very incompatible.
#from statsforecast import StatsForecast
from statsforecast.models import Naive, WindowAverage, AutoETS, AutoARIMA, AutoTBATS, CrostonOptimized
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def sf_feature(df, model_str):
    sf_models = {
        "naive": Naive(),
        "window7": WindowAverage(7),
        "ets": AutoETS(),
        "arima": AutoARIMA(),
        "croston": CrostonOptimized()
    }
    sf = StatsForecast(df=df, model=sf_models[model_str], freq="D", fallback_model=Naive())
    sf.fit(df)
    return sf.forecast(h)

def mlf_feature(df, model_str):
    mlf_models = {
        "lgbm": LGBMRegressor(verbosity=-1),
        "xgb": XGBRegressor()
    }
    mlf = MLForecast(df=df, model=mlf_models[model_str], freq="D")
    mlf.fit(df)
    return mlf.forecast(h)

# Darts' RegressionEnsembleModel??

#%%

# === Model
# SOON
# 1. AutoARIMAx (trend break etc) with rolling windows frame 60
# 2. LightGBM with feature engineering: X_(t-p), time since the incident?
# 3. TBATS?

import re
from mlforecast.target_transforms import Differences

# Modelling objects
sf_models = [Naive(), WindowAverage(7), AutoETS(), AutoARIMA(), CrostonOptimized()]
ml_models = [LGBMRegressor(verbosity=-1)]

sf = StatsForecast(sf_models, freq="D", fallback_model=Naive())
mlf = MLForecast(ml_models, freq="D", target_transforms=[Differences([11])],
                 lags=range(1, 20), date_features=["month"]) #seems working well with BNGSF

# Preprocessing datasets differently
sol_sf = pd.DataFrame(sol_nixtla)[["unique_id", "ds", "y", "Temperature", "Solar Irradiance"]]
sol_mlf = pd.DataFrame(sol_nixtla).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# prep = sf.preprocess(df)

# %%

# === Cross-validation
# For simplicity, load the objects from Pickle

from os.path import isfile
import pickle

h = lib.h
ds_limit = 500
if not isfile("data/cv_obj.pkl"):
    # Manual cvs to allow integration of various models
    # Check: arima_string(sf.fitted_[0,0].model_)
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
errors = evaluate_cv(cvs, [mape, mse])
#print(errors)

#%%

# ! The best model for unique_id, h should be saved and then prediction remade
#   But the specifics of the model like p,q should also be recorded
#   not sure if Nixtla saves them from

hists = sol_sf
final = cvs[cvs["cutoff"] == cvs.groupby("unique_id")["cutoff"].transform(max)]
final = final.merge(errors[errors['metric'] == "mse"][["unique_id", "best_model", "h"]], \
                    on=["unique_id", "h"], how="outer")
final = final.melt(id_vars = ["unique_id", "ds", "best_model"])
preds = final[final["best_model"] == final["variable"]] \
            .drop(columns="variable") \
            .sort_values(by = ["unique_id", "ds"]) \
            .rename(columns = {"value": "y"})
#print(preds)


#%% ML TRY
from mlforecast.auto import AutoLightGBM, AutoMLForecast
def evaluate(df, group):
    results = []
    for model in df.columns.drop(['unique_id', 'ds']):
        model_res = M4Evaluation.evaluate(
            'data', group, df[model].to_numpy().reshape(-1, horizon)
        )
        model_res.index = [model]
        results.append(model_res)
    return pd.concat(results).T.round(2)

auto_mlf = AutoMLForecast(models={'lgb': AutoLightGBM()}, freq="D", season_length=365)
#auto_mlf.fit(sol_mlf,
#    n_windows=10, h=7, num_samples=5,  # number of trials to run
#)
#preds = auto_mlf.predict(h)
#preds = preds.rename(columns = {"lgb": "y"})


# %%

# === Basic visualisation

import streamlit as st
import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default = "plotly_mimetype+notebook_connected"

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("Solar Supply Forecast in South Australia")

curr_loc = st.selectbox("Location code",
                        sorted(sol["Name"].unique()),
                        index=0,
                        placeholder="Select")

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
