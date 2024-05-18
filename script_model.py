#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta, AutoETS, AutoARIMA, CrostonOptimized
from lightgbm import LGBMClassifier


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
    plt.plot(sol[sol["Name"] == name]["Energy"], label=name)
    plt.title("Solar PV Generated Energy")
    plt.legend()
    plt.show()

#%%

# Baseline models
def naive_model(df):
    gap = 365

def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

# Nixtla models:
sol_sf = sol[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
models = [AutoARIMA(), AutoTheta(), WindowAverage(7)]
sf = StatsForecast(models, freq="D", df=sol_sf)

#%%

# Generalise input-output modelling
def format_preprocess(df, mode):
    if (mode in ["ARIMA", "Theta", "ETL"]):
        df = df[["Name", "Date", "Energy", "Temperature", "Solar Irradiance"]] \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })

    
    return df

def format_postprocess(df, mode):
    return None

# Potentially an alternative basic model: "stabilised" seasonal avg
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2
def seasonal_recent_model(df):
    gap = 365
    return (df.iloc[(len(df)-gap-3):(len(df)-gap+3), ]["Energy"].mean() + df.iloc[len(df)-1, ]["Energy"])/2

valid = add_timestamp_features(valid)

x_train, y_train = train.drop(columns=['meter_reading']), train['meter_reading'].values
x_valid, y_valid = valid.drop(columns=['meter_reading']), valid['meter_reading'].values

params = {'num_leaves': 30,
          'n_estimators': 400,
          'max_depth': 8,
          'min_child_samples': 200,
          'learning_rate': 0.1,
          'subsample': 0.50,
          'colsample_bytree': 0.75
         }

model = lgb.LGBMRegressor(**params)
model = model.fit(x_train.drop(columns=['timestamp']), y_train)

# %%

# Cross-val separated based on short vs long time series
# Note:
# 1. BNGSF1, BNGSF2, TBSF are very regular. Model: ARIMA
# 2. MWPS, PAREPW, HVWW; MBPS2 and MAPS2 are somewhat regular with a bit volatility
# 3. BOLIVAR, ADP; TB2SF and CBWWBA are like, wtf. Model: CrostonClassic
ds_counts = sol_sf.groupby("unique_id").size()
ds_count_limit = 500
sol_sf_filter = sol_sf["unique_id"].isin(ds_counts[ds_counts > ds_count_limit].index)
sol_sf_long = sol_sf[sol_sf_filter]
sol_sf_short = sol_sf[-sol_sf_filter]
# Soooo long like 10 mins, so save it in pickle
if not isfile("save_model.pkl"):
    cv_sol_sf_long = sf.cross_validation(df=sol_sf_long, h=7, n_windows=5, step_size = 100)
    cv_sol_sf_short = sf.cross_validation(df=sol_sf_short, h=7, n_windows=10, step_size = 10)
    with open('save_model.pkl', 'wb') as outp:
        pickle.dump(cv_sol_sf_long, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cv_sol_sf_short, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('save_model.pkl', 'rb') as inp:
        cv_sol_sf_long = pickle.load(inp)
        cv_sol_sf_short = pickle.load(inp)

#%%

# Calculate the error
# AIC, BIC?
def evaluate_cross_validation(df, metric):
    df["h"] = (df["ds"] - df["cutoff"]).dt.days
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y', 'h']).columns.tolist()
    evals = []
    for cutoff in df['cutoff'].unique():
        df_smp = df[df['cutoff'] == cutoff]
        for h in df_smp['h'].unique():
            df_smp_smp = df_smp[df_smp['h'] == h]
            eval_ = evaluate(df_smp_smp, metrics=[metric], models=models)
            eval_['h'] = h
            evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby(["unique_id", "h"]).mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals
model_error_sol_sf_long = evaluate_cross_validation(cv_sol_sf_long.reset_index(), mse)
model_error_sol_sf_short = evaluate_cross_validation(cv_sol_sf_short.reset_index(), mse)
print(model_error_sol_sf_long)
print(model_error_sol_sf_short)

#%%

# Collect final predictions
pred_list = []
for index, row in pd.concat([model_error_sol_sf_long, model_error_sol_sf_short]).iterrows():
    loc_str, h = index
    best_model_str = row["best_model"]
    best_model_class = globals()[row["best_model"]]
    best_model = best_model_class() if best_model_str != "WindowAverage"  else best_model_class(7)
    sol_sf_loc = sol_sf[sol_sf["unique_id"] == loc_str]
    best_sf = StatsForecast([best_model], freq="D", df=sol_sf_loc[:-7])
    pred = best_sf.forecast(h, df=sol_sf_loc[:-7], X_df = sol_sf_loc[-7:(-7+h)].drop("y", axis=1))[h]
    #print(loc_str, h, sol_sf_loc, pred)
    pred.columns.values[1] = "y"
    pred_list.append(pred)
preds = pd.concat(pred_list)
hists = sol_sf

# %%
# Basic visualisation
import plotly.graph_objects as go

def sol_points(unique_id):
    curr_hists = hists[hists["unique_id"] == unique_id]
    curr_preds = preds[preds.index == unique_id]
    return(dict(x = [curr_hists["ds"][-30:-7], curr_hists["ds"][-8:], pd.concat([curr_hists["ds"][-8:-7], curr_preds["ds"]])],
                y = [curr_hists["y"][-30:-7], curr_hists["y"][-8:], pd.concat([curr_hists["y"][-8:-7], curr_preds["y"]])],
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
fig
# %%
