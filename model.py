#%%

# Check if model exists
from os.path import isfile
if isfile("data/model.pkl"):
    yes = input("Model has been developed. Do you still want to continue (Y)? ")
    if yes != "Y":
        print("Quitting ...")
        exit()

#%%

import lib
import numpy as np
import pandas as pd

# Import data, no set aside test data (let the real data shows it)
sol = pd.read_csv(lib.etl_out_path)
# Ignore unnamed columns
sol = sol.loc[:, ~sol.columns.str.contains('^Unnamed')]

# Convert date string to date object
sol["Date"] = pd.to_datetime(sol["Date"])

# Separate data: questioned data, test data, meaningful historical data
h = lib.h
quest_sub_sols = []
test_sub_sols = []
hist_sub_sols = []
for name in sol["Name"].unique():
    sub_sol = sol[sol["Name"] == name].sort_values(by="Date")
    quest_sub_sol = sub_sol.iloc[-h:]
    quest_sub_sols.append(quest_sub_sol)
    test_sub_sol = sub_sol.iloc[-2*h:-h]
    test_sub_sols.append(test_sub_sol)
    estb_time_idx = sub_sol[sub_sol["Energy"] > 0].index[0]
    hist_sub_sol = sub_sol.loc[estb_time_idx:].iloc[:-2*h]
    hist_sub_sols.append(hist_sub_sol)
sol_quest = pd.concat(quest_sub_sols)
sol_test = pd.concat(test_sub_sols)
sol_hist = pd.concat(hist_sub_sols)

#%%

# === Exploratory Plot

import matplotlib.pyplot as plt
from statsforecast import StatsForecast

# Change header for modelling library
sol_nixtla = sol_hist.replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
StatsForecast.plot(sol_nixtla)

# Comparative 2D plot for forecast vs actual
def compare(df1, df2, df3=None):
    fig, ax = plt.subplots()
    ax.scatter(df1, df2, c=df3)

    # Plot y=x helpline
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()
    return

#%%

# === Basic feature engineering

from itertools import product

def generate_basic_features_(df):
    # Time editted vars
    df["doy"] = df["Date"].dt.day_of_year
    df["sin"] = np.sin(df["Date"].dt.day_of_year/365.25)
    df["cos"] = np.cos(df["Date"].dt.day_of_year/365.25)

    # Shift & rolling max vars
    shift_num = 7
    for i in range(1, shift_num + 1):
        df["xn-" + str(i)] = df["Energy"].shift(i).fillna(method='bfill').fillna(method='ffill')
    df["maxn-14"] = df["Energy"].rolling(14).max().fillna(method='bfill').fillna(method='ffill')
    df["minn-14"] = df["Energy"].rolling(14).min().fillna(method='bfill').fillna(method='ffill')

    # Second-degree
    exog_features = ["Solar Irradiance", "Temperature", "Precipitation"]
    for e1, e2 in product(exog_features, exog_features):
        df[e1 + "_times_" + e2] = df[e1] * df[e2]

    return(df)

def generate_basic_features(df, df_past = None):
    # Processing for each name
    dfs = []
    for name in df["Name"].unique():
        df_ = pd.DataFrame(df[df["Name"] == name])
        df_past_ = df_past[df_past["Name"] == name] if df_past is not None else None
        if df_past_ is None:
            df_ = generate_basic_features_(df_)
        else:
            df_len_ = len(df_)
            df_ = pd.concat([df_past_, df_])
            df_ = generate_basic_features_(df_)
            # df_ = generate_sf_features(df_)
            df_ = df_[-df_len_:]
        dfs.append(df_)
    return(pd.concat(dfs))

sol_hist = generate_basic_features(sol_hist)

#%%
# === Next step: Classical vars (SOON)
#from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, CrostonOptimized

def generate_sf_features(df, model_str, limit=30, h=1):
    df_sf = pd.DataFrame(df[["Name", "Date", "Energy"]]) \
                .replace("", np.nan).dropna() \
                .rename(columns = {
                    "Name": "unique_id",
                    "Date": "ds",
                    "Energy": "y"
                })
    sf_models = {
        "ets": AutoETS(),
        "arima": AutoARIMA(),
        "croston": CrostonOptimized()
    }
    df_sf = df_sf[-min(len(df_sf), 30):]
    sf = StatsForecast(model=sf_models[model_str], freq="D")
    sf.fit(df_sf)
    # for each df, vectorise
    return sf.forecast(h)

#%%

# === Accuracy from Ridge/Lasso Regression
# SOON: paralellisation with joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
from skopt.callbacks import DeltaYStopper

features = sol_hist.columns.drop(["Energy", "Date", "Name"])
bys = {}
for name in sol["Name"].unique():
    sol_hist_ = sol_hist[sol_hist["Name"] == name] \
                    .sample(frac=1).reset_index(drop=True) # shuffle
    x = sol_hist_[features]
    y = sol_hist_["Energy"]
    ridge_search = {"model": [Ridge()], "model__alpha": Real(0, 10)}
    lasso_search = {"model": [Lasso()], "model__alpha": Real(0, 10)}
    pipe = Pipeline([('model', LinearRegression())])
    by = BayesSearchCV(pipe, [(ridge_search, 30), (lasso_search, 30)], # n_iter=30
                       cv=5, scoring="r2",
                       optimizer_kwargs={'base_estimator': 'RF'},
                       fit_params={"callback": DeltaYStopper(delta=1e-2)},
                       n_jobs=-1)
    by.fit(x, y)
    print(f"{name} best param: {by.best_params_}")
    bys[name] = by
    # Soon: Standard error


# %%

# Quest iterative prediction (SOON)
def iter_predict(model, df, df_past):
    idxs = df.groupby("Name").apply(lambda x: x["Energy"].isna().index[0]).tolist()
    return generate_basic_features(df.loc[idxs, :], df_past)

# Arrange visualised datasets
hists = sol_hist #rename
tests = generate_basic_features(sol_test, sol_hist)
quests = generate_basic_features(sol_quest, pd.concat([sol_hist, sol_test]))
preds = []
for name in hists["Name"].unique():
    hist = hists[hists["Name"] == name]

    test = tests[tests["Name"] == name]
    model = bys[name].best_params_["model"]
    alpha = bys[name].best_params_["model__alpha"]
    model.set_params(alpha=alpha)
    model.fit(hist[features], hist["Energy"])
    pred = pd.DataFrame(test[["Name", "Date"]])
    pred["Energy"] = model.predict(test[features])
    preds.append(pred)

    quest_idx = (quests["Name"] == name)
    quests.loc[quest_idx, "Energy"] = model.predict(quests.loc[quest_idx, features])
preds = pd.concat(preds)

#  Save in pickle
import pickle
with open(lib.model_out_path, 'wb') as outp:
    for df in [hists, tests, preds, quests]:    
        pickle.dump(df, outp, protocol=pickle.HIGHEST_PROTOCOL)
    # Also saves the Bayes objects
    pickle.dump(bys, outp, protocol=pickle.HIGHEST_PROTOCOL)

# %%
