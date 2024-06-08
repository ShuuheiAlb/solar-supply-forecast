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

def generate_basic_features_(df):
    # Time editted vars
    df["sin"] = np.sin(df["Date"].dt.day_of_year/365.25)
    df["cos"] = np.cos(df["Date"].dt.day_of_year/365.25)

    # Shift & rolling max vars
    df["xn-1"] = df["Energy"].shift(1).fillna(method='bfill').fillna(method='ffill')
    df["xn-2"] = df["Energy"].shift(2).fillna(method='bfill').fillna(method='ffill')
    df["maxn-7"] = df["Energy"].rolling(7).max().fillna(method='bfill').fillna(method='ffill')
    df["minn-7"] = df["Energy"].rolling(7).min().fillna(method='bfill').fillna(method='ffill')
    return(df)

def generate_basic_features(df, df_past = None):
    # Processing for each name
    dfs = []
    if df_past is None:
        return(generate_basic_features_(df))
    for name in df["Name"].unique():
        df_ = df[df["Name"] == name]
        df_past_ = df_past[df_past["Name"] == name]
        if df_past_ is not None:
            df_len_ = len(df_)
            df_ = pd.concat([df_past_, df_])
            df_ = generate_basic_features_(df_)
            df_ = df_[-df_len_:]
        else:
            df_ = generate_basic_features_(df_)
        dfs.append(df_)
    return(pd.concat(dfs))

sol_hist = generate_basic_features(sol_hist)
#%%

# === Accuracy from Linear Regression
# Ideally need capacity data to anchor errors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

features = sol_hist.columns.drop(["Energy", "Date", "Name"])
for name in sol["Name"].unique():
    sol_hist_ = sol_hist[sol_hist["Name"] == name] \
                    .sample(frac=1).reset_index(drop=True) # shuffle
    x = sol_hist_[features]
    y = sol_hist_["Energy"]
    scores = cross_validate(LinearRegression(), x, y, cv=10,
                            scoring=("r2", "neg_root_mean_squared_error"),
                            return_train_score=True)
    print(f'{name} r2 score {scores["test_r2"].mean()} std {scores["test_r2"].std()}')
    print(f'{name} RMSE {-scores["test_neg_root_mean_squared_error"].mean()} std {scores["test_neg_root_mean_squared_error"].std()}')
    # Soon: Standard error

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
    return sf.forecast(h)



# %%

# Quest iterative prediction (SOON)
def iterative_predict(model, df, df_past):
    idxs = df.groupby("Name").apply(lambda x: x["Energy"].isna().index[0]).tolist()
    generate_basic_features(df.loc[idxs, :], df_past)

#  Save in pickle
hists = sol_hist #rename
tests = generate_basic_features(sol_test, sol_hist)
quests = generate_basic_features(sol_quest, pd.concat([sol_hist, sol_test]))
preds = []
for name in hists["Name"].unique():
    hist = hists[hists["Name"] == name]
    model = LinearRegression().fit(hist[features], hist["Energy"])
    test = tests[tests["Name"] == name]
    pred = pd.DataFrame(test[["Name", "Date"]])
    pred["Energy"] = model.predict(test[features])
    preds.append(pred)
    quests["Energy"] = model.predict(quests[features])
preds = pd.concat(preds)

with open('data/model.pkl', 'wb') as outp:
    hists.to_pickle(outp)
    tests.to_pickle(outp)
    preds.to_pickle(outp)
    quests.to_pickle(outp)

# %%
