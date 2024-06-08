#%%

# === Basic visualisation

import streamlit as st
import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default = "plotly_mimetype+notebook_connected"

import pickle
import lib
import pandas as pd

with open('data/model.pkl', 'rb') as inp:
    hists = pickle.load(inp)
    tests = pickle.load(inp)
    preds = pickle.load(inp)
    quests = pickle.load(inp)
stations = pd.read_csv(lib.station_path).set_index("Full Name")

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("Solar Supply Forecast in South Australia")

curr_loc_full_name = st.selectbox("Location",
                        sorted(stations.index),
                        index=0,
                        placeholder="Select")

def sol_points(name):
    curr_hists = hists[hists["Name"] == name]
    curr_tests = tests[tests["Name"] == name]
    curr_preds = preds[preds["Name"] == name]
    curr_quests = quests[quests["Name"] == name]
    return(dict(x = [
                    curr_hists["Date"][-60:],
                    pd.concat([curr_hists["Date"][-1:], curr_tests["Date"]]),
                    pd.concat([curr_hists["Date"][-1:], curr_preds["Date"]]),
                    pd.concat([curr_preds["Date"][-1:], curr_quests["Date"]])
                ],
                y = [
                    curr_hists["Energy"][-60:],
                    pd.concat([curr_hists["Energy"][-1:], curr_tests["Energy"]]),
                    pd.concat([curr_hists["Energy"][-1:], curr_preds["Energy"]]),
                    pd.concat([curr_preds["Energy"][-1:], curr_quests["Energy"]])
                ],
                visible = True))
curr_sol_points = sol_points(stations.loc[curr_loc_full_name, "Name"])

YELLOW = "#e6ba72"
GREEN = "#c1c87a"
MAUVE = "#E0B0FF"
fig = go.Figure()
fig.add_trace(go.Scatter(x = curr_sol_points["x"][0],
                         y = curr_sol_points["y"][0],
                         mode='lines', name = "Historic", line={"color": YELLOW}))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][1],
                         y = curr_sol_points["y"][1],
                         mode='lines', name = "Actual Test", line={"color": YELLOW, "dash": 'dot'}))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][2],
                         y = curr_sol_points["y"][2],
                         mode='lines', name = "Forecast Test", line={"color": GREEN}))
fig.add_trace(go.Scatter(x = curr_sol_points["x"][3],
                         y = curr_sol_points["y"][3],
                         mode='lines', name = "Forecast", line={"color": MAUVE}))
fig.update_layout(barmode = 'overlay', template = "plotly_white", yaxis_title = "Energy (MW)")

st.plotly_chart(fig, use_container_width=True)

