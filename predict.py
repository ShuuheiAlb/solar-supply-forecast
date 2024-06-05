#%%

# # Collect model
# ! The best model for unique_id, h should be saved and then prediction remade
#

with open('data/model.pkl', 'rb') as inp:
    cv_sol_sf = pickle.load(inp)
    
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

#print(preds)

# %%

# === Basic visualisation

# !! 1. Mark if the upper bound is really small, cs higher visual gap
# !! 2. Give location names (check json_parser)
# !! 3. Routine update?

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
