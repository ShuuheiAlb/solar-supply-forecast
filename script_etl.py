
#%%
import json
import sqlite3
from sys import exit
from datetime import datetime, timezone, timedelta
from os.path import isfile

import requests
import numpy as np
import pandas as pd

# Path functions
station_path = "/station"
def get_location_path (id):
    return f"/location/{id}"

added_station_locs = {"MWPS": {"lat": -34.022, "lng": 139.686}}

# Response functions
def opennem_response(path, params={}, exception_flag=True):
    url = "https://api.opennem.org.au" + path
    response = requests.get(url, params=params)
    if exception_flag:
        response.raise_for_status()
    return response

def open_meteo_response(path, params={}):
    url = "https://archive-api.open-meteo.com/v1/archive" + path
    response = requests.get(url, params=params)
    return response

# For debug
def json_print(data):
    print(json.dumps(data, indent=2))

def exploratory_test():
    path = "/weather/station" # free to update
    json_print(opennem_response(path).json())
    exit()

#exploratory_test()


#%%

# Check if we can create a database
# SOON: sqlite3
csv_path = "data/etl_out.csv"
db_path = "data/etl_out.db"
if isfile(csv_path):
    print(f"File {csv_path} already exists")
else:
    try:
        # Compile station list
        data = opennem_response(station_path).json()["data"]
        stations = set()
        for entry in data:
            station_code = entry["code"]
            station_name = entry["name"]
            location_id = entry["location_id"]
            for facility in entry["facilities"]:
                if not ("fueltech" in facility and facility["fueltech"]["code"] == "solar_utility"):
                    continue
                if not (facility["network_region"] == "SA1"):
                    continue
                stations.add((station_code, station_name, location_id))
                # soon: enter total unit capacity?

        # Extracting solar supply data for each station, daily for 5 years, until yesterday (GMT).
        # DF will have station code, date, energy, temperature + irradiance (weather),
        #   latitude (location)
        station_supply_dfs = []
        for station in stations:
            station_code, _, station_loc_id = station
            ebs_path = f"/stats/energy/station/NEM/{station_code}"
            params = {
                "interval": "1d",
                "period": "5Y"
            }
            response = opennem_response(ebs_path, params, False)
            if response.status_code != 200: # Some just do not have statistics available
                continue
            data = response.json()["data"]

            # Get the station's coordinate
            station_loc_record = opennem_response(get_location_path(station_loc_id)).json()["record"] \
                                    if not (station_code in added_station_locs) \
                                    else added_station_locs[station_code]
            station_coord = (station_loc_record["lng"], station_loc_record["lat"])

            # Get the total energy for all plants in a station
            for entry in data:
                if entry["data_type"] == "energy":
                    single_plant_supply_array = np.array(entry["history"]["data"][:-1])
                    station_supply_array = single_plant_supply_array \
                                            if station_supply_array is None \
                                            else station_supply_array + single_plant_supply_array

            # Get historic temperature + irradiance
            params = {"latitude": station_loc_record["lat"],
                "longitude": station_loc_record["lng"],
                "start_date": (datetime.now(timezone.utc) - timedelta(days=len(single_plant_supply_array))).strftime("%Y-%m-%d"),
                "end_date": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
                "daily": "temperature_2m_mean,shortwave_radiation_sum",
                "timezone": "Europe/London"
            }
            weather_data = open_meteo_response("", params).json()["daily"]
            mean_temps = weather_data["temperature_2m_mean"]
            tot_rads = weather_data["shortwave_radiation_sum"]

            # Create the energy supply for station
            station_supply_df = pd.DataFrame({"Name": [station_code] * len(station_supply_array),
                                    "Date" : [datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0) - timedelta(days=i)
                                              for i in range(len(station_supply_array), 0, -1)],
                                    "Energy": station_supply_array,
                                    "Temperature": mean_temps,
                                    "Solar Irradiance": tot_rads,
                                    "Latitude": [station_loc_record["lat"]] * len(station_supply_array)
                                })
            station_supply_dfs.append(station_supply_df)
        
        # Saving, with caveat: OpenMeteo has a range of 1-7 days missing
        solar_supply_df = pd.concat(station_supply_dfs)
        solar_supply_df.to_csv(csv_path)
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# %%
