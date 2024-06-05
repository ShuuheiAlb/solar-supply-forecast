
#%%
import lib

import json
from sys import exit
from datetime import datetime, timezone, timedelta
from os.path import isfile

import requests
import numpy as np
import pandas as pd

# Path functions
station_path = "/station"
def get_location_path(id):
    return f"/location/{id}"

added_station_locs = {"MWPS": {"lat": -34.022, "lng": 139.686}}

openmeteo_archive_undefined_past_days_limit = 7

# Response functions
def opennem_response(path, params={}, exception_flag=True):
    url = "https://api.opennem.org.au" + path
    response = requests.get(url, params=params)
    if exception_flag:
        response.raise_for_status()
    return response

def open_meteo_response(path, params={}, is_archive=True):
    main_url = "https://archive-api.open-meteo.com/v1/archive" if is_archive \
                else "https://api.open-meteo.com/v1/forecast"
    url = main_url + path
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
if isfile(lib.etl_out_path):
    print(f"File {lib.etl_out_path} already exists")
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

        # Extracting solar supply data for each station, daily for 5 years, until yesterday GMT (today's data unavailable)
        # DF will have station code, date, energy, temperature + irradiance (weather), location
        station_supply_dfs = []
        for station in stations:
            station_code, _, station_loc_id = station
            print(f"Extracting data from {station_code} station ...")

            ebs_path = f"/stats/energy/station/NEM/{station_code}"
            params = {
                "interval": "1d",
                "period": "5Y"
            }
            response = opennem_response(ebs_path, params, exception_flag=False)
            if response.status_code != 200: # Some just do not have statistics available
                continue
            data = response.json()["data"]

            # Get the station's coordinate
            station_loc_record = opennem_response(get_location_path(station_loc_id)).json()["record"] \
                                    if not (station_code in added_station_locs) \
                                    else added_station_locs[station_code]
            station_coord = (station_loc_record["lng"], station_loc_record["lat"])

            # Get the total energy for all plants in a station
            station_supply_array = None
            for entry in data:
                if entry["data_type"] == "energy":
                    single_plant_supply_array = np.array(entry["history"]["data"][:-1])
                    station_supply_array = single_plant_supply_array \
                                            if station_supply_array is None \
                                            else station_supply_array + single_plant_supply_array

            # In addition, collect historic weather from OpenMeteo, plus 7 day predictions afterward
            # Its archive API sometimes do not record last few days, but the std API does
            #    so we take some past_days_limit (i.e. 7) of data from the std API instead
            datetime_now = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
            archive_params = {"latitude": station_loc_record["lat"],
                "longitude": station_loc_record["lng"],
                "start_date": (datetime_now - timedelta(days=len(single_plant_supply_array))).strftime("%Y-%m-%d"),
                "end_date": (datetime_now - timedelta(days=1+openmeteo_archive_undefined_past_days_limit)).strftime("%Y-%m-%d"),
                "daily": "temperature_2m_mean,shortwave_radiation_sum",
                "timezone": "Europe/London"
                }
            forecast_params = {"latitude": station_loc_record["lat"],
                "longitude": station_loc_record["lng"],
                "past_days": openmeteo_archive_undefined_past_days_limit,
                "forecast_days": lib.h,
                "daily": "temperature_2m_mean,shortwave_radiation_sum",
                "timezone": "Europe/London"
                }
            archive_weather_data = open_meteo_response("", archive_params).json()["daily"]
            forecast_weather_data = open_meteo_response("", forecast_params, is_archive=False).json()["daily"]
            mean_temps = archive_weather_data["temperature_2m_mean"] + forecast_weather_data["temperature_2m_mean"]
            tot_rads = archive_weather_data["shortwave_radiation_sum"] + forecast_weather_data["shortwave_radiation_sum"]

            # Create the energy supply for station
            station_supply_array_padded = np.append(station_supply_array, [np.nan] * lib.h)
            station_supply_df_expected_len = len(station_supply_array_padded)
            station_supply_df = pd.DataFrame({"Name": [station_code] * station_supply_df_expected_len,
                                    "Date": [datetime_now - timedelta(days=i) + timedelta(days=lib.h) \
                                             for i in range(station_supply_df_expected_len, 0, -1)],
                                    "Energy": station_supply_array_padded,
                                    "Temperature": mean_temps,
                                    "Solar Irradiance": tot_rads,
                                    "Latitude": [station_loc_record["lat"]] * station_supply_df_expected_len
                                })
            station_supply_dfs.append(station_supply_df)
        
        # Saving, with caveat: OpenMeteo has a range of 1-7 days missing
        solar_supply_df = pd.concat(station_supply_dfs)
        solar_supply_df.to_csv(lib.etl_out_path)
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# %%
