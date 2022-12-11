
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\2_freq_nbinom_LSTM\1_cluster_demand_prediction\data\weather_data')))
os.chdir('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\\2_freq_nbinom_LSTM\\1_cluster_demand_prediction\data\weather_data')

# ########### API calls #############

# lat = 1.283658
# lon = 103.850455
# lat = str(lat)
# lon = str(lon)
# t = 1632412800 - 608400
# et = 1640257200
# cluster_list = []
# for i in range(1,14):
#     t = t + 608400
#     start = str(t)
#     cluster = "central_48_" + str(i)
#     cluster_list.append(cluster)
#     response = requests.get("https://history.openweathermap.org/data/2.5/history/city?lat=" + lat + "&lon=" + lon + "&type=hour&start=" + start + "&end=1640257200&appid=dda8e16f80893bfa91011748b8ecf77c")
#     with open(cluster+'.json',"w") as f:
#         json.dump(response.json(),f)

# ########### API calls #############

#working
#https://api.openweathermap.org/data/3.0/onecall/timemachine?lat=1.441065&lon=103.798087&dt=1634472000&appid=ce41e2be6cde1700cf8be0d874f3d717



# ########### SAVE region weather data #############

# cluster_list = ["central_48_1","central_48_2","central_48_3","central_48_4","central_48_5","central_48_6","central_48_7", "central_48_8","central_48_9","central_48_10","central_48_11","central_48_12","central_48_13"]
# date_time = np.array([])
# temp_clstr_48 = np.array([])
# hum_clstr_48 = np.array([])
# #wind_clstr_48 = np.array([])
# #wea_clstr_48 = np.array([])
# wea_desc_clstr_48 = np.array([])

# for c in cluster_list:
#     with open(c+".json",'r') as f:
#         data = json.load(f)
#         for i in range(len(data["list"])):
#             date_time = np.append(date_time, data["list"][i]["dt"])
#             temp_clstr_48 = np.append(temp_clstr_48, data["list"][i]["main"]["temp"])
#             hum_clstr_48 = np.append(hum_clstr_48, data["list"][i]["main"]["humidity"])
#             #wind_clstr_48 = np.append(wind_clstr_48, data["list"][i]["wind"]["speed"])
#             #wea_clstr_48 = np.append(wea_clstr_48, data["list"][i]["weather"][0]["main"])
#             wea_desc_clstr_48 = np.append(wea_desc_clstr_48, data["list"][i]["weather"][0]["description"])


# plt.plot(hum_clstr_48)
# plt.plot(temp_clstr_48)
# plt.title("cluster 175/wind speed")
# plt.show()

# df = pd.DataFrame()
# df["temp_clstr_48"] = temp_clstr_48
# df["hum_clstr_48"] = hum_clstr_48
# # df["wind_clstr_48"] = wind_clstr_48
# # df["wea_clstr_48"] = wea_clstr_48
# df["wea_desc_clstr_48"] = wea_desc_clstr_48
# df.to_csv("central_clstr_48_weather.csv")

# ########### SAVE region weather data #############

















