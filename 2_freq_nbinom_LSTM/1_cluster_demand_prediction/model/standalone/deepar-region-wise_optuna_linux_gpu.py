import os
import sys
"""Windows
os.chdir("c:/Work/WORK_PACKAGE/Demand_forecasting/github/DeepAR-pytorch/My_model/2_freq_nbinom_LSTM")
"""

"""Linux """
os.chdir("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM")
sys.path.append(os.path.abspath(os.path.join("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM")))


#from ctypes import FormatError
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import os,sys

# sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\\2_freq_nbinom_LSTM')))

# sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\\2_freq_nbinom_LSTM\\1_cluster_demand_prediction\data\weather_data')))
# sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\2_freq_nbinom_LSTM\1_cluster_demand_prediction\data\demand_data')))

import torch
torch.use_deterministic_algorithms(True)

from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics import SMAPE, RMSE
from torchmetrics import R2Score, SymmetricMeanAbsolutePercentageError, MeanSquaredError

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
import torch
from pytorch_forecasting.data.encoders import TorchNormalizer
import os,sys
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf,pacf
from scipy.signal import find_peaks
import operator
import statsmodels.api as sm
from itertools import combinations
import pickle
from pytorch_forecasting import Baseline
import random
from pytorch_forecasting import DeepAR,NegativeBinomialDistributionLoss
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState
import plotly
from deepar_RegionWise_LinuxGpu_prediction_dev import train_and_forecast

"""
Set Random seed
"""

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
## additional seeding to ensure reproduciblility.
pl.seed_everything(0)

"""windows
os.chdir("c:/Work/WORK_PACKAGE/Demand_forecasting/github/DeepAR-pytorch/My_model/2_freq_nbinom_LSTM/1_cluster_demand_prediction")
"""



"""
set inputs here
(hyperparameters grid search)

"""

######### MISCELLANEOUS ###################
region ="Tampines"
cov_lag_len= 0 #we can use forecasted values, even for inflow
Target = 'target'
encoder_length = 18
######### MISCELLANEOUS ###################


######### Network Architecture ###################
p = 10 # patience no. of epochs

Loss=NegativeBinomialDistributionLoss()

######### Network Architecture ###################


######### Training Routine ###################
fdv_steps = 10 # fast_dev_run
######### Training Routine ###################


############## Inputs for 2) Persistance model ( seasonal naive forecast ) #######################
season_len = 168 # length of season
num_past_seas = 6 # number of past seasons to use in averaging
#seas_pred_strt_idx = 2035 # seasonal naive forecast start index, in hours use the df dataframe
############## Inputs for 2) Persistance model ( seasonal naive forecast ) #######################

"""Linux"""
os.chdir("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM/1_cluster_demand_prediction/data/demand_data/standalone/region_level/train_val_test_data")

"""
Import pre-processed Data

response and target are the same thing
"""
all_clstr_train_dem_data = pd.read_csv(region+'_all_clstr_train_dem_data.csv')
all_clstr_full_train_dem_data = pd.read_csv(region+'_all_clstr_full_train_dem_data.csv')

all_clstr_val_dem_data = pd.read_csv(region+'_all_clstr_val_dem_data.csv')
all_clstr_test_dem_data = pd.read_csv(region+'_all_clstr_test_dem_data.csv')

train_data = all_clstr_train_dem_data
full_train_data = all_clstr_full_train_dem_data
val_data = all_clstr_val_dem_data
test_data = all_clstr_test_dem_data







#################### add date information ts ####################
train_data["datetime"] = pd.to_datetime(train_data["datetime"])
train_data['_hour_of_day'] = train_data["datetime"].dt.hour.astype(str)
train_data['_day_of_week'] = train_data["datetime"].dt.dayofweek.astype(str)
train_data['_day_of_month'] = train_data["datetime"].dt.day.astype(str)
train_data['_day_of_year'] = train_data["datetime"].dt.dayofyear.astype(str)
train_data['_week_of_year'] = train_data["datetime"].dt.weekofyear.astype(str)
train_data['_month_of_year'] = train_data["datetime"].dt.month.astype(str)
train_data['_year'] = train_data["datetime"].dt.year.astype(str)
#################### add date information ts ####################

#################### add date information ts ####################
full_train_data["datetime"] = pd.to_datetime(full_train_data["datetime"])
full_train_data['_hour_of_day'] = full_train_data["datetime"].dt.hour.astype(str)
full_train_data['_day_of_week'] = full_train_data["datetime"].dt.dayofweek.astype(str)
full_train_data['_day_of_month'] = full_train_data["datetime"].dt.day.astype(str)
full_train_data['_day_of_year'] = full_train_data["datetime"].dt.dayofyear.astype(str)
full_train_data['_week_of_year'] = full_train_data["datetime"].dt.weekofyear.astype(str)
full_train_data['_month_of_year'] = full_train_data["datetime"].dt.month.astype(str)
full_train_data['_year'] = full_train_data["datetime"].dt.year.astype(str)
#################### add date information ts ####################

#################### add date information ts ####################
val_data["datetime"] = pd.to_datetime(val_data["datetime"])
val_data['_hour_of_day'] = val_data["datetime"].dt.hour.astype(str)
val_data['_day_of_week'] = val_data["datetime"].dt.dayofweek.astype(str)
val_data['_day_of_month'] = val_data["datetime"].dt.day.astype(str)
val_data['_day_of_year'] = val_data["datetime"].dt.dayofyear.astype(str)
val_data['_week_of_year'] = val_data["datetime"].dt.weekofyear.astype(str)
val_data['_month_of_year'] = val_data["datetime"].dt.month.astype(str)
val_data['_year'] = val_data["datetime"].dt.year.astype(str)
#################### add date information ts ####################

#################### add date information ts ####################
test_data["datetime"] = pd.to_datetime(test_data["datetime"])
test_data['_hour_of_day'] = test_data["datetime"].dt.hour.astype(str)
test_data['_day_of_week'] = test_data["datetime"].dt.dayofweek.astype(str)
test_data['_day_of_month'] = test_data["datetime"].dt.day.astype(str)
test_data['_day_of_year'] = test_data["datetime"].dt.dayofyear.astype(str)
test_data['_week_of_year'] = test_data["datetime"].dt.weekofyear.astype(str)
test_data['_month_of_year'] = test_data["datetime"].dt.month.astype(str)
test_data['_year'] = test_data["datetime"].dt.year.astype(str)
#################### add date information ts ####################

#print(list(train_data.columns))

"""
CHecK for null values
"""

print("train_data has null values?",train_data.isnull().values.any())
print("train_data has null values?",full_train_data.isnull().values.any())
print("val_data has null values?",val_data.isnull().values.any())
print("test_data has null values?",test_data.isnull().values.any())

""" 
Drop datetime column
"""
train_data = train_data.drop(columns=['datetime'])
full_train_data = full_train_data.drop(columns=['datetime'])
val_data = val_data.drop(columns=['datetime'])
test_data = test_data.drop(columns=['datetime'])







"""
Full Training Routine 
with bayesisan hyperparmeter search

Load data into TimeSeriesDataSet object

for fast development run
uncomment fast_dev_run = fdv_steps

"""

#early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=p, verbose=False, mode="min")
lr_logger = LearningRateMonitor()


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)





def objective(trial,):  
  
  neu = trial.suggest_int(name="neu",low=600,high=800,step=25,log=False)
  lay = trial.suggest_int(name="lay",low=1,high=3,step=1,log=False)
  bat = trial.suggest_int(name="bat",low=4,high=12,step=4,log=False)
  lr = trial.suggest_float(name="lr",low=0.000001,high=0.01,log=True)
  num_ep = trial.suggest_int(name="num_ep",low=20,high=30,step=2,log=False)
  enc_len = encoder_length
  pred_len = 1
  drop = trial.suggest_float(name="dropout",low=0,high=0.4,step=0.1,log=False)


  num_cols_list = list(train_data.columns[3:-7])

  cat_dict = {"_hour_of_day": NaNLabelEncoder(add_nan=True).fit(train_data._hour_of_day), \
  "_day_of_week": NaNLabelEncoder(add_nan=True).fit(train_data._day_of_week), "_day_of_month" : NaNLabelEncoder(add_nan=True).fit(train_data._day_of_month), "_day_of_year" : NaNLabelEncoder(add_nan=True).fit(train_data._day_of_year), \
      "_week_of_year": NaNLabelEncoder(add_nan=True).fit(train_data._week_of_year), "_month_of_year": NaNLabelEncoder(add_nan=True).fit(train_data._month_of_year) ,"_year": NaNLabelEncoder(add_nan=True).fit(train_data._year) }
  cat_list = ["_hour_of_day","_day_of_week","_day_of_month","_day_of_year","_week_of_year","_month_of_year","_year"]  

  train_dataset = TimeSeriesDataSet(
      train_data,
      time_idx="time_idx",
      target=Target,
      categorical_encoders=cat_dict,
      group_ids=["group"],
      min_encoder_length=enc_len,
      max_encoder_length=enc_len,
      min_prediction_length=pred_len,
      max_prediction_length=pred_len,
      time_varying_unknown_reals=[Target],
      time_varying_known_reals=num_cols_list,
      time_varying_known_categoricals=cat_list,
      add_relative_time_idx=False,
      randomize_length=False,
      scalers={},
      target_normalizer=TorchNormalizer(method="identity",center=False,transformation=None )

  )

  val_dataset = TimeSeriesDataSet.from_dataset(train_dataset,val_data, stop_randomization=True, predict=False)

  train_dataloader = train_dataset.to_dataloader(train=True, batch_size=bat)
  val_dataloader = val_dataset.to_dataloader(train=False, batch_size=bat)
  ######### Load DATA #############


  """
  Machine Learning predictions START
  1) DeepAR 

  """

  metrics_callback = MetricsCallback()

  trainer = pl.Trainer(
      max_epochs=num_ep,
      gpus=-1, #-1
      auto_lr_find=False,
      gradient_clip_val=0.1,
      limit_train_batches=1.0,
      limit_val_batches=1.0,
      logger=True,
      val_check_interval=1.0,
      callbacks=[lr_logger,metrics_callback]
  )

  #print(f"training routing:\n \n {trainer}")
  deepar = DeepAR.from_dataset(
      train_dataset,
      learning_rate=lr,
      hidden_size=neu,
      rnn_layers=lay,
      dropout=drop,
      loss=Loss,
      log_interval=20,
      log_val_interval=6,
      log_gradient_flow=False,
      # reduce_on_plateau_patience=3,
  )


  torch.set_num_threads(10)
  trainer.fit(
      deepar,
      train_dataloaders=train_dataloader,
      val_dataloaders=val_dataloader,
  )


  metrics_list = [ metrics["val_RMSE"].item()  for metrics in  metrics_callback.metrics[1:]]
  min_val_rmse_epoch = np.argmin(metrics_list)
  min_val_rmse = np.min(metrics_list)


  trial.report(min_val_rmse, min_val_rmse_epoch)

  # Handle pruning based on the intermediate value.
  if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

  return min_val_rmse






########## optuna results #####################
if __name__ == "__main__":

  study = optuna.create_study(direction="minimize")
  study.optimize(objective, timeout=12000, n_trials=150)

  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial = study.best_trial

  print("  Value: ", trial.value)

  print("  Params: ")
  for key, value in trial.params.items(): ## this is same as study.best_params
      print("    {}: {}".format(key, value))

  fig = optuna.visualization.plot_parallel_coordinate(study)
  fig.show()

  fig = optuna.visualization.plot_optimization_history(study)
  fig.show()

  fig = optuna.visualization.plot_slice(study)
  fig.show()

  fig = optuna.visualization.plot_param_importances(study)
  fig.show()

  print("Best hyperparameters:", study.best_params)

  neurons = study.best_params["neu"]
  layers = study.best_params["lay"]
  batch_size = study.best_params["bat"]
  learning_rate = study.best_params["lr"]
  dropout = study.best_params["dropout"]
  encoder_length = encoder_length
  max_epochs = study.best_params["num_ep"]

  train_and_forecast(neurons,layers,batch_size,learning_rate,dropout,encoder_length,max_epochs,region,full_train_data,val_data,test_data)

  