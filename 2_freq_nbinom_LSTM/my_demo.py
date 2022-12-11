import matplotlib.pyplot as plt
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_forecasting import NegativeBinomialDistributionLoss, DeepAR
import torch
from pytorch_forecasting.data.encoders import TorchNormalizer
import os,sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")



sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\\2_freq_nbinom_LSTM')))

print(os.path.dirname(os.path.realpath(__file__)))

data = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\\2_freq_nbinom_LSTM\\1_f_nbinom_train.csv')

#data = pd.DataFrame(np.load('2_freq_stoch_nbinom_dem.npy'))
data["date"] = pd.Timestamp("2021-09-24") + pd.to_timedelta(data.time_idx, "H")


data['_hour_of_day'] = data["date"].dt.hour#.astype(str)
data['_day_of_week'] = data["date"].dt.dayofweek#.astype(str)
data['_day_of_month'] = data["date"].dt.day#.astype(str)
data['_day_of_year'] = data["date"].dt.dayofyear#.astype(str)
data['_week_of_year'] = data["date"].dt.weekofyear#.astype(str)
data['_month_of_year'] = data["date"].dt.month#.astype(str)
data['_year'] = data["date"].dt.year#.astype(str)

data['value'] = data['value'].astype(float)
print(type(data['value'][0])) 
print(len(data.iloc[0:-620]))



max_encoder_length = 60
max_prediction_length = 20
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data.iloc[0:-620],
    time_idx="time_idx",
    target="value",
    # categorical_encoders={"series": NaNLabelEncoder(add_nan=True).fit(data.series), "_hour_of_day": NaNLabelEncoder(add_nan=True).fit(data._hour_of_day), \
    #    "_day_of_week": NaNLabelEncoder(add_nan=True).fit(data._day_of_week), "_day_of_month" : NaNLabelEncoder(add_nan=True).fit(data._day_of_month), "_day_of_year" : NaNLabelEncoder(add_nan=True).fit(data._day_of_year), \
    #     "_week_of_year": NaNLabelEncoder(add_nan=True).fit(data._week_of_year), "_month_of_year": NaNLabelEncoder(add_nan=True).fit(data._month_of_year) ,"_year": NaNLabelEncoder(add_nan=True).fit(data._year)},
    group_ids=["series"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["value"],
    time_varying_known_reals=["_hour_of_day","_day_of_week","_day_of_month","_day_of_year","_week_of_year","_month_of_year","_year" ],
    add_relative_time_idx=False,
    randomize_length=None,
    scalers={},
    target_normalizer=TorchNormalizer(method="identity",center=False,transformation=None )

)




validation = TimeSeriesDataSet.from_dataset(
    training,
    data.iloc[-620:-420],
    # predict=True,
    stop_randomization=True,
)       



batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


# save datasets
training.save("training.pkl")
validation.save("validation.pkl")


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()



trainer = pl.Trainer(
    max_epochs=1,
    gpus=0,
    auto_lr_find=True,
    gradient_clip_val=0.1,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    #fast_dev_run=True,
    # logger=logger,
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],
)



deepar = DeepAR.from_dataset(
    training,
    hidden_size=32,
    dropout=0.1,
    loss=NegativeBinomialDistributionLoss(),
    log_interval=10,
    log_val_interval=3,
    log_gradient_flow=True,
    # reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {deepar.size()/1e3:.1f}k")

#deepar.summarize("full")


torch.set_num_threads(10)
trainer.fit(
    deepar,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


predictions = deepar.predict(data=val_dataloader,mode='prediction',return_index=True,num_workers=8,show_progress_bar=True)

# print('\n',trainer.logged_metrics["val_loss"].item(),'\n')

# deepar.hparams['my_val_loss'] = trainer.logged_metrics["val_loss"].item()
# deepar.hparams['my_train_loss'] = trainer.logged_metrics["train_loss_epoch"].item()

# print('\n',deepar.hparams['my_val_loss'],'\n')
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('tb')

# writer.add_graph(DeepAR)
# writer.close()




