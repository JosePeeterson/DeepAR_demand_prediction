import json
import os
import numpy as np
from deepar_RegionWise_LinuxGpu_prediction import train_and_forecast

neurons = 660
layers = 1
batch_size = 8
learning_rate = 0.00006
dropout = 0.2
encoder_length = 24
max_epoch = 1
region = "tampines"


train_and_forecast(neurons,layers,batch_size,learning_rate,dropout,encoder_length,max_epoch,region)
