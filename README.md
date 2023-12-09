![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/36c1fc50-6147-4ccb-b41c-660f05bc27ea)# Background

# Why is demand prediction necessary?

Vehicle Demand Prediction is necessary to perform effective vehicle rebalancing and meet future demand to maximise the expected profits at the end of a period.

# What is DeepAR?

Although Autoregressive conditional Negative-binomial model (ARCNBM) captures the visible and discernible underlying patters in exploratory data analysis there may still be indiscernible underlying patterns that elude the human notion of patterns like seasonality. 

These can be captured by a deep learning model that can learn latent features in data. 

Specifically, Recurrent neural networks (RNN) are used to detect patterns in sequential data and make future predictions. Multi-variate probabilistic forecast is needed to quantify uncertainty and get bounds on the demand forecast. This is helpful for the downstream task of vehicle rebalancing.

DeepAR (David Salinas, 2020) is a sequence to sequence model (either a many-to-one or many-to-many model) that summarizes the context information in the inputs to produce appropriate multi-step outputs. 

DeepAR uses LSTM networks to output probabilistic forecasts including quantile estimates by training them to learn parameters of probabilistic distributions, namely the negative binomial distribution. 

Beyond univariate time series forecasting, DeepAR can also perform multivariate time series forecasting where it learns local and global patterns in the data across all the related target time series.

# How is DeepAR trained? (ie. how do we get a trained DeepAR model)

### Training Mechanism

| DeepAR Encoder Network                                                                                                  | DeepAR Decoder Network                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d57479a0-5cee-4f90-8f26-07c9f5f647ad)| ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d9a31f75-12fe-4264-9f6b-beaaf355d3d3)|

### Train, Validation and Test Dataset Creation

### Automatic Hyperparameter Tuning using Optuna

### Training, Validation and Testing using Pytorch-forecasting

# Results of DeepAR and comparison Historic Average (HA) Model 

### Cluster-wise results

### Region-wise results (Tampines)
