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

| DeepAR Encoder Network| DeepAR Decoder Network|
| --------------------- | --------------------- |
| ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d57479a0-5cee-4f90-8f26-07c9f5f647ad) | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d9a31f75-12fe-4264-9f6b-beaaf355d3d3) |

The encoder and decoder networks are identical. The decoder network proceeds the encoder network. The inputs to the networks are the target, z and its covarites, x . Grey rectangles with h represents the cells states of the LSTM cells of the network. l represents the likelihood distribution given the parameters of the distribution output from the LSTM cells as shown by equation x below. z represents the mode of the likelihood distribution. During training, the target’s ground truth is known and it is compared with z  which is also the next element of the sequence to calculate the loss for both the encoder and decoder networks. In the encoder network z.

During training, some combination of inputs in the encoder will be correlated to the output in the decoder. The weights are trained to identify this combination. The network outputs the mean, μ and the shape, α parameters.  θ(μ,α). The r and p parameters of the negative binomial distribution are calculated using the formulas: r=1/α , p=μ/(μ+r).

The output must be the mode because it minimizes the negative log-likelihood distribution that is used as the loss function during training.

The log likelihood is further aggregated over all the targets, i of multivariate forecasting. 

![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d6a93427-0b82-4153-9808-ebd81175869d)

Stochastic gradient descent with Adam optimizer is used to learn the weights, Θ of the LSTM network.

### Train, Validation and Test Dataset Creation

The response (target) time series, numerical covariates and categorical covariates are concatenated into a single dataset data frame.

Date information time series such as hour-of-the-day, day-of-the-week, day-of-the-month, day-of-the-year, week-of-the-year, month-of-the-year and year are concatenated to the single dataset data frame.

A 70-20-10 split is applied to train-validation-test split of the entire dataset

For the Tampines cluster, all the covariate time series include 'wea_desc_clstr_175', 'temp_clstr_175', 'hum_clstr_175', 'clstr_171', 'clstr_168', 'clstr_175_lag_502', 'clstr_175_lag_166', 'clstr_175_lag_334', '175_inflow', '175_lag_502_inflow', '175_lag_166_inflow' and '175_lag_334_inflow'.

However, only two covariates from the following list are used as covariates at any one training run. This list includes 'wea_desc_clstr_175', 'temp_clstr_175', 'hum_clstr_175', 'clstr_171', 'clstr_168'. This is feature selection. It is a hyperparameter. If all the features are used in one go, it can lead to longer times for learning and the model may have a higher bias due to underfit. This can lead to poor convergence due to not all the features having a high correlation with the targe and ultimately this will lead to poor results. 

Examples of covariate pairs list include:
1.	[('wea_desc_clstr_175', 'temp_clstr_175'),
2.	('wea_desc_clstr_175', 'hum_clstr_175'),
3.	('wea_desc_clstr_175', 'clstr_171'),
4.	('wea_desc_clstr_175', 'clstr_168'),
5.	('temp_clstr_175', 'hum_clstr_175'),
6.	('temp_clstr_175', 'clstr_171'),
7.	('temp_clstr_175', 'clstr_168'),
8.	('hum_clstr_175', 'clstr_171'),
9.	('hum_clstr_175', 'clstr_168'),
10.	('clstr_171', 'clstr_168')]

Finally, all the possible columns used for training include the following: 'time_idx', 'group', 'clstr_175', 'wea_desc_clstr_175', 'temp_clstr_175', 'hum_clstr_175', 'clstr_171', 'clstr_168', 'clstr_175_lag_502', 'clstr_175_lag_166', 'clstr_175_lag_334','175_inflow', 'date', '_hour_of_day', '_day_of_week', '_day_of_month', '_day_of_year', '_week_of_year', '_month_of_year' and '_year'.

### Automatic Hyperparameter Tuning using Optuna

Optuna is a hyperparameter optimization framework with visualizations for interpretability. It can efficiently search large spaces and prune unpromising trials for faster results compared to grid search.

The range of all the hyperparameters is specified together with the RMSE of the validation dataset as the objective function to minimize. Optuna finds the optimal hyperparameter combination an intelligent way without trying out all the possible combination as in grid search.

### Training, Validation and Testing using Pytorch-forecasting

After splitting the dataset into train, validation, and test sets, the optimal hyperparameter is selected from the results of Optuna below and the training routine is initiated. 

The categorical variables are first one-hot-encoded and goes through an embedding layer to reduce dimensionality. 

The ‘train_dataset’ object is created specifying the different columns of the data frame like the target, numerical and categorical covariates.  

Three data loaders called are created. train_dataloader,  val_dataloader, and  test_dataloader are created to combines a dataset and a sampler to provides an iterable over the given dataset.

A ‘trainer’ object is initialized specifying the details of the training routine like maximum number of epochs, use of GPUs, gradient clipping and so on.

DeepAR neural network architecture is defined through the ‘deepar’ object. Specifications like training dataset, learning rate, number of neurons to use in the LSTM cells, the number of LSTM layers, dropout and the negative binomial likelihood loss are specified. 
•	‘trainer.fit’ is applied to fit the model to training data and validate on the validation set.
•	Prediction is made on the test set using ‘deepar.predict()’
•	Prediction is made one time step at a time as this gives the greatest accuracy. The MAE and RMSE for the entire test set is calculated.  The plot of ground-truth versus DeepAR predictions are also created.   

# Results of DeepAR and comparison with Historic Average (HA) Model 

### Region-wise results

| Region    | DeepAR MAE | HA MSE | DeepAR RMSE | HA RMSE | DeepAR Prediction graph | HA prediction graph |
| --------  | ---------- |------- |------------ | ------ |------------------------ |-------------------- |
| Tampines  | 1.19       | 1.63   | 1.79        | 2.13   |![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/ee9ccf71-8c6f-41e4-8ae9-c372731b3fc7) | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d67aab0b-57af-41be-b81d-9acb032e47a7)|
| Central   | -          | -      | 0.97        | 1.71   | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/6b508931-2f81-409d-a7d0-ad9d38b9cfae)| ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/9541c400-58ff-4052-ba12-f947bbbaa51b)|
| Woodlands | -          | -      | 1.17        | 1.43   | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/91d9e770-3fd1-4d55-9581-212380aaab59)|![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/d4d7eafa-1303-4e4a-81ae-47d6ed8dfd21)|

### Cluster-wise results (Tampines Region)

| Cluster   | DeepAR RMSE | HA RMSE | DeepAR Prediction graph | HA prediction graph |
| --------  |------------ | ------ |------------------------ |-------------------- |
| 126       | 1.01        | 1.55   |![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/63a453ca-4b13-41b4-8b32-c8ec02371e53)| ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/62c150d7-145e-42b8-8d86-515a6ad261c1) |
| 166       | 0.65        | 0.89   | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/c277add9-f267-45d5-84b8-5ac5f599c7bb) | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/040f4d78-ebde-44bc-9274-ff03111372d0) |
| 175       | 1.58        | 2.34   | ![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/54ccd970-8cdf-44a7-af67-6d5d54dc32b4) |![image](https://github.com/JosePeeterson/DeepAR_demand_prediction/assets/76463517/13317c77-4ebf-4305-8565-6f9d1858c7c2) |
