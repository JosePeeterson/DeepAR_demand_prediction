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

### Automatic Hyperparameter Tuning using Optuna

### Training, Validation and Testing using Pytorch-forecasting

# Results of DeepAR and comparison Historic Average (HA) Model 

### Cluster-wise results

### Region-wise results (Tampines)
