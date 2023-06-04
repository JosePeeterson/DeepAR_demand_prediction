
library(tscount)
# before running this run to data_preprocess_data_viz,r to load the variables and data


campyfit_nbin <- tsglm(tam_of_clstrs, model = list(past_obs = 1, past_mean = 13),
                       xreg = interventions, distr = "nbinom")




