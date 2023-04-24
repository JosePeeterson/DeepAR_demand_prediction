
def train_and_forecast(neurons,layers,batch_size,learning_rate,dropout,encoder_length,max_epoch,region,full_train_data,val_data,test_data):

    import os,sys
    os.chdir("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM/1_cluster_demand_prediction/model/standalone")
    sys.path.append(os.path.abspath(os.path.join("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM/1_cluster_demand_prediction/model/standalone")))

    import json    
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from pytorch_forecasting.data.encoders import TorchNormalizer
    from pytorch_forecasting.metrics import SMAPE, RMSE
    from torchmetrics import R2Score, SymmetricMeanAbsolutePercentageError, MeanSquaredError
    import matplotlib.pyplot as plt
    import pandas as pd
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error,confusion_matrix,ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    import seaborn as sn

    """
    Set Random seed
    """
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    ## additional seeding to ensure reproduciblility.
    pl.seed_everything(0)

    Target = 'target'

    print(list(full_train_data.columns))

    """
    set inputs here
    (hyperparameters grid search)

    """
    ######### Network Architecture definition ###################

    ###### Create hyperparameters grid ###### 
    pred_len = 1
    hparams_grid = {"LSTM_neuron_size":[neurons],
                    "num_layers":[layers],
                    "batch_size":[batch_size],
                    "learning_rate":[learning_rate],
                    "max_encoder_length":[encoder_length],
                    "max_prediction_length":[pred_len],
                    "dropout":[dropout],
                    #"cov_pair":cov_pairs_list,# [cov_pairs_list[7]],
                    "Num_epochs":[max_epoch]}#[18,20,22,24,26,28,30]}
                    #"Num_epochs":[16,18,20,22,24,26,28]}
    ###### Create hyperparameters grid ###### 

    p = 6 # patience no. of epochs
    Loss=NegativeBinomialDistributionLoss()
    ######### Network Architecture definition ###################
    

    ######### Training Routine ###################
    fdv_steps = 10 # fast_dev_run
    ######### Training Routine ###################


    ############## Inputs for 2) Persistance model ( seasonal naive forecast ) #######################
    season_len = 168 # length of season
    num_past_seas = 2 # number of past seasons to use in averaging
    #seas_pred_strt_idx = 2035 # seasonal naive forecast start index, in hours use the df dataframe
    ############## Inputs for 2) Persistance model ( seasonal naive forecast ) #######################


    param_comb_cnt=0
    for neu,lay,bat,lr,enc_len,pred_len,drop,num_ep in product(*[x for x in hparams_grid.values()]):
        print(param_comb_cnt,neu,lay,bat,lr,enc_len,pred_len,drop,num_ep)
        param_comb_cnt+=1


    """
    Full Training Routine 
    with hyperparmeter grid search

    Load data into TimeSeriesDataSet object

    for fast development run
    uncomment fast_dev_run = fdv_steps

    """
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-8, patience=p, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    RMSE_list = [] # FIND minimum RMSE case
    hyperparams_list = [] # FIND minimum RMSE case

    num_cols_list = []

    cat_dict = {"_hour_of_day": NaNLabelEncoder(add_nan=True).fit(full_train_data._hour_of_day), \
    "_day_of_week": NaNLabelEncoder(add_nan=True).fit(full_train_data._day_of_week), "_day_of_month" : NaNLabelEncoder(add_nan=True).fit(full_train_data._day_of_month), "_day_of_year" : NaNLabelEncoder(add_nan=True).fit(full_train_data._day_of_year), \
        "_week_of_year": NaNLabelEncoder(add_nan=True).fit(full_train_data._week_of_year), "_month_of_year": NaNLabelEncoder(add_nan=True).fit(full_train_data._month_of_year) ,"_year": NaNLabelEncoder(add_nan=True).fit(full_train_data._year) }
    cat_list = ["_hour_of_day","_day_of_week","_day_of_month","_day_of_year","_week_of_year","_month_of_year","_year"]  

    num_cols_list = list(full_train_data.columns[3:-7]) 

    # param_comb_cnt=-1
    for neu,lay,bat,lr,enc_len,pred_len,drop,num_ep in product(*[x for x in hparams_grid.values()]):
            
        ######### Load DATA #############
        full_train_dataset = TimeSeriesDataSet(
            full_train_data,
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

        val_dataset = TimeSeriesDataSet.from_dataset(full_train_dataset,val_data, stop_randomization=True, predict=False)
        test_dataset = TimeSeriesDataSet.from_dataset(full_train_dataset,test_data, stop_randomization=True)

        full_train_dataloader = full_train_dataset.to_dataloader(train=True, )
        val_dataloader = val_dataset.to_dataloader(train=False,  )
        test_dataloader = test_dataset.to_dataloader(train=False, )
        ######### Load DATA #############


        """
        Machine Learning predictions START
        1) DeepAR

        """
        trainer = pl.Trainer(
            max_epochs=num_ep,
            gpus=-1, #-1
            auto_lr_find=False,
            gradient_clip_val=0.1,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            #fast_dev_run=fdv_steps,
            logger=True,
            #log_every_n_steps=10,
            # profiler=True,
            callbacks=[lr_logger]#, early_stop_callback],
            #enable_checkpointing=True,
            #default_root_dir="C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\2_freq_nbinom_LSTM\1_cluster_demand_prediction\logs"
        )


        #print(f"training routing:\n \n {trainer}")
        deepar = DeepAR.from_dataset(
            full_train_dataset,
            learning_rate=lr,
            hidden_size=neu,
            rnn_layers=lay,
            dropout=drop,
            loss=Loss,
            log_interval=20,
            log_val_interval=1.0,
            log_gradient_flow=False,
            # reduce_on_plateau_patience=3,
        )

            
        #print(f"Number of parameters in network: {deepar.size()/1e3:.1f}k")
        # print(f"Model :\n \n {deepar}")
        #torch.set_num_threads(10)
        trainer.fit(
            deepar,
            train_dataloaders=full_train_dataloader,
            val_dataloaders=val_dataloader,
        )

        """ 
        ########## DEEPAR Prediction #####################
        """
        ########## DEEPAR Prediction #####################
        test_output = deepar.predict(data=test_dataloader,mode='prediction',return_index=True,num_workers=8,show_progress_bar=True)

        ## test_output data structure ##
        ## test_output = [ [x,x,x,x,x,...], [ {'time_idx': [x,x,x,x,x,...] , 'group': [x,x,x,x,x,...]} ]  ]
        ## len(test_output[0]) includes all the groups combined

        actual = {}
        prediction = {}

        for i in range(len(test_output[0])):
            time_idx_i = test_output[1]['time_idx'][i]
            grp_i = test_output[1]['group'][i]

            actual[(time_idx_i,grp_i)] = test_data[(test_data['time_idx'] == time_idx_i) & (test_data['group'] == grp_i)]['target'].values[0]
            prediction[(time_idx_i,grp_i)] = test_output[0][i]

        groups_list = np.unique(test_output[1]['group'])
        time_idx_list = np.unique(test_output[1]['time_idx'])

        deepar_RMSE_grp_dict = {}
        deepar_actuals_grp_dict = {}
        deepar_predictions_grp_dict = {}

        for grp in groups_list:
            err_list = np.array([])
            actuals_grp_list = []
            predictions_grp_list = []
            for t in time_idx_list:
                err_list = np.append(err_list,(actual[(t,grp)] - prediction[(t,grp)] )**2)
                actuals_grp_list.append( int(actual[(t,grp)] ))
                predictions_grp_list.append(int(prediction[(t,grp)]))

            deepar_actuals_grp_dict[str(grp)] = actuals_grp_list
            deepar_predictions_grp_dict[str(grp)] = predictions_grp_list

            plt.figure(figsize=(25,5))
            plt.title('line plot, cluster: '+str(grp))  
            plt.plot(actuals_grp_list,'^-')
            plt.plot(predictions_grp_list,'*-')
            plt.show()

            plt.xlabel('actual')
            plt.ylabel('prediction')
            plt.title('scatter plot, cluster: '+str(grp))       
            plt.scatter(actuals_grp_list,predictions_grp_list )
            plt.show()

            cm = confusion_matrix(actuals_grp_list,predictions_grp_list)
            max_classes = max(len(np.unique(actuals_grp_list)),len(np.unique(predictions_grp_list)))
            df_cm = pd.DataFrame(cm, index = [i for i in range(max_classes)],
                            columns = [i for i in range(max_classes)])
            plt.figure(figsize = (10,7))      
            s = sn.heatmap(df_cm, annot=True, )
            s.set(xlabel='Predicted-Label', ylabel='True-Label',title='cluster: '+str(grp))

            deepar_RMSE_grp_dict[str(grp)] = np.sqrt(np.mean(err_list))
            print(f'RMSE from cluster: {grp} = {deepar_RMSE_grp_dict[str(grp)]}')

        total_actual_list = []
        total_prediction_list = []
        Total_error_list = np.array([])
        for k in actual.keys():
            Total_error_list = np.append(Total_error_list,(actual[(k[0],k[1])] - prediction[(k[0],k[1])] )**2)
            total_actual_list.append(int(actual[(k[0],k[1])]))
            total_prediction_list.append(int(prediction[(k[0],k[1])] ))

        cm = confusion_matrix(total_actual_list,total_prediction_list)
        max_classes = max(len(np.unique(total_actual_list)),len(np.unique(total_prediction_list)))
        df_cm = pd.DataFrame(cm, index = [i for i in range(max_classes)],
                        columns = [i for i in range(max_classes)])
        plt.figure(figsize = (10,7))      
        s = sn.heatmap(df_cm, annot=True, )
        s.set(xlabel='Predicted-Label', ylabel='True-Label',title='Region Total: '+str(region))

        Total_rmse = np.sqrt(np.mean(Total_error_list))
        print('Total RMSE from all clusters: ',Total_rmse)

        print('\n Hyperparameters: neu,lay,bat,lr,enc_len,pred_len,drop,\n')
        print(neu,lay,bat,lr,enc_len,pred_len,drop,' \n')

        ########## DEEPAR Prediction #####################

        """ 
        ########## HISTORIC AVERAGE Prediction #####################
        """
        ########## HA Prediction #####################
        #TODO: merge full_train_data and test_data to do historic average predictions


        ########## HA Prediction #####################






    ############### Saving Results and optimal hyperparameters ########################
    """
    SAVE all results and hparams to json files for dashboard visualization
    
    """

    results_dict = {"actuals_dem":deepar_actuals_grp_dict,"all_clstrs_deepar_pred":deepar_predictions_grp_dict}#, "all_clstrs_hist_avg":all_clstrs_hist_avg_pred}
    results_dict["best_deepar_hyperparams"] = {"neu":neurons,"lay":layers,"bat":batch_size,"lr":learning_rate,"drop":dropout,"enc_len":encoder_length}
    results_dict["RMSE_DAR"] = deepar_RMSE_grp_dict
    # results_dict["HA_RMSE"] = RMSE_list_HA_pred

    os.chdir("/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/2_freq_nbinom_LSTM/1_cluster_demand_prediction/data/results_data")

    # Serializing json
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()
    json_object = json.dumps(results_dict, indent=4, default=np_encoder)
    # Writing to .json
    with open(region+".json", "w") as outfile:
        outfile.write(json_object)
    ############### Saving Results and optimal hyperparameters ########################



        """
        Machine Learning predictions END
        """

    return

