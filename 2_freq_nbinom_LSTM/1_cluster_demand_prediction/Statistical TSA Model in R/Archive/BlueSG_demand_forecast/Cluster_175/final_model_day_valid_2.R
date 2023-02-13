#####################################################
##Find lags due to the smallest and largest ACF and PACF values.##
## reg_lags is same as past_obs
min_error=100

for(r_acf in seq(1:5))
{
  for(r_pacf in seq(1:2))
  {
    r_acf
    r_pacf
    
    no_of_reg_par = r_acf # from ACF
    n = no_of_reg_par
    
    reg_lags_acf = append( order(tr_acf)[1:n]   , order(tr_acf)[(length(tr_acf)-(n-1)) : length(tr_acf)]     )
    reg_lags_acf = append(reg_lags_acf,c(1,24,48,168)) # standard lags observed to have effect
    reg_lags_acf = unique(reg_lags_acf)
    reg_lags_acf = sort(reg_lags_acf)
    
    
    
    no_of_PACF_reg_par = r_pacf # from PACF
    p = no_of_PACF_reg_par
    
    reg_lags_pacf = append( order(tr_pacf)[1:p]   , order(tr_pacf)[(length(tr_pacf)-(p-1)) : length(tr_pacf)]     )
    reg_lags = unique( append(reg_lags_acf, reg_lags_pacf) )
    reg_lags = sort(reg_lags)
    
    
    #####################################################
    
    
    
    
    #####################################################
    ## Train the model ##
    
    ts175_train_fit = tscount::tsglm(ts=ts175_train, link='log', model =
                                       list(past_obs = reg_lags, past_mean = reg_lags),distr = 'nbinom')
    
    
    summary(ts175_train_fit)
    train_score = tscount::scoring(ts175_train_fit)
    train_score
    
    ##### plot the fitted model time series with observed time series #####
    data = data.frame(seq(1:length(ts175_train)), ts175_train, fitted.values(ts175_train_fit))
    
    fig3 <- plot_ly(data, x = ~data[1])
    for(trace in colnames(data)[2:3])
      fig3 <- fig3 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig3
    
    #####################################################
    
    
    
    
    
    #####################################################
    ##### Model Validation and number of parameters Tuning #####
    
    val_hrs = length(ts175_val)
    
    ts175_val_fit = predict( ts175_train_fit, n.ahead=val_hrs )
    
    
    val_data = data.frame(seq(1:val_hrs), ts175_val, ts175_val_fit$pred)
    
    
    
    fig1 <- plot_ly(val_data, x = ~val_data[1])
    
    for(trace in colnames(val_data)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score = tscount::scoring(ts175_val, ts175_val_fit$pred)
    val_score
    
    new_error = norm(c(val_score[4], val_score[6], val_score[7]), type ="2" )
    new_error
    
    if( new_error <= min_error)
    {
      true_r_acf = r_acf
      true_r_pacf = r_pacf
      true_reg_lags = reg_lags
      min_error = new_error
      min_error
    }
    
    
  }
  
}
r_acf
r_pacf

ts175_train_fit_TRUE = tscount::tsglm(ts=ts175_train, link='log', model =
                                   list(past_obs = true_reg_lags, past_mean = true_reg_lags),distr = 'nbinom')



#####################################################




