


library(plotly)

#####################################################
##### Loading the data and saving in dataframe tam_of_clstrs #####


# read time series from csv
outflow_df = read.csv(file = "outflow_clstr_dem.csv",header = TRUE)

tampines_clstrs = c(167,175,168,169,170,171,172,173,126,174)
tampines_clstrs = tampines_clstrs + 1 # +1 for R indexing, original index starts from 0.


tam_of_clstrs = data.frame(time=seq(1:2180)) # tampines outflow clusters
for(c in tampines_clstrs)
{
  new = unlist(as.vector(outflow_df[c]))
  tam_of_clstrs[ ,ncol(tam_of_clstrs) + 1] = new
  colnames(tam_of_clstrs)[ncol(tam_of_clstrs)] <- paste0("clstr_", c-1)
}

#####################################################




#####################################################
###### TRAIN TEST SPLIT #######

ts175 = ts( tam_of_clstrs['clstr_175'] )

## full train range = 1 - 2017 ### Train # 1009 give 6 weeks of training

## full test range = 2018 - 2179

# 2017
# 2041
# 2065
# 2089
# 2113
# 2137
# 2161


window = 0

tr_strt = 1009 + 24*window

tr_stop = 2017#(6*7*24 + tr_strt - 1) #(6 weeks, 7 days, 24 hours)  

val_strt1 = tr_stop - 7*24 + 1 
val_stop1 = tr_stop - 7*24 + 24

val_strt2 = tr_stop - 7*2*24 + 1 
val_stop2 = tr_stop - 7*2*24 + 24

val_strt3 = tr_stop - 7*3*24 + 1 
val_stop3 = tr_stop - 7*3*24 + 24

val_strt4 = tr_stop - 7*4*24 + 1 
val_stop4 = tr_stop - 7*4*24 + 24

val_strt5 = tr_stop - 7*5*24 + 1 
val_stop5 = tr_stop - 7*5*24 + 24

val_strt6 = tr_stop - 7*6*24 + 1 
val_stop6 = tr_stop - 7*6*24 + 24

ts_strt = tr_stop + 1 # use same day on next week for testing.
ts_stop = tr_stop + 24

ts175_train = window(ts175,start=tr_strt,end=tr_stop)

ts175_val1 = window(ts175,start=val_strt1,end=val_stop1)
ts175_val2 = window(ts175,start=val_strt2,end=val_stop2)
ts175_val3 = window(ts175,start=val_strt3,end=val_stop3)
ts175_val4 = window(ts175,start=val_strt4,end=val_stop4)
ts175_val5 = window(ts175,start=val_strt5,end=val_stop5)
ts175_val6 = window(ts175,start=val_strt6,end=val_stop6)

ts175_test = window(ts175,start=ts_strt,end=ts_stop)


base_pred = (as.vector(ts175_val1)+as.vector(ts175_val2)+as.vector(ts175_val3)+as.vector(ts175_val4)+as.vector(ts175_val5)+as.vector(ts175_val6))/6
#############################################################




#####################################################
######## ACF plot ###### 
ts175_tr_acf = acf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_acf = data.frame(ts175_tr_acf$acf)

tr_acf = df_tr_acf$ts175_tr_acf.acf


fig1 = plot_ly(df_tr_acf, x = seq(0,(length(tr_acf)-1)), y=~tr_acf, type="scatter",mode='lines')%>%
  layout( yaxis = list(title = 'ACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig1
#####################################################



#####################################################
######## PACF plot ######
ts175_tr_pacf = pacf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_pacf = data.frame(ts175_tr_pacf$acf)

tr_pacf = df_tr_pacf$ts175_tr_pacf.acf


fig2 = plot_ly(df_tr_pacf, x = seq(1:length(tr_pacf)), y=~tr_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig2
#####################################################




################### MODEL 1 ######################
##Find lags due to the smallest and largest ACF and PACF values.##
## reg_lags is same as past_obs
min_error1=100


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
    
    val_hrs1 = length(ts175_val1)
    
    ts175_val_fit1 = predict( ts175_train_fit, n.ahead=val_hrs1 )
    
    
    val_data1 = data.frame(seq(1:val_hrs1), ts175_val1, ts175_val_fit1$pred)
    
    
    
    fig1 <- plot_ly(val_data1, x = ~val_data1[1])
    
    for(trace in colnames(val_data1)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score1 = tscount::scoring(ts175_val1, ts175_val_fit1$pred)
    val_score1
    
    new_error1 = norm(c(val_score1[4], val_score1[6], val_score1[7]), type ="2" )
    new_error1
    
    if( new_error1 <= min_error1)
    {
      true_r_acf1 = r_acf
      true_r_pacf1 = r_pacf
      true_reg_lags1 = reg_lags
      min_error1 = new_error1
      min_error1
      ts175_train_fit_TRUE1 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf1
true_r_acf1
true_reg_lags1

################# MODEL 1 END ##########################
########################################################




############## MODEL 2 #############################

min_error2=100

for(r_acf in seq(1:5))
{
  for(r_pacf in  seq(1:2))
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
    
    val_hrs2 = length(ts175_val2)
    
    ts175_val_fit2 = predict( ts175_train_fit, n.ahead=val_hrs2 )
    
    
    val_data2 = data.frame(seq(1:val_hrs2), ts175_val2, ts175_val_fit2$pred)
    
    
    
    fig1 <- plot_ly(val_data2, x = ~val_data2[1])
    
    for(trace in colnames(val_data2)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score2 = tscount::scoring(ts175_val2, ts175_val_fit2$pred)
    val_score2
    
    new_error2 = norm(c(val_score2[4], val_score2[6], val_score2[7]), type ="2" )
    new_error2
    
    if( new_error2 <= min_error2)
    {
      true_r_acf2 = r_acf
      true_r_pacf2 = r_pacf
      true_reg_lags2 = reg_lags
      min_error2 = new_error2
      min_error2
      ts175_train_fit_TRUE2 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf2
true_r_acf2
true_reg_lags2



################# model 2 end ####################################
##################################################################



############## MODEL 3 #############################

min_error3=100

for(r_acf in seq(1:5))
{
  for(r_pacf in  seq(1:2))
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
    
    val_hrs3 = length(ts175_val3)
    
    ts175_val_fit3 = predict( ts175_train_fit, n.ahead=val_hrs3 )
    
    
    val_data3 = data.frame(seq(1:val_hrs3), ts175_val3, ts175_val_fit3$pred)
    
    
    
    fig1 <- plot_ly(val_data3, x = ~val_data3[1])
    
    for(trace in colnames(val_data3)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score3 = tscount::scoring(ts175_val3, ts175_val_fit3$pred)
    val_score3
    
    new_error3 = norm(c(val_score3[4], val_score3[6], val_score3[7]), type ="2" )
    new_error3
    
    if( new_error3 <= min_error3)
    {
      true_r_acf3 = r_acf
      true_r_pacf3 = r_pacf
      true_reg_lags3 = reg_lags
      min_error3 = new_error3
      min_error3
      ts175_train_fit_TRUE3 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf3
true_r_acf3
true_reg_lags3

################# MODEL 3 END ##########################
########################################################



############## MODEL 4 #############################

min_error4=100

for(r_acf in  seq(1:5))
{
  for(r_pacf in  seq(1:2))
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
    
    val_hrs4 = length(ts175_val4)
    
    ts175_val_fit4 = predict( ts175_train_fit, n.ahead=val_hrs4 )
    
    
    val_data4 = data.frame(seq(1:val_hrs4), ts175_val4, ts175_val_fit4$pred)
    
    
    
    fig1 <- plot_ly(val_data4, x = ~val_data4[1])
    
    for(trace in colnames(val_data4)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score4 = tscount::scoring(ts175_val4, ts175_val_fit4$pred)
    val_score4
    
    new_error4 = norm(c(val_score4[4], val_score4[6], val_score4[7]), type ="2" )
    new_error4
    
    if( new_error4 <= min_error4)
    {
      true_r_acf4 = r_acf
      true_r_pacf4 = r_pacf
      true_reg_lags4 = reg_lags
      min_error4 = new_error4
      min_error4
      ts175_train_fit_TRUE4 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf4
true_r_acf4
true_reg_lags4


################# MODEL 4 END ##########################
########################################################





############## MODEL 5 #############################

min_error5=100

for(r_acf in  seq(1:5))
{
  for(r_pacf in  seq(1:2))
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
    
    val_hrs5 = length(ts175_val5)
    
    ts175_val_fit5 = predict( ts175_train_fit, n.ahead=val_hrs5 )
    
    
    val_data5 = data.frame(seq(1:val_hrs5), ts175_val5, ts175_val_fit5$pred)
    
    
    
    fig1 <- plot_ly(val_data5, x = ~val_data5[1])
    
    for(trace in colnames(val_data5)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score5 = tscount::scoring(ts175_val5, ts175_val_fit5$pred)
    val_score5
    
    new_error5 = norm(c(val_score5[4], val_score5[6], val_score5[7]), type ="2" )
    new_error5
    
    if( new_error5 <= min_error5)
    {
      true_r_acf5 = r_acf
      true_r_pacf5 = r_pacf
      true_reg_lags5 = reg_lags
      min_error5 = new_error5
      min_error5
      ts175_train_fit_TRUE5 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf5
true_r_acf5
true_reg_lags5

################# MODEL 5 END ##########################
########################################################






############## MODEL 6 #############################

min_error6=100

for(r_acf in  seq(1:5))
{
  for(r_pacf in  seq(1:2))
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
    
    val_hrs6 = length(ts175_val6)
    
    ts175_val_fit6 = predict( ts175_train_fit, n.ahead=val_hrs6 )
    
    
    val_data6 = data.frame(seq(1:val_hrs6), ts175_val6, ts175_val_fit6$pred)
    
    
    
    fig1 <- plot_ly(val_data6, x = ~val_data6[1])
    
    for(trace in colnames(val_data6)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score6 = tscount::scoring(ts175_val6, ts175_val_fit6$pred)
    val_score6
    
    new_error6 = norm(c(val_score6[4], val_score6[6], val_score6[7]), type ="2" )
    new_error6
    
    if( new_error6 <= min_error6)
    {
      true_r_acf6 = r_acf
      true_r_pacf6 = r_pacf
      true_reg_lags6 = reg_lags
      min_error6 = new_error6
      min_error6
      ts175_train_fit_TRUE6 = ts175_train_fit
    }
    
    
  }
  
}
true_r_pacf6
true_r_acf6
true_reg_lags6

################# MODEL 6 END ##########################
########################################################



#####################################################
##### Model testing and comparison with baseline (previously observed value) #####

test_hrs = length(ts175_test)

ts175_test_fit1 = predict( ts175_train_fit_TRUE1, n.ahead=test_hrs)
ts175_test_fit2 = predict( ts175_train_fit_TRUE2, n.ahead=test_hrs)
ts175_test_fit3 = predict( ts175_train_fit_TRUE3, n.ahead=test_hrs)
ts175_test_fit4 = predict( ts175_train_fit_TRUE4, n.ahead=test_hrs)
ts175_test_fit5 = predict( ts175_train_fit_TRUE5, n.ahead=test_hrs)
ts175_test_fit6 = predict( ts175_train_fit_TRUE6, n.ahead=test_hrs)

base_score = tscount::scoring(ts175_test, base_pred)
base_score
#bs_score_list = append(bs_score_list,base_score)

## AVERAGE POOLING ########
ts175_test_fit =  (ts175_test_fit1$pred + ts175_test_fit2$pred + ts175_test_fit3$pred + ts175_test_fit4$pred + ts175_test_fit5$pred + ts175_test_fit6$pred)/6
###########################

######## HISTOGRAM ########
# ts175_all_test =  data.frame(ts175_test_fit1$pred, ts175_test_fit2$pred ,ts175_test_fit3$pred , ts175_test_fit4$pred , ts175_test_fit5$pred , ts175_test_fit6$pred)
# rows = nrow(ts175_all_test)
# ts175_test_fit = vector()
# for (r in seq(1:rows))
# {
#   y = hist(as.numeric(ts175_all_test[r,]))
#   
#   mx = which.max(y$density)
#   
#   #mx = max(which(y$density == max(y$density)))
#   
#   pred = (y$breaks[mx] + y$breaks[mx+1])/2
#   
#   ts175_test_fit = append(ts175_test_fit, pred)
# }
###########################

test_score = tscount::scoring(ts175_test, ts175_test_fit)
test_score

base_sum_abs_err = sum(abs(as.vector(ts175_test) - base_pred))
base_sum_abs_err
#bs_SAE = append(bs_SAE, base_sum_abs_err)

test_sum_abs_err = sum(abs(as.vector(ts175_test) - ts175_test_fit))
test_sum_abs_err
#####################################################




#####################################################
###### plot test result ##########

test_data = data.frame(seq(1:test_hrs), ts175_test,base_pred, ts175_test_fit)

#text_vec = c(as.character(base_sum_abs_err),as.character(base_score), as.character(test_sum_abs_err),as.character(test_score) )
text_vec = vector()
text_vec = append(text_vec,'base' )
text_vec = append(text_vec,base_sum_abs_err )
text_vec = append(text_vec,base_score[7] )
text_vec = append(text_vec,'test' )
text_vec = append(text_vec,test_sum_abs_err )
text_vec = append(text_vec,test_score[7]  )
text_vec = paste(text_vec,collapse=" ")



fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:4])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title=text_vec)

fig4
#####################################################



























