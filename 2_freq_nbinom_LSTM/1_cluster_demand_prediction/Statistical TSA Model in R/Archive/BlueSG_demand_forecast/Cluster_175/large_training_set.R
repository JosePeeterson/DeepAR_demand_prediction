


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

ts175 = ts( tam_of_clstrs['clstr_175'] )


window = 1

tr_strt = 1 + 24*window
tr_stop = tr_strt - 1 + 14*24

ts_strt = tr_stop + 1
ts_stop = ts_strt - 1 + 24 


ts175_train = window(ts175,start=tr_strt,end=tr_stop)
ts175_test = window(ts175,start=ts_strt,end=ts_stop)



fig = plot_ly(df,x=~seq(1,length(ts175_train)), y=~ts175_train, type='scatter', mode='lines')
fig


######## ACF plot ###### 
ts175_tr_acf = acf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_acf = data.frame(ts175_tr_acf$acf)

tr_acf = df_tr_acf$ts175_tr_acf.acf


fig1 = plot_ly(df_tr_acf, x = seq(0,(length(tr_acf)-1)), y=~tr_acf, type="scatter",mode='lines')%>%
  layout( yaxis = list(title = 'ACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig1
######## ACF plot #######


######## PACF plot ###
ts175_tr_pacf = pacf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_pacf = data.frame(ts175_tr_pacf$acf)

tr_pacf = df_tr_pacf$ts175_tr_pacf.acf


fig2 = plot_ly(df_tr_pacf, x = seq(1:length(tr_pacf)), y=~tr_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig2

######## PACF plot #######











#####################################################

SAE_vec = vector()


for(r_acf in c(10))
{
  for(r_pacf in c(5))
  {
    r_acf
    r_pacf
    
    no_of_reg_par = r_acf # from ACF
    n = no_of_reg_par
    
    reg_lags_acf = append( order(tr_acf)[1:n]   , order(tr_acf)[(length(tr_acf)-(n-1)) : length(tr_acf)]     )
    reg_lags_acf = append(reg_lags_acf,c(1,24,48)) # standard lags observed to have effect
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
                                       list(past_obs = reg_lags, past_mean = reg_lags),distr = 'poisson')
    
    
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
    ##### Model testing #####
    
    ts_hrs = length(ts175_test)
    
    ts175_fit = predict( ts175_train_fit, n.ahead=ts_hrs )
    
    
    test_df = data.frame(seq(1:ts_hrs), ts175_test, ts175_fit$pred)
    
    
    
    fig1 <- plot_ly(test_df, x = ~test_df[1])
    
    for(trace in colnames(test_df)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    test_score = tscount::scoring(ts175_test, ts175_fit$pred)
    test_score
    
    test_sum_abs_err = sum(abs(as.vector(ts175_test) - ts175_fit$pred))
    test_sum_abs_err
    
    SAE_vec = append(SAE_vec,test_sum_abs_err)
    
    
    
  }
}
#####################################################




fig4 <- plot_ly(test_df, x = ~test_df[1])

for(trace in colnames(test_df)[2:3])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines')

fig4




























