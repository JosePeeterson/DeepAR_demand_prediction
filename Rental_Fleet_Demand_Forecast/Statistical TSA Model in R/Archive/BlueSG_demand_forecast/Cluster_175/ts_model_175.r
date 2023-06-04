
library(plotly)


#####################################################
##Find lags due to the 10 smallest and 10 largest ACF and PACF values.##
## reg_lags is same as past_obs

col1_mon1_acf = col1_mon1
col1_mon1_acf = col1_mon1_acf[1:length(col1_mon1_acf)] 
no_of_reg_par = 10 # from ACF
n = no_of_reg_par
reg_lags_acf = append( order(col1_mon1_acf)[1:n]   , order(col1_mon1_acf)[(length(col1_mon1_acf)-(n-1)) : length(col1_mon1_acf)]     )
reg_lags_acf = sort(reg_lags_acf)


col1_mon1_pacf
no_of_PACF_reg_par = 1 # from PACF
p = no_of_PACF_reg_par
reg_lags_pacf = append( order(col1_mon1_pacf)[1:p]   , order(col1_mon1_pacf)[(length(col1_mon1_pacf)-(p-1)) : length(col1_mon1_pacf)]     )

#reg_lags_acf = append(reg_lags_acf, c(71,72,73,95,96,97,119,120,121, 143,144,145,167,168,169))

reg_lags = unique( append(reg_lags_acf, reg_lags_pacf) )
reg_lags = sort(reg_lags)

#####################################################






##########################################
## TS Modeling #######

ts_fit_175_mon1 = tscount::tsglm(ts=ts175_mon1, link='log', model =
        list(past_obs = reg_lags, past_mean = reg_lags),distr = 'nbinom')

summary(ts_fit_175_mon1)
tscount::scoring(ts_fit_175_mon1)
#plot(ts_fit_175_mon1)


data = data.frame(seq(1:768), ts175_mon1, fitted.values(ts_fit_175_mon1))

fig <- plot_ly(data, x = ~data[1])

for(trace in colnames(data)[2:3])
  fig <- fig %>% add_trace(y = as.formula(paste0("~", trace)),
                           name = trace, type='scatter', mode = 'lines')
fig

##########################################





## Prediction #######
pred_hrs = 24

ts175_mon1_fit_pred = predict( ts_fit_175_mon1, n.ahead=pred_hrs ) #newobs =ts175_mon1_pred[1:24] , level=0.9, global = TRUE

#ts175_mon1_fit_pred = predict( ts_fit_175_mon1, n.ahead=48, newobs =ts175_mon1_pred[1:24] , 
#level=0, global = TRUE,method="bootstrap",estim='ignore'  )

pred_data = data.frame(seq(1:pred_hrs), ts175_mon1_pred[1:pred_hrs], ts175_mon1_fit_pred$pred[1:pred_hrs])

#pred_data = data.frame(seq(1:48), ts175_mon1_pred[1:48], ts175_mon1_fit_pred$pred[1:48])

fig1 <- plot_ly(pred_data, x = ~pred_data[1])

for(trace in colnames(pred_data)[2:3])
  fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                           name = trace, type='scatter', mode = 'lines')
fig1

tscount::scoring(ts175_mon1_pred[1:pred_hrs], ts175_mon1_fit_pred$pred)

##########################################









