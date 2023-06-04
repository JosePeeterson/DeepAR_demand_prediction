library(plotly)
library(fUnitRoots)
library(lubridate)
library(forecast)
library(caret)

#rm(list = ls())


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
##### train test split #####

tam_175 = ts(tam_of_clstrs['clstr_175'],start=c(1,1), frequency = 168)

## full train range = 1 - 2017
tr_strt = 1345     ### Train # 1345 give 4 weeks of training
tr_stop = 2161


## full test range = 2018 - 2179
ts_strt = 2162  ### Test
ts_stop = 2179

# 2017
# 2041
# 2065
# 2089
# 2113
# 2137
# 2161

# tot_hrs = 24*(as.Date("1-12-22 00:00:00")-as.Date("1-9-24 00:00:00")) + 44
# ts175 = ts(tam_175$clstr_175 ,start=c(1,1) , end=c(1,tot_hrs),frequency=24*365)


ts175_train = window(tam_175,start=c(1,tr_strt),end=c(1,tr_stop))

ts175_test = window(tam_175,start=c(1,ts_strt),end=c(1,ts_stop))

######## test plot
# test_hrs = length(ts175_test)
# test_data = data.frame(seq(1:test_hrs), ts175_test)
# names(test_data)[1] <- "time"
# fig <- plot_ly(test_data, type = 'scatter', mode = 'lines',x=~time,y=~clstr_175 )%>%
#   add_trace(name = 'test')
#
# fig

# ######## train plot
# tr_hrs = length(ts175_train)
# tr_data = data.frame(seq(1:tr_hrs), ts175_train)
# names(tr_data)[1] <- "time"
# fig <- plot_ly(tr_data, type = 'scatter', mode = 'lines',x=~time,y=~clstr_175 )%>%
#   add_trace(name = 'train')
#
# fig
#####################################################




#####################################################
######## ACF plot ######
ts175_tr_acf = acf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_acf1 = data.frame(ts175_tr_acf$acf)

tr_acf = df_tr_acf1$ts175_tr_acf.acf


fig1 = plot_ly(df_tr_acf1, x = seq(0,(length(tr_acf)-1)), y=~tr_acf, type="scatter",mode='lines')%>%
  layout( yaxis = list(title = 'ACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=1.96/(sqrt(length(ts175_train))))%>%
  add_trace(y=-1.96/(sqrt(length(ts175_train))))%>%
  layout(title='ACF_175_train')
fig1
#####################################################



#####################################################
#### ACF after Seasonal difference ###

seas_diff = 168
ts175_train_diff_168 = diff(ts175_train,seas_diff) # length = 1849

ts175_tr_acf = acf(ts175_train_diff_168,pl=FALSE,lag.max=length(ts175_train_diff_168))

df_tr_acf1 = data.frame(ts175_tr_acf$acf)

tr_acf = df_tr_acf1$ts175_tr_acf.acf


fig1 = plot_ly(df_tr_acf1, x = seq(0,(length(tr_acf)-1)), y=~tr_acf, type="scatter",mode='lines')%>%
  layout( yaxis = list(title = 'ACF_ts175_train_diff_168'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=1.96/(sqrt(length(ts175_train_diff_168))))%>%
  add_trace(y=-1.96/(sqrt(length(ts175_train_diff_168))))%>%
  layout(title='ACF_ts175_train_diff_168')
fig1

#####################################################



#####################################################
######## PACF plot ###
ts175_tr_pacf = pacf(ts175_train,pl=FALSE,lag.max=length(ts175_train))

df_tr_pacf = data.frame(ts175_tr_pacf$acf)

tr_pacf = df_tr_pacf$ts175_tr_pacf.acf


fig2 = plot_ly(df_tr_pacf, x = seq(1:length(tr_pacf)), y=~tr_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_175_train'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=1.96/(sqrt(length(ts175_train))))%>%
  add_trace(y=-1.96/(sqrt(length(ts175_train))))%>%
  layout(title='PACF_175_train')
fig2

#####################################################



#####################################################
######## PACF plot after Seasonal difference  ###
ts175_tr_pacf = pacf(ts175_train_diff_168,pl=FALSE,lag.max=length(ts175_train_diff_168))

df_tr_pacf = data.frame(ts175_tr_pacf$acf)

tr_pacf = df_tr_pacf$ts175_tr_pacf.acf


fig2 = plot_ly(df_tr_pacf, x = seq(1:length(tr_pacf)), y=~tr_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_ts175_train_diff_168'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=1.96/(sqrt(length(ts175_train_diff_168))))%>%
  add_trace(y=-1.96/(sqrt(length(ts175_train_diff_168))))%>%
  layout(title='PACF_ts175_train_diff_168')
fig2

#####################################################


#####################################################
######## Fit and Prediction on differenced time series  ###

horizon = ts_stop - ts_strt + 1

autofit = auto.arima(ts175_train_diff_168, trace=TRUE)

autofit

ts_autofit = predict(autofit,n.ahead = horizon)

plot(forecast(autofit,h=horizon))

# futurVal <- forecast::Arima(autofit,h=horizon, level=c(95.0))
# plot.forecast(futurVal)

#####################################################




#####################################################
######## Results and Analysis  ###
test_sum_abs_err = sum(abs(as.vector(ts175_test) - floor(0.5 + ts_autofit$pred)))
test_sum_abs_err

test_score = tscount::scoring(ts175_test, ts_autofit$pred)
test_score


text_vec = vector()
text_vec = append(text_vec,'test' )
text_vec = append(text_vec,test_sum_abs_err )
text_vec = append(text_vec,test_score[7]  )
text_vec = paste(text_vec,collapse=" ")

test_hrs = length(ts175_test)

test_data = data.frame(seq(1:test_hrs), ts175_test, ts_autofit)

fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:3])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title=text_vec)


fig4

#####################################################



#####################################################
##### Inverse transformation to obtain predictions on original time series ###

diff_lag = seas_diff

strt_idx = length(ts175_train) - diff_lag

final_ts_autofit = c() # final_ts_autofit

for(i in c(1:length(ts_autofit$pred)))
{

  final_ts_autofit = append( final_ts_autofit,(ts175_train[strt_idx + i] + ts_autofit$pred[i]),after=i)

}

final_ts_autofit = floor(0.5+final_ts_autofit) # round it to integer

test_hrs = length(ts175_test)

test_data = data.frame(seq(1:test_hrs), ts175_test, final_ts_autofit)

fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:3])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title='final prediction')


fig4

#####################################################


#######################################################
##### Error metrics ######
print("MAE and RMSE")
MAE(final_ts_autofit,as.numeric(ts175_test))
RMSE(final_ts_autofit,as.numeric(ts175_test))




#######################################################

