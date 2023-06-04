

library(plotly)
library(beepr)
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

# bs_score_list = vector()
# ts_score_list = vector()
# bs_SAE = vector() # base sum absolute error
# ts_SAE = vector() 


tr_strt = 1 + 24*1
tr_stop = (5*7*24 + tr_strt - 1) #(5 weeks, 7 days, 24 hours)  

val_strt = tr_stop + 1 
val_stop = tr_stop + 7*24 # (1 week of data for validation)

ts_strt = val_stop + 1
ts_stop = val_stop + 24

# while(tr_strt < 1009)
# {
  

ts175_train = window(ts175,start=tr_strt,end=tr_stop)
ts175_val = window(ts175,start=val_strt,end=val_stop)
ts175_test = window(ts175,start=ts_strt,end=ts_stop)


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






