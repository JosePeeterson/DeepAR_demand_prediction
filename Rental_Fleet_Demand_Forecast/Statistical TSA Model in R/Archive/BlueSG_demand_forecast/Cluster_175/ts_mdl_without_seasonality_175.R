


library(plotly)

#####################################################
##Find lags due to the 10 smallest and 10 largest ACF and PACF values.##
## reg_lags is same as past_obs


ts175 = ts( tam_of_clstrs['clstr_175'] )
ts175_mon1 = window(ts175,start=1,end=768)
ts175_mon1_pred = window(ts175,start=769,end=936)


ts175_mon1_rm_12 = lag(ts175_mon1,-12)
ts175_mon1_diff = ts175_mon1 - ts175_mon1_rm_12
ts175_mon1_diff = ts175_mon1_diff + abs(min(ts175_mon1_diff ))


ts175_mon1_rm_12_pred = lag(ts175_mon1_pred,-12)
ts175_mon1_diff_pred = ts175_mon1_pred - ts175_mon1_rm_12_pred
ts175_mon1_diff_pred = ts175_mon1_diff_pred + abs(min(ts175_mon1_diff_pred ))

#####################################################




######## ACF plot ###
ts175_mon1_acf = acf(ts175_mon1_diff,pl=FALSE,lag.max=760)
ts175_mon1_pred_acf = acf(ts175_mon1_diff_pred,pl=FALSE,lag.max=168)

df_mon1_acf = data.frame(ts175_mon1_acf$acf)
df_mon1_pred_acf = data.frame(ts175_mon1_pred_acf$acf)
# df = list(df_mon1_acf,df_mon1_pred_acf)

col1_mon1 = df_mon1_acf$ts175_mon1_acf.acf
col1_mon1_pred = df_mon1_pred_acf$ts175_mon1_pred_acf.acf
# col = list(col1_mon1,col1_mon1_pred)

# fig = vector()
# subplot_figs = list()

fig1 = plot_ly(df_mon1_acf, x = seq(1:length(col1_mon1)), y=~col1_mon1, type="scatter")%>%
  layout( yaxis = list(title = 'ACF_175_mon1'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig1

fig2 = plot_ly(df_mon1_pred_acf, x = seq(1:length(col1_mon1_pred)), y=~col1_mon1_pred, type="scatter", mode= 'lines')%>%
  layout( yaxis = list(title = 'ACF_175_mon1_pred'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig2
######## ACF plot ###



######## PACF plot ###
ts175_mon1_pacf = pacf(ts175_mon1,pl=FALSE,lag.max=760)
ts175_mon1_pred_pacf = pacf(ts175_mon1_pred,pl=FALSE,lag.max=168)

df_mon1_pacf = data.frame(ts175_mon1_pacf$acf)
df_mon1_pred_pacf = data.frame(ts175_mon1_pred_pacf$acf)
df = list(df_mon1_acf,df_mon1_pred_acf)

col1_mon1_pacf = df_mon1_pacf$ts175_mon1_pacf.acf
col1_mon1_pred_pacf = df_mon1_pred_pacf$ts175_mon1_pred_pacf.acf
col = list(col1_mon1,col1_mon1_pred)


fig1 = plot_ly(df_mon1_pacf, x = seq(1:length(col1_mon1_pacf)), y=~col1_mon1_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_175_mon1'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig1

fig2 = plot_ly(df_mon1_pred_pacf, x = seq(1:length(col1_mon1_pred_pacf)), y=~col1_mon1_pred_pacf, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'PACF_175_mon1_pred'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig2
######## PACF plot ###






















#####################################################
##Find lags due to the 10 smallest and 10 largest ACF and PACF values.##
## reg_lags is same as past_obs

col1_mon1_acf = col1_mon1
no_of_reg_par = 10 # from ACF
n = no_of_reg_par
reg_lags_acf = append( order(col1_mon1_acf)[1:n]   , order(col1_mon1_acf)[(length(col1_mon1_acf)-(n-1)) : length(col1_mon1_acf)]     )

col1_mon1_pacf
no_of_PACF_reg_par = 5 # from PACF
p = no_of_PACF_reg_par
reg_lags_pacf = append( order(col1_mon1_pacf)[1:p]   , order(col1_mon1_pacf)[(length(col1_mon1_pacf)-(p-1)) : length(col1_mon1_pacf)]     )

#reg_lags_acf = append(reg_lags_acf, c(71,72,73,95,96,97,119,120,121, 143,144,145,167,168,169))

reg_lags = unique( append(reg_lags_acf, reg_lags_pacf) )
reg_lags = sort(reg_lags)

#####################################################

























