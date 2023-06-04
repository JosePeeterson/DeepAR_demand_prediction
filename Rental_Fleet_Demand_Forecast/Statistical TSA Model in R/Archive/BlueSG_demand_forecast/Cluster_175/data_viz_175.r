
#####################################################
# DATA PREPROCESSING & LOADING

#libraries
library(plotly)
#library(ggplot2)

# read time series from csv
outflow_df = read.csv(file = "outflow_clstr_dem.csv",header = TRUE)
inflow_df = read.csv(file = "inflow_clstr_dem.csv",header = TRUE)

# Tampines region
tampines_clstrs = c(167,175,168,169,170,167,171,172,173,172,173,167,126,174,174,170,175,175,175,175,168,168)
tampines_clstrs = unique(tampines_clstrs) + 1 # +1 for R indexing, original index starts from 0.


tam_of_clstrs = data.frame(time=seq(1:2180)) # tampines outflow clusters


for(c in tampines_clstrs)
{
  new = unlist(as.vector(outflow_df[c]))
  tam_of_clstrs[ ,ncol(tam_of_clstrs) + 1] = new
  colnames(tam_of_clstrs)[ncol(tam_of_clstrs)] <- paste0("clstr_", c-1)
}
#####################################################


######################################################
##DATA VISUALIZATION - 1
fig = vector()
subplot_figs = list()

for (i in   names(tam_of_clstrs)[5:6] )
{

  fig[i] = plot_ly(tam_of_clstrs, x = ~time, y= tam_of_clstrs[[i]], name=paste(i), type="scatter", mode= 'lines' )

  subplot_figs = append(subplot_figs,fig[i])

}

fig = subplot(subplot_figs,nrows=2)
fig


#####################################################












#####################################################


ts175 = ts( tam_of_clstrs['clstr_175'] )
ts170 = ts( tam_of_clstrs['clstr_170'] )


ts175_mon1 = window(ts175,start=1,end=768)
ts175_mon1_pred = window(ts175,start=769,end=936)



######## ACF plot ###
ts175_mon1_acf = acf(ts175_mon1,pl=TRUE,lag.max=760)
ts175_mon1_pred_acf = acf(ts175_mon1_pred,pl=TRUE,lag.max=168)

df_mon1_acf = data.frame(ts175_mon1_acf$acf)
df_mon1_pred_acf = data.frame(ts175_mon1_pred_acf$acf)
# df = list(df_mon1_acf,df_mon1_pred_acf)

col1_mon1 = df_mon1_acf$ts175_mon1_acf.acf
col1_mon1_pred = df_mon1_pred_acf$ts175_mon1_pred_acf.acf
# col = list(col1_mon1,col1_mon1_pred)

# fig = vector()
# subplot_figs = list()

fig1 = plot_ly(df_mon1_acf, x = seq(0,(length(col1_mon1)-1)), y=~col1_mon1, type="scatter",mode='lines')%>%
       layout( yaxis = list(title = 'ACF_175_mon1'),xaxis=  list(title = 'lags/hours') )%>%
       add_trace(y=0.05)%>%
       add_trace(y=-0.05)
fig1

fig2 = plot_ly(df_mon1_pred_acf, x = seq(0,(length(col1_mon1_pred)-1)), y=~col1_mon1_pred, type="scatter", mode= 'lines')%>%
      layout( yaxis = list(title = 'ACF_175_mon1_pred'),xaxis=  list(title = 'lags/hours') )%>%
      add_trace(y=0.05)%>%
      add_trace(y=-0.05)
fig2
######## ACF plot ###



######## PACF plot ###
ts175_mon1_pacf = pacf(ts175_mon1,pl=TRUE,lag.max=760)
ts175_mon1_pred_pacf = pacf(ts175_mon1_pred,pl=TRUE,lag.max=168)

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
 
























