# ts 1-6 = as.numeric(ts175_test)
# fs 1-6 = final_ts_autofit
# 
# pred_clstr_175 = c(fs1,fs2,fs3,fs4,fs5,fs6)
# clstr_175_GT = c(ts1,ts2,ts3,ts4,ts5,ts6)

library(caret)

fig2 = plot_ly(x = seq(1:length(clstr_175_GT)), y=~clstr_175_GT, type="scatter", mode='lines')%>%
  layout( yaxis = list(title = 'demand'),xaxis=  list(title = 'time') )%>%
  add_trace(y=pred_clstr_175 )%>%
  layout(title='final_prediction')
fig2


MAE(pred_clstr_175, clstr_175_GT  )
RMSE(pred_clstr_175, clstr_175_GT  )