
library(plotly)
avg_6 = c(24,38,27,22,25,25,23,33,37,26,27,26,31,29,36)
base_all = c(29,29,32, 25,23,25,24,26,36,28,27,20,24,26,36)


fig = plot_ly(x=seq(1:length(avg_6)), y=~avg_6, type="scatter", mode='lines')%>% add_trace(y=~base_all)
fig

