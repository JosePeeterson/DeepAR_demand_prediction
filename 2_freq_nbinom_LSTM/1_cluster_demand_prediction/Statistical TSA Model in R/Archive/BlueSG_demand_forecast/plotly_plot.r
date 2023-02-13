# plot_ly()
tamp_of_167 <- data.frame(time =seq(1:length(outflow_167)),of1 =outflow_167)
tamp_of_175 <- data.frame(time =seq(1:length(outflow_175)),of =outflow_175)


fig <- plot_ly(tamp_of_175, x = ~time, y= ~of, name='175', type="scatter", mode= 'lines' )
#fig <- fig %>% add_trace(y = outflow_167, name = '167', mode = 'lines+markers')
#fig

fig1 <- plot_ly(tamp_of_167, x = ~time, y= ~of1, name='167', type="scatter", mode= 'lines' )

#fig1



fig = subplot(fig1,fig,nrows=2)

fig = fig %>% layout(title = 'Subplots using Plotly')
fig

# raw viz time series
#par(mfrow=c(2,1))
#plot(seq(1:2180),outflow_167,type='b')
#plot(seq(1:2180),outflow_175,type='b')
