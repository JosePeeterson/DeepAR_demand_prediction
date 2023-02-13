
# DATA PREPROCESSING & VISUALIZATION - 1

#libraries
library(plotly)
#library(ggplot2)

# read time series from csv
outflow_df = read.csv(file = "outflow_clstr_dem.csv",header = TRUE)
inflow_df = read.csv(file = "inflow_clstr_dem.csv",header = TRUE)

# Tampines region
tampines_clstrs = c(167,175,168,169,170,167,171,172,173,172,173,167,126,174,174,170,175,175,175,175,168,168)
tampines_clstrs = unique(tampines_clstrs) + 1 # +1 for R indexing


tam_of_clstrs = data.frame(time=seq(1:2180)) # tampines outflow clusters


for(c in tampines_clstrs)
{
 new = unlist(as.vector(outflow_df[c]))
 tam_of_clstrs[ ,ncol(tam_of_clstrs) + 1] = new
 colnames(tam_of_clstrs)[ncol(tam_of_clstrs)] <- paste0("clstr_", c-1) 
}

fig = vector()
subplot_figs = list()

for (i in   names(tam_of_clstrs)[5:6] )
{

  
  fig[i] = plot_ly(tam_of_clstrs, x = ~time, y= tam_of_clstrs[[i]], name=paste(i), type="scatter", mode= 'lines' )
  
  subplot_figs = append(subplot_figs,fig[i])

}

fig = subplot(subplot_figs,nrows=2) 
fig 







