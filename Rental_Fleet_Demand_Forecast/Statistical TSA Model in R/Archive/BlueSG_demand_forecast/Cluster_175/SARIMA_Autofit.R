

library(plotly)
library(beepr)
library(fUnitRoots)
library(lubridate)

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



tam_175 = ts(tam_of_clstrs['clstr_175'],start=c(1,1), frequency = 24)


window = 0

tr_strt = 1 + 24*window
tr_stop = (6*7*24 + tr_strt - 1) #(6 weeks, 7 days, 24 hours)  

ts_strt = tr_stop + 1 # use same day on next week for testing.
ts_stop = tr_stop + 24

# tot_hrs = 24*(as.Date("1-12-22 00:00:00")-as.Date("1-9-24 00:00:00")) + 44
# ts175 = ts(tam_175$clstr_175 ,start=c(1,1) , end=c(1,tot_hrs),frequency=24*365)


ts175_train = window(tam_175,start=c(1,tr_strt),end=c(1,tr_stop))
 
ts175_test = window(tam_175,start=c(1,ts_strt),end=c(1,ts_stop))


ts_decomp = decompose(ts175_train)
plot(ts_decomp)

ts175_tr_adj = ts175_train - ts_decomp$trend 
ts175_tr_adj = ts175_tr_adj - ts_decomp$seasonal
length(ts175_tr_adj)


urkpssTest(ts175_tr_adj, type = c("tau"), lags = c("short"),use.lag = NULL, doplot = TRUE)
 
acf(ts175_tr_adj, lag.max = 100, plot = TRUE,na.action = na.pass)
pacf(ts175_tr_adj, lag.max = 100, plot = TRUE,na.action = na.pass)


autofit = auto.arima(ts175_train, trace=TRUE) 

ts_autofit = predict(autofit,n.ahead = 24)




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

test_data = data.frame(seq(1:test_hrs), ts175_test, ts_autofit$pred)

fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:3])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title=text_vec)


fig4


