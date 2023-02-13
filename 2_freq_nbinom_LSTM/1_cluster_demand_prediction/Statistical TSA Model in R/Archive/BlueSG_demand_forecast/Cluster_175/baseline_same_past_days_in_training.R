


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


window = 0

tr_strt = 1 + 24*window


tr_stop = (6*7*24 + tr_strt - 1) #(6 weeks, 7 days, 24 hours)  

val_strt1 = tr_stop - 7*24 + 1 
val_stop1 = tr_stop - 7*24 + 24

val_strt2 = tr_stop - 7*2*24 + 1 
val_stop2 = tr_stop - 7*2*24 + 24

val_strt3 = tr_stop - 7*3*24 + 1 
val_stop3 = tr_stop - 7*3*24 + 24

val_strt4 = tr_stop - 7*4*24 + 1 
val_stop4 = tr_stop - 7*4*24 + 24

val_strt5 = tr_stop - 7*5*24 + 1 
val_stop5 = tr_stop - 7*5*24 + 24

val_strt6 = tr_stop - 7*6*24 + 1 
val_stop6 = tr_stop - 7*6*24 + 24

ts_strt = tr_stop + 1 # use same day on next week for testing.
ts_stop = tr_stop + 24




ts175_train = window(ts175,start=tr_strt,end=tr_stop)

ts175_val1 = window(ts175,start=val_strt1,end=val_stop1)
ts175_val2 = window(ts175,start=val_strt2,end=val_stop2)
ts175_val3 = window(ts175,start=val_strt3,end=val_stop3)
ts175_val4 = window(ts175,start=val_strt4,end=val_stop4)
ts175_val5 = window(ts175,start=val_strt5,end=val_stop5)
ts175_val6 = window(ts175,start=val_strt6,end=val_stop6)

ts175_test = window(ts175,start=ts_strt,end=ts_stop)


base_pred = (as.vector(ts175_val1)+as.vector(ts175_val2)+as.vector(ts175_val3)+as.vector(ts175_val4)+as.vector(ts175_val5)+as.vector(ts175_val6))/6


base_score = tscount::scoring(ts175_test, base_pred)
base_score


base_sum_abs_err = sum(abs(as.vector(ts175_test) - floor(0.5+base_pred)))
base_sum_abs_err

#####################################################


###### plot test result ##########

test_hrs = length(ts175_test)

test_data = data.frame(seq(1:test_hrs), ts175_test,base_pred)

#text_vec = c(as.character(base_sum_abs_err),as.character(base_score), as.character(test_sum_abs_err),as.character(test_score) )
text_vec = vector()
text_vec = append(text_vec,'base (avg)' )
text_vec = append(text_vec,base_sum_abs_err )
text_vec = append(text_vec,base_score[7] )
text_vec = paste(text_vec,collapse=" ")



fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:3])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title=text_vec)



fig4
###### plot test result ##########


#####################################################

