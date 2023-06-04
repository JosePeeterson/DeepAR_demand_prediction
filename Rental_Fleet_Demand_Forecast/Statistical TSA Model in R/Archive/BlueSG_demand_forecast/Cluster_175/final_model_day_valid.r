

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



window = 0

tr_strt = 1 + 24*window


tr_stop = (4*7*24 + tr_strt - 1) #(5 weeks, 7 days, 24 hours)  

val_strt = tr_stop - 7*2*24 + 1 # use only 1 day (same day as test) for validation from train
val_stop = tr_stop - 7*2*24 + 7*2*24

ts_strt = tr_stop + 1 # use same day on next week for testing.
ts_stop = tr_stop + 24

# while(tr_strt < 1009)
# {


ts175_train = window(ts175,start=tr_strt,end=tr_stop)
ts175_val = window(ts175,start=val_strt,end=val_stop)
ts175_test = window(ts175,start=ts_strt,end=ts_stop)


base_pred = ts175_val[(7*24 + 1):(7*24 + 24)]

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



#####################################################
##Find lags due to the smallest and largest ACF and PACF values.##
## reg_lags is same as past_obs
min_error=100

for(r_acf in seq(1:5))
{
  for(r_pacf in seq(1:2))
  {
    r_acf
    r_pacf
    
    no_of_reg_par = r_acf # from ACF
    n = no_of_reg_par
    
    reg_lags_acf = append( order(tr_acf)[1:n]   , order(tr_acf)[(length(tr_acf)-(n-1)) : length(tr_acf)]     )
    reg_lags_acf = append(reg_lags_acf,c(1,24,48,168)) # standard lags observed to have effect
    reg_lags_acf = unique(reg_lags_acf)
    reg_lags_acf = sort(reg_lags_acf)
    
    
    
    no_of_PACF_reg_par = r_pacf # from PACF
    p = no_of_PACF_reg_par
    
    reg_lags_pacf = append( order(tr_pacf)[1:p]   , order(tr_pacf)[(length(tr_pacf)-(p-1)) : length(tr_pacf)]     )
    reg_lags = unique( append(reg_lags_acf, reg_lags_pacf) )
    reg_lags = sort(reg_lags)
    
    
    #####################################################
    
    
    
    
    #####################################################
    ## Train the model ##
    
    ts175_train_fit = tscount::tsglm(ts=ts175_train, link='log', model =
                                       list(past_obs = reg_lags, past_mean = reg_lags),distr = 'nbinom')
    
    
    summary(ts175_train_fit)
    train_score = tscount::scoring(ts175_train_fit)
    train_score
    
    ##### plot the fitted model time series with observed time series #####
    data = data.frame(seq(1:length(ts175_train)), ts175_train, fitted.values(ts175_train_fit))
    
    fig3 <- plot_ly(data, x = ~data[1])
    for(trace in colnames(data)[2:3])
      fig3 <- fig3 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig3
    
    #####################################################
    
    
    
    
    
    #####################################################
    ##### Model Validation and number of parameters Tuning #####
    
    val_hrs = length(ts175_val)
    
    ts175_val_fit = predict( ts175_train_fit, n.ahead=val_hrs )
    
    
    val_data = data.frame(seq(1:val_hrs), ts175_val, ts175_val_fit$pred)
    
    
    
    fig1 <- plot_ly(val_data, x = ~val_data[1])
    
    for(trace in colnames(val_data)[2:3])
      fig1 <- fig1 %>% add_trace(y = as.formula(paste0("~", trace)),
                                 name = trace, type='scatter', mode = 'lines')
    fig1
    
    val_score = tscount::scoring(ts175_val, ts175_val_fit$pred)
    val_score
    
    new_error = norm(c(val_score[4], val_score[6], val_score[7]), type ="2" )
    new_error
    
    if( new_error <= min_error)
    {
      true_r_acf = r_acf
      true_r_pacf = r_pacf
      true_reg_lags = reg_lags
      min_error = new_error
      min_error
    }
    
    
  }
  
}
true_r_pacf
true_r_acf
true_reg_lags

ts175_train_fit_TRUE = tscount::tsglm(ts=ts175_train, link='log', model =
                                        list(past_obs = true_reg_lags, past_mean = true_reg_lags),distr = 'nbinom')



#####################################################






#####################################################
##### Model testing and comparison with baseline (previously observed value) #####

test_hrs = length(ts175_test)


#tim_btw_tr_ts = length(ts175[(tr_stop + 1) : (ts_strt-1)])
#ts175_test_fit = predict( ts175_train_fit_TRUE, n.ahead=(test_hrs+tim_btw_tr_ts), newobs = ts175[(tr_stop + 1) : (ts_strt-1)], level=0)


ts175_test_fit = predict( ts175_train_fit_TRUE, n.ahead=test_hrs)



# same day in last week, go back only 6 days instead of 7 to account for 1 day used in validation.
#base_pred = ts175_train[(length(ts175_train) - (6*test_hrs) - test_hrs + 1) : (length(ts175_train) - (6*test_hrs))] 



base_score = tscount::scoring(ts175_test, base_pred)
base_score
#bs_score_list = append(bs_score_list,base_score)

test_score = tscount::scoring(ts175_test, ts175_test_fit$pred)
test_score
#ts_score_list = append(ts_score_list,test_score)


base_sum_abs_err = sum(abs(as.vector(ts175_test) - base_pred))
base_sum_abs_err
#bs_SAE = append(bs_SAE, base_sum_abs_err)

test_sum_abs_err = sum(abs(as.vector(ts175_test) - floor(0.5+ts175_test_fit$pred)))
test_sum_abs_err
#ts_SAE = append(ts_SAE, test_sum_abs_err)


###### plot test result ##########

test_data = data.frame(seq(1:test_hrs), ts175_test,base_pred, ts175_test_fit$pred)

#text_vec = c(as.character(base_sum_abs_err),as.character(base_score), as.character(test_sum_abs_err),as.character(test_score) )
text_vec = vector()
text_vec = append(text_vec,'base' )
text_vec = append(text_vec,base_sum_abs_err )
text_vec = append(text_vec,base_score[7] )
text_vec = append(text_vec,'test' )
text_vec = append(text_vec,test_sum_abs_err )
text_vec = append(text_vec,test_score[7]  )
text_vec = paste(text_vec,collapse=" ")



fig4 <- plot_ly(test_data, x = ~test_data[1])

for(trace in colnames(test_data)[2:4])
  fig4 <- fig4 %>% add_trace(y = as.formula(paste0("~", trace)),
                             name = trace, type='scatter', mode = 'lines') %>%
  layout(title=text_vec)



fig4
###### plot test result ##########


#####################################################

# tr_strt = tr_strt + 1009
# tr_stop = tr_stop + 24
# 
# val_strt = val_strt + 24
# val_stop = val_stop + 24
# 
# ts_strt = ts_strt + 24
# ts_stop = ts_stop + 24

#}



# bs_score_list
# ts_score_list
# bs_SAE
# ts_SAE



