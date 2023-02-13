

#####################################################
##### Model testing and comparison with baseline (previously observed value) #####

test_hrs = length(ts175_test)


#tim_btw_tr_ts = length(ts175[(tr_stop + 1) : (ts_strt-1)])
#ts175_test_fit = predict( ts175_train_fit_TRUE, n.ahead=(test_hrs+tim_btw_tr_ts), newobs = ts175[(tr_stop + 1) : (ts_strt-1)], level=0)


ts175_test_fit = predict( ts175_train_fit_TRUE, n.ahead=test_hrs)



# same day in last week, go back only 6 days instead of 7 to account for 1 day used in validation.
#base_pred = ts175_train[(length(ts175_train) - (6*test_hrs) - test_hrs + 1) : (length(ts175_train) - (6*test_hrs))] 
base_pred = ts175_val


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


