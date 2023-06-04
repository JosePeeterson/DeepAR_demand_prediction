
library(tscount)
pred_hrs = 24

base_pred = ts175_mon1[(length(ts175_mon1) - (7*pred_hrs) - pred_hrs + 1) : (length(ts175_mon1) - (7*pred_hrs))] 
# same day in last week
  
 # 

base_score = tscount::scoring(ts175_mon1_pred[1:pred_hrs], base_pred)

base_score


plot(ts175_mon1_pred[1:pred_hrs])
plot(base_pred)

sum(abs(ts175_mon1_pred[1:pred_hrs] - base_pred))

sum(abs(ts175_mon1_pred[1:pred_hrs] - floor(0.5+ts175_mon1_fit_pred$pred)))















