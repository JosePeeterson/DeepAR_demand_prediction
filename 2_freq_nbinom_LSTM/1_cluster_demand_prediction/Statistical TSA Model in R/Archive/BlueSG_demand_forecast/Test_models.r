
#run data_preprocess_data_viz.r to load the time series data first

library(tscount)
library(plotly)

# x = tam_of_clstrs['clstr_175']
# 
# x = unlist( as.vector(x) )
# length(x)
# 
# x = x[1:1500]
# 
# #plot(seq(1:1500),x, type='b')
# 
# 
# #index 3 for clstr_175
# 
# # fig = plot_ly(tam_of_clstrs, x=~tam_of_clstrs[[1]], y=~tam_of_clstrs[[3]],type='scatter',mode='lines')
# # fig
# 
# ts_175 = ts(tam_of_clstrs[[3]])
# 
# obs_past = unlist(as.vector(tam_of_clstrs[[3]][1:100]))
# 
# 
# fit_175 = tscount::tsglm(ts= ts_175, model=list(past_obs=NULL, past_mean=NULL) ) 
# summary(fit_175)
# plot(fit_175)

###############################


###############################
# Toy example

# tim_ser = ts(unlist(as.vector(tam_of_clstrs[[3]][1:20])))
# fit_ts = tscount::tsglm(ts= tim_ser, model=list(past_obs=c(1,2,3,4,5,6,9,10,13), past_mean=c(1,18) )) 
# summary(fit_ts)
# pit(fit_ts)
# plot(tim_ser,type='o')
# 
# q = predict(fit_ts, n.ahead = 2,newobs=c(2),level=0.01,type = 'quantiles',method = "conddistr" )
# 
# q
###############################


###############################
### Campylobacter infections
# 
# tim_ser = ts(unlist(as.vector(tam_of_clstrs[[3]][1:20])))
# fit_ts = tscount::tsglm(ts= tim_ser, model=list(past_obs=c(1,2,3,4,5,6,9,10,13), past_mean=c(1,18) )) 
# summary(fit_ts)
# 
# 
# 
# fitted_ts = fitted.values(fit_ts)
# fitted_ts = fitted(fit_ts) #same as fitted_ts = fitted.values(fit_ts)#
# #fitted_ts = lag(fitted_ts,1)
# residual_ts = residuals(fit_ts)
# 
# plot(tim_ser,type='o', col="green")
# lines(fitted_ts,type='o',col="red")
# lines(residual_ts,type='o',col="blue", )

###############################


###############################
### Campylobacter infections

# tim_ser = ts(unlist(as.vector(tam_of_clstrs[[3]][1:20])))
# acf_tim_ser = acf(tim_ser,pl=TRUE,lag.max = 20)
# 
# campyfit_pois = tscount::tsglm(ts= tim_ser, link="log", model=list(past_obs = c(1,6,10), past_mean = c(1,6,10)),
#                                distr = 'poisson')
# 
# campyfit_nbin = tscount::tsglm(ts= tim_ser,link = "log", model=list(past_obs = c(1,6,10), past_mean = c(1,6,10)),
#                                distr = 'nbinom')
# 
# acf(residuals(campyfit_pois),main='ACF of response residuals')
# marcal(campyfit_pois, main = "Marginal calibration")
# lines(marcal(campyfit_nbin, plot=FALSE), lty = "dashed")
# legend("bottomright", legend = c("Pois", "NegBin"), lwd = 1, lty = c("solid", "dashed"))
# 
# pit(campyfit_pois, ylim=c(0,1.5),main = "PIT Poisson" )
# pit(campyfit_nbin, ylim=c(0,1.5), main="PIT Negbin")
# 
# fitted_ts = fitted.values(campyfit_pois)
# fitted_ts = fitted(campyfit_pois)
# #fitted_ts = lag(fitted_ts,1)
# residual_ts = residuals(campyfit_pois)
# 
# plot(tim_ser,type='o', col="green")
# lines(fitted_ts,type='o',col="red")
# lines(residual_ts,type='o',col="blue")
# 
# 
# rbind(Poisson = scoring(campyfit_pois), NegBin = scoring(campyfit_nbin))
# 
# summary(campyfit_pois)
# summary(campyfit_nbin)



###############################




###############################

# Road accidents in Great Britain


ts_175 = ts(unlist(as.vector(tam_of_clstrs[[3]][1:100])))
# plot(ts_175,type='b',main='ts_175')
#acf(tim_ser,pl=TRUE, lag.max = 25)

cov_170 = ts(unlist(as.vector(tam_of_clstrs[[6]][1:100])))
# plot(cov_170,type='b',main='cov_170')
#acf(tim_ser,pl=TRUE,lag.max = 50)


reg = cbind(cov = cov_170, linearTrend = seq(along=cov_170)/6 )



ts_untill70 = window(ts_175,end = 70)
cov_untill70 = window(cov_170,end = 70)


#ts_fit =  tscount::tsglm(ts_untill70, model=list(past_obs=c(1,6,10), past_mean=c(1,6)),
#                                                 link="log",distr = 'nbinom', xreg = cov_untill70 )

#summary(ts_fit)

ts_70to100 = window(ts_175,start = 70, end = 100)
cov_70to100 = window(cov_170,start = 70, end = 100)

#pred_70to100 = predict(ts_fit, n.ahead=31, level=0.9,B=1000, global=TRUE, newxreg = cov_70to100 )$pred

#pred_70to100

ts_fitALL =  tscount::tsglm(ts_175, model=list(past_obs=c(1,6,10), past_mean=c(1,6,10)),
                                                 link="log",distr = 'nbinom', xreg = cov_170, init.method='firstobs' )

summary(ts_fitALL)

# plot(ts_70to100, type='o', col='red')
# lines(pred_70to100, type='o', col='blue' )

class(ts_fitALL$coefficients)

shape = 1/0.2122



plot(fitted.values(ts_fitALL), type='o', col='red')
lines(ts_175, type='o', col='blue')


#x3 = dnbinom(x=seq(1:10), size=(1/0.2122), mu=3)


#plot(reg[,2],type='b',main='linearTrend')
###############################


