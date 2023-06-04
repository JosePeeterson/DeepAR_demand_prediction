# #############################################
# library(pracma)
# 
# print('Approximate entropy and Sample entropy to measure regularity and predictability of
#       time series\n')
# 
# appx_entr = vector()
# sample_entr = vector()
# 
# for (i in names(tam_of_clstrs)[2:11])
# {
# ts1 = unlist(as.vector(tam_of_clstrs[[i]]))
# 
# ae =pracma::approx_entropy(ts1, edim = 2, r = 0.2*sd(ts1), elag = 1)
# appx_entr = append(appx_entr,ae)
# 
# se = pracma::sample_entropy(ts1, edim = 2, r = 0.2*sd(ts1), tau = 1)
# sample_entr = append(sample_entr,se)
# 
# }
# 
# print('approximate_entropy\n')
# print(appx_entr)
# 
# print('sample_entropy\n')
# print(sample_entr)
###############################################



# ##############################################
# print('acf of cluster 175')
# 
# library(tseries)
# 
# 
# acf_val = acf(tam_of_clstrs$clstr_175,pl=TRUE,lag.max = 2180)
# 
# df = data.frame(acf_val$acf)
# df_acf = df$acf_val.acf
# 
# print(df)
# 
# fig = plot_ly(df, x = seq(1:2180), y=df_acf, type="scatter", mode= 'lines')%>%
#       layout( yaxis = list(title = 'acf_175'),xaxis=  list(title = 'lags/hours') )%>%
#       add_trace(y=0.05)%>%
#       add_trace(y=-0.05)
# fig
# ##############################################


# ##############################################
# # ACF of (one) differenced time series
# tam_of_clstr_175_diff = diff(tam_of_clstrs$clstr_175,1)
# acf(tam_of_clstr_175_diff,pl=TRUE,lag.max = 2180)
# 
# ##############################################




##############################################
print('PACF of cluster 175')

library(tseries)


pacf_val = pacf(tam_of_clstrs$clstr_175,pl=TRUE,lag.max = 2180)

pacf_val$acf

# -------------------------------------------------------------------------



df = data.frame(pacf_val$acf)
df_acf = df$pacf_val.acf

print(df)

fig = plot_ly(df, x = seq(1:2179), y=df_acf, type="scatter", mode= 'lines')%>%
  layout( yaxis = list(title = 'PACF_175'),xaxis=  list(title = 'lags/hours') )%>%
  add_trace(y=0.05)%>%
  add_trace(y=-0.05)
fig


##############################################


