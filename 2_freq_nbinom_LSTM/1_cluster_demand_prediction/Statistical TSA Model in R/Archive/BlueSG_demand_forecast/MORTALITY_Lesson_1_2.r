mort=scan("cmort.dat")

plot(mort,type='o')
mort = ts(mort)


mortdiff = diff(mort,1)

plot(mortdiff,type='o')
acf(mortdiff,xlim=c(1,24))

mortdifflag1=lag(mortdiff,-1)

acf(mortdifflag1)

pacf(mortdifflag1)

y=cbind(mortdiff,mortdifflag1)

mortdiffar1 = lm(y[,1]~y[,2])

summary(mortdiffar1)

#acf(mortdiffar1$residuals,xlim=c(1,18))
