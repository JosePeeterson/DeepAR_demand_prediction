


wt_1= rnorm(1,0,1)
wt_2= rnorm(1,0,1)

x = vector()

for (i in seq(1:200))
{
wt = rnorm(1,0,1)
x = append(x,10 + wt + 0.5*wt_1 + 0.3*wt_2 )

wt_2 =  wt_1
wt_1= wt

}

plot(x,type='b')

acf(x,pl=TRUE)