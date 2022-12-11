import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom



dem_len = 5600


def dem_hr(hr, n, p,v,len):

    dem_hr = np.array([])
    for i in range(5601):
        d =  v + np.random.negative_binomial(n, p, 1)
        z = np.array([0]*(hr-1))
        dem_hr =  np.append(dem_hr, d)
        dem_hr =  np.append(dem_hr, z)

    dem_hr = dem_hr[:len]

    return dem_hr


def gen_data(len):

    n1 = 4
    n2 = 9

    p1 = 0.75
    p2 = 0.5

    d4 = dem_hr(4, n1, p1, 0,len)
    d8 = dem_hr(8, n2, p2,0,len)


    dsub = d8 - d4
    dem = np.where(dsub>=0,d8,d4)


    dem = np.array(dem,dtype=np.float32)
    return dem


dem = gen_data(len=dem_len)

#np.save('2_freq_stoch_nbinom_dem',dem)


n1 = 4
n2 = 9

p1 = 0.75
p2 = 0.5

shape = 1/n1
mean = (1-p1)/(p1*shape)
print(mean,shape) # 9.0 0.1111111111111111   1.3333333333333333 0.25

plt.plot(dem)
plt.show()

















n1 = 4
n2 = 9

p1 = 0.75
p2 = 0.5

x = np.arange(nbinom.ppf(0.01, n1, p1),
              nbinom.ppf(0.99, n1, p1))

plt.plot(nbinom.pmf(x,n1,p1))
plt.show()

x=[]
y = []
for i in range(500):
    x.append(np.random.negative_binomial(n1, p1, 1)) # 0-2 
    y.append(np.random.negative_binomial(n2, p2, 1)) # 0-2 

plt.plot(y)
plt.show()
plt.plot(x)
plt.show()



# n = 2
# p = 0.95 # 0-3 vehicles

# mean = (n/p) -n
# shape = ((mean/p) - mean)/(mean**2)  

# print(mean,shape) 

# n = 1
# p = 0.8 # 0-4 vehicles

# mean = (n/p) -n
# shape = ((mean/p) - mean)/(mean**2)
# #print(mean,shape) # 0.75 0.3333333333333333


# x = np.arange(nbinom.ppf(0.01, n, p),
#               nbinom.ppf(0.99, n, p))
# plt.plot(x,nbinom.pmf(x,1,0.85))
# plt.plot(x,nbinom.pmf(x,2,0.95))
# plt.show()

# print()


# x=[]
# for i in range(500):
#     x.append(np.random.negative_binomial(1, 0.8, 1)) # 0-2 

# plt.plot(x)
# plt.show()
















