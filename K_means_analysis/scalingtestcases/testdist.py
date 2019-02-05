


import numpy as np
import pandas as pd

N_sample = 5

"""case 1"""

a10 = 10
a10_v = list(np.multiply(np.ones(N_sample), a10))

a11 = 5
a11_v = list(np.multiply(np.ones(N_sample), a11))

# x11 normal (0,1)
mu_x11 = 0
std_x11 = 1
x11 = np.random.normal(mu_x11, std_x11, N_sample).tolist()

y1 = np.add(np.multiply(a11_v, x11), a10_v)






"""case 2"""


a20 = 10
a20_v = list(np.multiply(np.ones(N_sample), a20))

a21 = 5
a21_v = list(np.multiply(np.ones(N_sample), a21))


# x21 normal (mu,std)
mu_x21 = 4
std_x21 = 7
x21 = np.random.normal(mu_x21, std_x21, N_sample).tolist()

y2 = np.add(np.multiply(a21_v, x21), a20_v)



"""case 3"""

a30 = 10
a30_v = list(np.multiply(np.ones(N_sample), a30))

a31 = 5
a31_v = list(np.multiply(np.ones(N_sample), a31))

a32 = 13
a32_v = list(np.multiply(np.ones(N_sample), a32))

# x31 normal (mu,std)
mu_x31 = 4
std_x31 = 7
x31 = np.random.normal(mu_x31, std_x31, N_sample).tolist()
x31_sqr = np.multiply(x31, x31)

y3 = np.add(np.multiply(a31_v, x31), np.multiply(a32_v, x31_sqr))
y3 = np.add(y3, a30_v)

"""case 4"""

a40 = 10
a40_v = list(np.multiply(np.ones(N_sample), a40))

a41 = 5
a41_v = list(np.multiply(np.ones(N_sample), a41))

a42 = 13
a42_v = list(np.multiply(np.ones(N_sample), a42))

# x41 normal (mu,std)
mu_x41 = 4
std_x41 = 7
x41 = np.random.normal(mu_x41, std_x41, N_sample).tolist()

# x42 normal (mu,std)
mu_x42 = 6
std_x42 = 9
x42 = np.random.normal(mu_x42, std_x42, N_sample).tolist()



x41_a41 = np.multiply(x41, a41_v)
x42_a42 = np.multiply(x42, a42_v)
y4 = np.add(x41_a41, x41_a41)
y4 = np.add(y4, a40_v)


"""case 5"""


a50 = 10
a50_v = list(np.multiply(np.ones(N_sample), a50))

a51 = 5
a51_v = list(np.multiply(np.ones(N_sample), a51))

a52 = 13
a52_v = list(np.multiply(np.ones(N_sample), a52))

# x51 uniform (mu,std)
mu_x51 = 4
std_x51 = 7
x51 = np.random.normal(mu_x51, std_x51, N_sample).tolist()

# x52 uniform (mu,std)

mu_x52 = 6
std_x52 = 9
x52 = np.random.normal(mu_x52, std_x52, N_sample).tolist()



x51_a51 = np.multiply(x51, a51_v)
x52_a52 = np.multiply(x52, a52_v)
y5 = np.add(x51_a51, x52_a52)
y5 = np.add(y5, a50_v)





ncores = np.zeros(N_sample).tolist()

for i in range(2,41,2):
    ncores[int(i/2 -1)] = i
    ncores[20+int(i/2 -1)] = i
    ncores[40+int(i/2 -1)] = i
    ncores[60+int(i/2 -1)] = i
    ncores[80+int(i/2 -1)] = i



datasize = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]


run = []

for i in range(0,N_sample):
    run.append(i)


Data = np.zeros((int(N_sample), 7))


data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y
data_dict['x1'] = x1
data_dict['x2'] = x2
data_dict['x3'] = x3
data_dict['datasize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df,'newd.csv')

l