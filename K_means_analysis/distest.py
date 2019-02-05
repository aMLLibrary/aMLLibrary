


import numpy as np
import pandas as pd

N_sample = 100
alpha1 = 10
alpha2 = 5
mu_x1 = 1
mu_x2 = 4
mu_x3 = 6
std_x1 = 2
std_x2 = 8
std_x3 = 1


x1 = np.random.normal(mu_x1, std_x1, N_sample).tolist()

x2 = np.random.normal(mu_x2, std_x2, N_sample).tolist()

x3 = np.random.normal(mu_x3, std_x3, N_sample).tolist()

noise = np.random.normal(0, 1, N_sample).tolist()


y = []
for i in range(0,100):
    y.append(alpha1*x1[i]+alpha2*x2[i]+noise[i])




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

len(ncores)
len(datasize)
len(x3)
len(x2)
len(x1)
len(run)
len(y)




