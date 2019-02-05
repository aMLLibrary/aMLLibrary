import pandas as pd
import numpy as np

N_sample = 100

a50 = 10
a50_v = list(np.multiply(np.ones(N_sample), a50))

a51 = 5
a51_v = list(np.multiply(np.ones(N_sample), a51))

a52 = 13
a52_v = list(np.multiply(np.ones(N_sample), a52))

# x41 normal (mu,std)
mu_x51 = 4
std_x51 = 7
x51 = np.random.uniform(mu_x51, std_x51, N_sample).tolist()

# x42 normal (mu,std)
mu_x52 = 6
std_x52 = 9
x52 = np.random.uniform(mu_x52, std_x52, N_sample).tolist()



x51_a51 = np.multiply(x51, a51_v)
x52_a52 = np.multiply(x52, a52_v)
y5 = np.add(x51_a51, x51_a51)
y5 = np.add(y5, a50_v)

"""make other parts of the data """
x53 = np.random.normal(8, 3, N_sample).tolist()

ncores = np.zeros(N_sample).tolist()

for i in range(2, 41, 2):
    ncores[int(i / 2 - 1)] = i
    ncores[20 + int(i / 2 - 1)] = i
    ncores[40 + int(i / 2 - 1)] = i
    ncores[60 + int(i / 2 - 1)] = i
    ncores[80 + int(i / 2 - 1)] = i

# datasize = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
datasize = []
for i in range(N_sample):
    datasize.append(1)

run = []

for i in range(0, N_sample):
    run.append(i)



Data = np.zeros((int(N_sample), 7))

data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y5
data_dict['x1'] = x51
data_dict['x2'] = x52
data_dict['x3'] = x53
data_dict['dataSize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'testcase5.csv')
