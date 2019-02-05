import pandas as pd
import numpy as np

N_sample = 100
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

"""make other parts of the data """
x32 = np.random.normal(3, 2, N_sample).tolist()
x33 = np.random.normal(8, 3, N_sample).tolist()

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

"""make other parts of the data """
x22 = np.random.normal(3, 2, N_sample).tolist()
x23 = np.random.normal(8, 3, N_sample).tolist()

Data = np.zeros((int(N_sample), 7))

data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y3
data_dict['x1'] = x31
data_dict['x2'] = x32
data_dict['x3'] = x33
data_dict['dataSize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'testcase3.csv')
