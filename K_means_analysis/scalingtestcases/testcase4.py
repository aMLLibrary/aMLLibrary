import pandas as pd
import numpy as np

N_sample = 100

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

"""make other parts of the data """
x43 = np.random.normal(8, 3, N_sample).tolist()

ncores = np.zeros(N_sample).tolist()

for i in range(2, 41, 2):
    ncores[int(i / 2 - 1)] = i
    ncores[20 + int(i / 2 - 1)] = i
    ncores[40 + int(i / 2 - 1)] = i
    ncores[60 + int(i / 2 - 1)] = i
    ncores[80 + int(i / 2 - 1)] = i

datasize = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
#datasize = []
#for i in range(N_sample):
#    datasize.append(1)

run = []

for i in range(0, N_sample):
    run.append(i)



Data = np.zeros((int(N_sample), 7))

data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y4
data_dict['x1'] = x41
data_dict['x2'] = x42
data_dict['x3'] = x43
data_dict['dataSize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'testcase4.csv')
