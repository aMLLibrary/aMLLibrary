import numpy as np
import pandas as pd

N_sample = 100

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

"""make other parts of the data """
x12 = np.random.normal(3, 2, N_sample).tolist()
x13 = np.random.normal(8, 3, N_sample).tolist()


ncores = np.zeros(N_sample).tolist()

for i in range(2,41,2):
    ncores[int(i/2 -1)] = i
    ncores[20+int(i/2 -1)] = i
    ncores[40+int(i/2 -1)] = i
    ncores[60+int(i/2 -1)] = i
    ncores[80+int(i/2 -1)] = i


#datasize = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
datasize = []
for i in range(100):
    datasize.append(1)

run = []

for i in range(0,N_sample):
    run.append(i)

Data = np.zeros((int(N_sample), 7))


data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y1
data_dict['x1'] = x11
data_dict['x2'] = x12
data_dict['x3'] = x12
data_dict['dataSize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'testcase1.csv')
