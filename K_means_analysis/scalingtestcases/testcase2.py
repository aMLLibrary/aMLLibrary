import pandas as pd
import numpy as np


N_sample = 100

a20 = 10
a20_v = list(np.multiply(np.ones(N_sample), a20))

a21 = 5
a21_v = list(np.multiply(np.ones(N_sample), a21))


# x21 normal (mu,std)
mu_x21 = 4
std_x21 = 7
x21 = np.random.normal(mu_x21, std_x21, N_sample).tolist()

y2 = np.add(np.multiply(a21_v, x21), a20_v)

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
for i in range(N_sample):
    datasize.append(1)

run = []

for i in range(0,N_sample):
    run.append(i)


"""make other parts of the data """
x22 = np.random.normal(3, 2, N_sample).tolist()
x23 = np.random.normal(8, 3, N_sample).tolist()

Data = np.zeros((int(N_sample), 7))


data_dict = {}
data_dict['run'] = run
data_dict['applicationCompletionTime'] = y2
data_dict['x1'] = x21
data_dict['x2'] = x22
data_dict['x3'] = x23
data_dict['dataSize'] = datasize
data_dict['nContainers'] = ncores

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'testcase2.csv')
