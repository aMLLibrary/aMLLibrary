import os
import csv
import datetime
import shutil
import subprocess
import numpy as np
import configparser as cp
import pandas as pd

test_core_cases = [[8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
                   [8, 10, 14, 16, 20, 22, 26, 28, 32, 34, 38, 40, 44, 46],
                   [8, 10, 12, 16, 18, 20, 24, 26, 28, 32, 34, 36, 40, 42, 44, 46],
                   [8, 10, 12, 14, 18, 20, 22, 24, 28, 30, 32, 34, 38, 40, 44, 46],
                   [10, 12, 14, 16, 20, 22, 24, 26, 28, 32, 34, 36, 38, 42, 44, 46],
                   [8, 12, 14, 18, 20, 22, 26, 28, 30, 32, 36, 38, 40, 42, 44, 46],
                   [8, 10, 12, 14, 18, 20, 22, 24, 28, 30, 32, 34, 38, 40, 42, 44, 48]]

train_core_cases = [[6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
                    [6, 12, 18, 24, 30, 36, 42, 48],
                    [6, 14, 22, 30, 38, 48],
                    [6, 16, 26, 36, 42, 48],
                    [6, 8, 18, 30, 40, 48],
                    [6, 10, 16, 24, 34, 48],
                    [6, 16, 26, 36, 46]]

all_cores_ml = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]

test_core_cases_tpc_ds = [[8, 12, 16, 22, 26, 30, 34, 38, 42],
                          [8, 10, 14, 16, 22, 24, 28, 30, 34, 36, 40, 42],
                          [8, 10, 12, 16, 18, 22, 26, 28, 30, 34, 36, 38, 42],
                          [8, 10, 12, 14, 18, 22, 24, 26, 28, 32, 34, 36, 38, 40],
                          [10, 12, 14, 16, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42],
                          [8, 12, 14, 18, 22, 24, 28, 30, 32, 34, 36, 38, 40, 42]]

train_core_cases_tpc_ds = [[6, 10, 14, 18, 24, 28, 32, 36, 40, 44],
                           [6, 12, 18, 26, 32, 38, 44],
                           [6, 14, 24, 32, 40, 44], 
                           [6, 16, 30, 42, 44],
                           [6, 8, 18, 32, 44],
                           [6, 10, 16, 26, 44]]

all_cores_tpc_ds = [6, 8, 10, 12, 14, 16, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]

test_core_cases_runbest = [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
                           [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
                           [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]]

train_core_cases_runbest = [[2, 8, 14, 20, 26, 32, 38, 44],
                            [2, 10, 18, 26, 34, 42],
                            [2, 12, 22, 32, 42]]

scores = {'SVR': 0, 'LinearRegressionWithLasso': 0, 'DecisionTree': 0, 'RandomForest': 0}
#scores = {}
#scores['NeuralNetwork'] = 0

if not os.path.exists("results"):
    os.makedirs("results")

config_file_path = "./parameters.ini"
input_file_path = "./inputs/P8_logistic_50.csv"

conf = cp.ConfigParser()
conf.optionxform = str
conf.read(config_file_path)

if conf.get('DataPreparation', 'hybrid_ml') == 'on':
    subprocess.call("python analytical_model_tool.py -c " + config_file_path + " -i " + input_file_path, shell = True)
    analytical_file_path = os.path.splitext(input_file_path)[0] + "_analytical.csv"
else:
    analytical_file_path = ""

algorithm = os.path.splitext(os.path.basename(input_file_path))[0]
test_data_size = "[50]"
train_data_size = "[50]"
data_sizes = "_50_50"
num_run = 10

if 'query' in algorithm:
    test_cores = test_core_cases_tpc_ds
    train_cores = train_core_cases_tpc_ds
    all_cores = all_cores_tpc_ds
    case_num = 7
elif 'runbest' in algorithm:
    test_cores = test_core_cases_runbest
    train_cores = train_core_cases_runbest
    case_num = 4
else:
    test_cores = test_core_cases
    train_cores = train_core_cases
    all_cores = all_cores_ml
    case_num = 8

result_folder = os.path.join('results', algorithm + data_sizes + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.makedirs(result_folder)

for j in range(0, num_run):
    print("Run " + str(j))
    seed = np.power(5, j)
    run_folder = os.path.join(result_folder, 'run_' + str(j + 1))
    os.makedirs(run_folder)
    if case_num == -1:
        print("BASE CASE")
        scenario = 'base_case'
        conf.set('DataPreparation', 'scenario', scenario)
        conf.set('DataPreparation', 'test_cores', '')
        conf.set('DataPreparation', 'train_cores', '')
        conf.set('DataPreparation', 'test_data_size', '')
        conf.set('DataPreparation', 'train_data_size', '')
        with open(config_file_path, 'w') as configfile:
            conf.write(configfile)

        if analytical_file_path:
            subprocess.call("python run.py " + str(seed) + " -c " + config_file_path + " -i " + input_file_path + " -a " + analytical_file_path, shell = True)
        else:
            subprocess.call("python run.py " + str(seed) + " -c " + config_file_path + " -i " + input_file_path, shell = True)

        df = pd.read_csv(os.path.join('outputs', scenario + '_' + algorithm + '_output.csv'))
        df = df.sort_values('Test Mape Error (%)', ascending = True).reset_index()
        for index, row in df.iterrows():
            scores[row['Algorithm']] += index

    else:
        for i in range(0, case_num):
            print("Case " + str(i))
            if i == 0:
                scenario = 'Case0'
                conf.set('DataPreparation', 'scenario', scenario)
                conf.set('DataPreparation', 'test_cores', '')
                conf.set('DataPreparation', 'train_cores', '')
                conf.set('DataPreparation', 'test_data_size', test_data_size)
                conf.set('DataPreparation', 'train_data_size', train_data_size)
            else:
                scenario = 'Case' + str(i) + data_sizes
                conf.set('DataPreparation', 'scenario', scenario)
                #if test_data_size == train_data_size:
                conf.set('DataPreparation', 'test_cores', str(test_cores[i - 1]))
                #else:
                #    conf.set('DataPreparation', 'test_cores', str(all_cores))
                conf.set('DataPreparation', 'train_cores', str(train_cores[i - 1]))
                conf.set('DataPreparation', 'test_data_size', test_data_size)
                conf.set('DataPreparation', 'train_data_size', train_data_size)
            with open(config_file_path, 'w') as configfile:
                conf.write(configfile)
        
            if analytical_file_path:
                subprocess.call("python run.py " + str(seed) + " -c " + config_file_path + " -i " + input_file_path + " -a " + analytical_file_path, shell = True)
            else:
                subprocess.call("python run.py " + str(seed) + " -c " + config_file_path + " -i " + input_file_path, shell = True)
            
            df = pd.read_csv(os.path.join('outputs', scenario + '_' + algorithm + '_output.csv'))
            df = df.sort_values('Test Mape Error (%)', ascending = True).reset_index()
            for index, row in df.iterrows():
                scores[row['Algorithm']] += index
    
    f = open(os.path.join('outputs', 'model_ranking.csv'), 'w')
    writer = csv.writer(f, delimiter = ',')

    writer.writerow(['Score', 'Algorithm'])
    for key, value in scores.items():
        writer.writerow([value, key])
    
    scores['LinearRegressionWithLasso'] = 0
    scores['RandomForest'] = 0
    scores['DecisionTree'] = 0
    scores['SVR'] = 0
    #scores['NeuralNetwork'] = 0
    
    shutil.move('plots', run_folder)
    shutil.move('outputs', run_folder)
    shutil.copy(input_file_path, run_folder)

out_folder = os.path.join(result_folder, 'outputs')
os.makedirs(out_folder)

for i in range(0, case_num):
    if i == 0:
        scenario = 'Case' + str(i)
    else:
        scenario = 'Case' + str(i) + data_sizes
    train_mape = 0
    cv_mape = 0
    test_mape = 0
    algorithms = 0
    time_to_find = 0
    for j in range(0, num_run):
        df = pd.read_csv(os.path.join(result_folder, 'run_' + str(j + 1), 'outputs', scenario + '_' + algorithm + '_output.csv'))
        algorithms = df['Algorithm']
        train_mape += df['Train Mape Error (%)']
        cv_mape += df['Cross-Validation Mape Error (%)']
        test_mape += df['Test Mape Error (%)']
        time_to_find += df['Found in (min)']
    
    train_mape /= num_run
    cv_mape /= num_run
    test_mape /= num_run
    time_to_find /= num_run
    df = pd.concat([algorithms, train_mape, cv_mape, test_mape, time_to_find], axis = 1)
    df.columns = ['Algorithm', 'Train Mape Error (%)', 'Cross-Validation Mape Error (%)', 'Test Mape Error (%)', 'Found in (min)']
    df.to_csv(os.path.join(out_folder, scenario + '_' + algorithm + '_output.csv'), header = True, index = False)

for i in range(0, case_num):
    if i == 0:
        scenario = 'Case' + str(i)
    else:
        scenario = 'Case' + str(i) + data_sizes
    df = pd.read_csv(os.path.join(out_folder, scenario + '_' + algorithm + '_output.csv'))
    df = df.sort_values('Test Mape Error (%)', ascending = True).reset_index()
    for index, row in df.iterrows():
        scores[row['Algorithm']] = scores[row['Algorithm']] + index + 1

f = open(os.path.join(out_folder, 'model_ranking.csv'), 'w')
writer = csv.writer(f, delimiter = ',')

writer.writerow(['Score', 'Algorithm'])
for key, value in scores.items():
    writer.writerow([value, key])

f.close()
shutil.copy('./run_library.py', result_folder)

