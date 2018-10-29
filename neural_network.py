"""
Copyright 2018 Elif Sahin
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import csv
import torch
import logging
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

class Net(nn.Module):
    def __init__(self, train_features, train_labels, cv_features, cv_labels, test_features, test_labels, debug, scenario, input_name, nn_loss_plots):
        super(Net, self).__init__()
        self.dtype = torch.FloatTensor
        self.inputs = Variable(torch.from_numpy(train_features.values).float()).type(self.dtype)
        self.labels = Variable(torch.from_numpy(train_labels.values.reshape(-1, 1)).float()).type(self.dtype)
        
        self.train_features = train_features
        self.train_labels = train_labels
        self.cv_features = cv_features
        self.cv_labels = cv_labels
        self.test_features = test_features
        self.test_labels = test_labels

        self.debug = debug
        self.nn_loss_plots = nn_loss_plots

        if self.nn_loss_plots == 'on':
            self.num_combination = 1

            self.train_loss = {}
            self.cv_loss = {}
            self.test_loss = {}

            self.out_dir = os.path.join('outputs', scenario + '_' + input_name + '_nn')
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.DEBUG) if debug else logging.basicConfig(level = logging.INFO)

    def prepare(self, params):
        self.num_perceptrons = params['num_perceptrons']
        self.num_layers = params['num_layers']
        self.num_minibatch = params['num_minibatch']
        self.activation_functions = params['activation_functions']
        
        all_combinations = []
        self.layer_dict = {}
        result = []
        
        for layer in self.num_layers:
            sub_perceptrons = self.num_perceptrons[0][:int(layer)]
            self.perceptron_structure = list(itertools.product(*sub_perceptrons))
            result = []
            sub_result = []

            for element in self.perceptron_structure:
                for num_perceptron in range(0 , len(element)):
                    sub_result.extend([element[num_perceptron]] * 2)

                result.append(sub_result)
                sub_result = []
            
            self.layer_dict[layer] = result
        
        combination_dict = {}
        for i in self.num_layers:
            combination_dict['num_layers'] = [i * 2]
            combination_dict['num_perceptrons'] = self.layer_dict[i]
            for activation_func in self.activation_functions:
                combination_dict['activation_functions'] = [self.get_full_list(activation_func)]
                
                for optimizer in params['optimizer']:
                    if optimizer == 'Adam':
                        combination_dict['beta1'] = params['beta1']
                        combination_dict['optimizer'] = ['Adam']
                        if 'momentum' in combination_dict.keys():
                            del combination_dict['momentum']
                    else:
                        combination_dict['optimizer'] = ['SGD']
                        combination_dict['momentum'] = params['momentum']
                        if 'beta1' in combination_dict.keys():
                            del combination_dict['beta1']
                    combination_dict['learning_rate'] = params['learning_rate']
                    combination_dict['loss'] = params['loss']
                    combination_dict['num_minibatch'] = params['num_minibatch']
                    combination_dict['l2_penalty'] = params['l2_penalty']

                    all_combinations.append(combination_dict.copy())
        return all_combinations

    def get_full_list(self, activation_func):
        len_activation_func = len(activation_func)
        result = activation_func[:]
        for i in range(0, len_activation_func * 2, 2):
            result.insert(i, 'linear')
        return result

    def forward(self, x):
        for i in range(0, self.num_layers):
            self.layer = self.layers[i]
            x = self.layer(x)

        self.output = self.layers[len(self.layers) - 1]
        out = self.output(x)
        return out

    def set_params(self, **params):
        self.criterion = nn.MSELoss()
        self.epochs = 10000
        self.num_perceptrons = params['num_perceptrons']
        self.num_layers = params['num_layers']
        self.num_minibatch = params['num_minibatch']
        self.activation_functions = params['activation_functions']
        self.layers = nn.ModuleList()
        
        for i in range(0, self.num_layers):
            if self.activation_functions[i] == 'linear' and i == 0:
                self.layers.append(nn.Linear(self.inputs.size()[1], self.num_perceptrons[i]))
            elif self.activation_functions[i] == 'linear':
                self.layers.append(nn.Linear(self.num_perceptrons[i - 1], self.num_perceptrons[i]))
            
            if self.activation_functions[i] == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif self.activation_functions[i] == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation_functions[i] == 'tanh':
                self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(self.num_perceptrons[self.num_layers - 1], 1))

        if params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr = params['learning_rate'], weight_decay = params['l2_penalty'])
        elif params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr = params['learning_rate'], weight_decay = params['l2_penalty'])
        elif params['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr = params['learning_rate'], weight_decay = params['l2_penalty'])

        self.logger.debug('Structure of the layers')
        self.logger.debug(self.layers)

        return self

    def fit(self, train_features, train_labels, weights):
        self.sample_weight = weights
        for epoch in range(self.epochs):
            epoch += 1

            for batch in range(self.num_minibatch):
                batch += 1

                self.optimizer.zero_grad()
                outputs = self.forward(self.inputs)
                loss = self.criterion(outputs, self.labels)
                loss.backward()
                self.optimizer.step()
                # self.loss_data = loss.item()
                
                if self.nn_loss_plots == 'on' and epoch % 1000 == 0:
                    train_predicted = self.predict(self.train_features)
                    cv_predicted = self.predict(self.cv_features)
                    test_predicted = self.predict(self.test_features)

                    self.train_loss[str(epoch)] = mean_squared_error(self.train_labels, train_predicted)
                    self.cv_loss[str(epoch)] = mean_squared_error(self.cv_labels, cv_predicted)
                    self.test_loss[str(epoch)] = mean_squared_error(self.test_labels, test_predicted)

        if self.nn_loss_plots == 'on':
            self.write_csv_files()
            self.num_combination += 1
        return self

    def predict(self, cv_features):
        return self.forward(Variable(torch.from_numpy(cv_features.values).float())).data.numpy()

    def mean_squared_error(self, cv_labels, cv_predicted):
        return np.mean(np.power(cv_labels - cv_predicted, 2))
        
    def write_csv_files(self):
        if self.nn_loss_plots == 'on':
            f = open(os.path.join(self.out_dir, 'Train_loss_' + str(self.num_combination) + '.csv'), 'w')
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(['Epoch', 'Loss'])
            for row in self.train_loss.items():
                writer.writerow(row)

            f = open(os.path.join(self.out_dir, 'Cv_loss_' + str(self.num_combination) + '.csv'), 'w')
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(['Epoch', 'Loss'])
            for row in self.cv_loss.items():
                writer.writerow(row)

            f = open(os.path.join(self.out_dir, 'Test_loss_' + str(self.num_combination) + '.csv'), 'w')
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(['Epoch', 'Loss'])
            for row in self.test_loss.items():
                writer.writerow(row)
