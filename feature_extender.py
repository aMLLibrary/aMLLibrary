import sys

import pandas as pd
import numpy as np
import ast
import logging
import configparser as cp
import operator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold


import warnings
from sklearn.exceptions import DataConversionWarning

INV_PREFIX = 'Inv_'
COMB_CHAR = '-'
warnings.filterwarnings(action = 'ignore',category = DataConversionWarning)


def foundTermInv(element, res_list):
    if element.startswith(INV_PREFIX):
        if element.split(INV_PREFIX)[1] in res_list:
            return True
    elif (INV_PREFIX + element) in res_list:
        return True
    return False


def checkTermsInv(indices, feature_names):
    temp_list = list()
    for i in indices:
        temp_list.append(str(feature_names[i]))
    for el in temp_list:
        others_list = temp_list.copy()
        others_list.remove(el)
        if foundTermInv(str(el), others_list):
            return False
    return True




class FeatureExtender(object):
    def __init__(self):
        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
        self.logger = logging.getLogger(__name__)

        self.features = []
        self.feature_names = []
        self.combinatorial_terms_indices = {}
        self.combinatorial_terms_names = {}
        self.features_with_combinatorial_terms = []
        self.combinatorial_term_index = 0
        self.generated_features = pd.DataFrame()

    def read_inputs(self,config_path,features):
        self.conf.read(config_path)
        debug = ast.literal_eval(self.conf.get('DebugLevel','debug'))
        logging.basicConfig(level = logging.DEBUG) if debug else logging.basicConfig(level = logging.INFO)

        self.generate_features = None
        if len(self.conf.get('tf_deepspeech', 'generate_features')) != 0:
            self.generate_features = self.conf.get('tf_deepspeech','generate_features')
            self.generate_features = ast.literal_eval(self.generate_features)

        self.features = pd.DataFrame.as_matrix(features)
        self.features_df = features
        self.combinatorial_term_index = self.features.shape[1]
        self.features_with_combinatorial_terms = pd.DataFrame.as_matrix(features)
        self.feature_names = list(features.columns.values)
        cnt = 0
        for org_feature_name in features.columns:
            self.combinatorial_terms_indices[org_feature_name] = cnt
            cnt += 1 
        self.logger.info("Feature Extension") 

    def generate(self):
        if not self.generate_features is None:
            for new_feature in self.generate_features:
                name_list = str(new_feature).split(sep=COMB_CHAR)
                self.addInverse(name_list)
                if not all(el in self.features_df.columns for el in name_list):
                    print('Column not found')
                    for el in name_list:
                        if el not in self.features_df.columns:
                            print(el)
                    sys.exit()
                self.create_column(name_list)
                self.generated_features = self.generated_features.set_index(self.features_df.index)
        return self.generated_features

    def extend(self):
        n_terms = (self.conf.get('FeatureExtender','n_terms'))
        n_terms = [int(i) for i in ast.literal_eval(n_terms)]
        self.added= []
        for nn in n_terms:
            self.add_combinatorial_terms(nn,0,self.features.shape[1],[])
        adds = np.array(self.added).T
        if self.added:
            self.features_with_combinatorial_terms = np.concatenate((self.features_with_combinatorial_terms, adds), axis=1)
        result_df = pd.DataFrame(self.features_with_combinatorial_terms)
        self.combinatorial_terms_names = {val:key for (key, val) in self.combinatorial_terms_indices.items()}
        #result_df = result_df.rename(index=str, columns=self.combinatorial_terms_names)
        result_df = result_df.rename(columns=self.combinatorial_terms_names)
        result_df = result_df.set_index(self.features_df.index)
        return result_df,self.feature_names

    '''
    n_term: viene dalla lista che si imposta nel file di configurazione(è un elemento della lista di interi) 
    index: è l'indice della feature che sto considerando
    features_num: credo sia il numero totale di feature
    indices: struttura dati con gli indici delle feature che sto combinando
    '''

    def add_combinatorial_terms(self,n_term,index,features_num,indices):
        if n_term == 0:
            if checkTermsInv(indices,self.feature_names):
                #la colonna nuova è calcolata come prodotto tra tutte le feature facenti parte della struttura indices
                new_col = self.calculate_new_col_2(self.features,indices)
                # self.features_with_combinatorial_terms = np.c_[self.features_with_combinatorial_terms,new_col]
                self.added.append(new_col)
                new_col_name = ""
                #Crea il nome della nuova feature combinata
                elems = list()
                for ii in indices:
                    elems.append(str(self.feature_names[ii]))
                new_col_name = self.getName(elems)
                # for ii in indices :
                #     new_col_name = new_col_name + str(self.feature_names[ii]) + "-"

                #Aggiunge il nome della feature combinata alle altre feature
                self.feature_names.append(new_col_name[:])
                self.combinatorial_terms_indices[self.feature_names[-1]] = self.combinatorial_term_index
                self.combinatorial_term_index += 1
        else :
            iterations = list(range(index, features_num))
            for ii in iterations:
                self.add_combinatorial_terms((n_term-1),ii,features_num,(indices+[ii]))       

    def calculate_new_col(self,X,indices):
        index = 0
        new_col = X[:,indices[index]]
        for ii in list(range(1,len(indices))):
            new_col = np.multiply(new_col,X[:,indices[index+1]])
            index += 1
        return new_col

    def calculate_new_col_2(self,X,indices):
        new_col_2 = np.multiply.reduce(X[:, indices], axis=1)
        return new_col_2

    def getName(self, alist):
        alist.sort(key=str.lower)
        res = ''
        for el in alist:
            if not len(res) == 0:
                res += COMB_CHAR
            res += el
        return res

    def create_column(self, name_list):
        new_name = self.getName(name_list)
        temp_df = pd.DataFrame(np.ones(self.features_df.shape[0]),columns=[new_name], index=self.features_df.index)
        for el in name_list:
            temp_df[new_name] = temp_df[new_name].values * self.features_df[el].values

        self.generated_features = pd.concat([self.generated_features, temp_df], axis=1)
        return

    def addInverse(self, name_list):
        for el in name_list:
            if el.startswith(INV_PREFIX):
                original_feature = el[len(INV_PREFIX):]
                self.features_df[el] = 1 / self.features_df[original_feature]

        return