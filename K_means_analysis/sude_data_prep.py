def read_inputs(self, config_path, input_path, gaps_path):
    print(input_path)
    self.conf.read(config_path)
    debug = ast.literal_eval(self.conf.get('DebugLevel', 'debug'))
    logging.basicConfig(level=logging.DEBUG) if debug else logging.basicConfig(level=logging.INFO)
    target_column = int(self.conf.get('DataPreparation', 'target_column'))
    self.input_name = self.conf.get('DataPreparation', 'input_name')
    self.sparkdl_run = self.conf.get('DataPreparation', 'sparkdl_run')
    self.case = self.conf.get('DataPreparation', 'case')
    self.split = self.conf.get('DataPreparation', 'split')
    self.use_spark_info = bool(ast.literal_eval(self.conf.get('DataPreparation', 'use_spark_info')))
    self.fixed_features = bool(ast.literal_eval(self.conf.get('DataPreparation', 'fixed_features')))
    self.test_without_apriori = bool(ast.literal_eval(self.conf.get('DataPreparation', 'test_without_apriori')))

    if self.input_name in self.sparkdl_inputs:
        self.df = self.merge_files(input_path, gaps_path)
    if self.input_name in self.tpcds_inputs:
        self.df = pd.read_csv(input_path)
    # I added following 2 lines
    # else:
    # self.df = pd.read_csv(input_path)

    if target_column != 1:
        column_names = list(self.df)
        column_names[1], column_names[target_column] = column_names[target_column], column_names[1]
        self.df = self.df.reindex(columns=column_names)

    if self.input_name in self.sparkdl_inputs:
        self.df['inverse_nCores'] = 1 / self.df['nCores']
        self.df['inverse_nCoresTensorflow'] = 1 / self.df['nCoresTensorflow']
        if self.input_name == "runbest":
            self.df = self.df.drop(['run', 'users'], axis=1)
        else:
            self.df = self.df.drop(['run'], axis=1)
    if self.input_name in self.tpcds_inputs:
        self.df['inverse_nContainers'] = 1 / self.df['nContainers']
        self.df = self.df.drop(['run', 'users'], axis=1)

    self.logger.info("Data Preparation")


def scale_data(self, df):
    scaled_array = self.scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
    return scaled_df, self.scaler


def split_data(self, seed):
    data_conf = {}
    data_conf["case"] = self.case
    data_conf["split"] = self.split
    data_conf["input_name"] = self.input_name
    data_conf["sparkdl_run"] = self.sparkdl_run

    if self.input_name != "classifierselection":
        # Drop the constant columns
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]
        self.df = shuffle(self.df, random_state=seed)

        image_nums_train_data = (self.conf.get('DataPreparation', 'image_nums_train_data'))
        image_nums_train_data = [int(i) for i in ast.literal_eval(image_nums_train_data)]

        image_nums_test_data = (self.conf.get('DataPreparation', 'image_nums_test_data'))
        image_nums_test_data = [int(i) for i in ast.literal_eval(image_nums_test_data)]

        # If dataSize column has different values
        if "dataSize" in self.df.columns:
            self.data_size_indices = pd.DataFrame(
                [[k, v.values] for k, v in self.df.groupby('dataSize').groups.items()], columns=['col', 'indices'])

            self.data_size_train_indices = \
            self.data_size_indices.loc[(self.data_size_indices['col'].isin(image_nums_train_data))]['indices']
            self.data_size_test_indices = \
            self.data_size_indices.loc[(self.data_size_indices['col'].isin(image_nums_test_data))]['indices']

            self.data_size_train_indices = np.concatenate(list(self.data_size_train_indices), axis=0)
            self.data_size_test_indices = np.concatenate(list(self.data_size_test_indices), axis=0)

        else:

            self.data_size_train_indices = range(0, self.df.shape[0])
            self.data_size_test_indices = range(0, self.df.shape[0])

        data_conf["image_nums_train_data"] = image_nums_train_data
        data_conf["image_nums_test_data"] = image_nums_test_data

        if self.input_name in self.sparkdl_inputs:
            self.core_num_indices = pd.DataFrame([[k, v.values] for k, v in self.df.groupby('nCores').groups.items()],
                                                 columns=['col', 'indices'])
        if self.input_name in self.tpcds_inputs:
            self.core_num_indices = pd.DataFrame(
                [[k, v.values] for k, v in self.df.groupby('nContainers').groups.items()], columns=['col', 'indices'])
        # I added the following 2 lines:
        # else:
        # self.core_num_indices = pd.DataFrame([[k,v.values] for k,v in self.df.groupby('nContainers').groups.items()],columns=['col','indices'])

        core_nums_train_data = (self.conf.get('DataPreparation', 'core_nums_train_data'))
        core_nums_train_data = [int(i) for i in ast.literal_eval(core_nums_train_data)]

        core_nums_test_data = (self.conf.get('DataPreparation', 'core_nums_test_data'))
        core_nums_test_data = [int(i) for i in ast.literal_eval(core_nums_test_data)]

        # For interpolation and extrapolation, put all the cores to the test set.
        if set(image_nums_train_data) != set(image_nums_test_data):
            core_nums_test_data = core_nums_test_data + core_nums_train_data

        self.core_num_train_indices = \
        self.core_num_indices.loc[(self.core_num_indices['col'].isin(core_nums_train_data))]['indices']
        self.core_num_test_indices = \
        self.core_num_indices.loc[(self.core_num_indices['col'].isin(core_nums_test_data))]['indices']

        self.core_num_train_indices = np.concatenate(list(self.core_num_train_indices), axis=0)
        self.core_num_test_indices = np.concatenate(list(self.core_num_test_indices), axis=0)

        data_conf["core_nums_train_data"] = core_nums_train_data
        data_conf["core_nums_test_data"] = core_nums_test_data

        # Take the intersect of indices of datasize and core
        train_indices = np.intersect1d(self.core_num_train_indices, self.data_size_train_indices)
        test_indices = np.intersect1d(self.core_num_test_indices, self.data_size_test_indices)

    # Classifier selection, the analysis are only on the train set
    if self.input_name == "classifierselection":
        # Drop the constant columns
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]
        cores = self.df["nCores"]
        # Read
        image_nums_train_data = (self.conf.get('DataPreparation', 'image_nums_train_data'))
        image_nums_train_data = [int(i) for i in ast.literal_eval(image_nums_train_data)]

        core_nums_train_data = (self.conf.get('DataPreparation', 'core_nums_train_data'))
        core_nums_train_data = [int(i) for i in ast.literal_eval(core_nums_train_data)]

        data_conf["core_nums_train_data"] = core_nums_train_data
        data_conf["core_nums_test_data"] = []
        data_conf["image_nums_train_data"] = image_nums_train_data
        data_conf["image_nums_test_data"] = []

        train_indices = range(0, len(cores))
        test_indices = []

    # USING FIXED FEATURES
    if self.fixed_features == True:
        data_conf["fixed_features"] = True
        fixed_features_names = ["applicationCompletionTime", "maxTask_S8", "gap_value_3", \
                                "avgTask_S8", "maxTask_S5", "maxTask_S4", "maxTask_S3", \
                                "gap_value_2", "avgTask_S3", "avgTask_S5", "gap_value_1", \
                                "avgTask_S4", "SHmax_S3", "dataSize", "inverse_nCoresTensorflow", \
                                "nTask_S8", "maxTask_S2", "maxTask_S0", "avgTask_S2", "SHmax_S4", \
                                "avgTask_S0", "nCoresTensorflow", "nCores", "maxTask_S1", \
                                "inverse_nCores", "avgTask_S1", "SHavg_S4", "Bmax_S4", \
                                "nTask_S2", "SHavg_S3", "Bavg_S4"]
        if self.input_name in self.sparkdl_inputs:
            cores = self.df[["nCores"]]
        if self.input_name in self.tpcds_inputs:
            cores = self.df[["nContainers"]]
        self.df = self.df[fixed_features_names]
        train_cores = cores.ix[train_indices]
        test_cores = cores.ix[test_indices]
        data_conf["train_cores"] = train_cores
        data_conf["test_cores"] = test_cores
    else:
        data_conf["fixed_features"] = False

    # TESTING WITH APRIORI KNOWLEDGE
    if self.test_without_apriori == True and image_nums_train_data == image_nums_test_data:
        data_conf["test_without_apriori"] = True
        self.change_test_set(train_indices, test_indices)
    else:
        data_conf["test_without_apriori"] = False

    # Scale the data.
    self.df, scaler = self.scale_data(self.df)
    train_df = self.df.ix[train_indices]
    test_df = self.df.ix[test_indices]
    train_labels = train_df.iloc[:, 0]
    train_features = train_df.iloc[:, 1:]
    test_labels = test_df.iloc[:, 0]
    test_features = test_df.iloc[:, 1:]

    features_names = list(self.df.columns.values)[1:]
    data_conf["train_features_org"] = train_features.as_matrix()
    data_conf["test_features_org"] = test_features.as_matrix()
    return train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf
