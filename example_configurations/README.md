# Documentation for configuration files
A configuration file is a text file containing all needed information to run `a-MLLibrary`.
All files in this folder conventionally use the `.ini` extension, but in principle they can be of any type of plain text file.
The content of a configuration file looks something like this:
```
[SectionName1]
option1 = value1
option2 = value2

[SectionName2]
option3 = value3  # This is a comment, it will be ignored in the parsing of the file
```

## List of options
### `General` section
| Option | type | description |
| ------ | ---- | ----------- |
| `General`
| `run_num`  | integer | number of runs for the given experiment campaign |
| `techniques` | list of strings | TODO |
-------------------------------------
| `DataPreparation` |
| `input_path` | string | TODO |
-------------------------------------

```
[General]
techniques = ['RandomForest', 'SVR', 'XGBoost']
hp_selection = HoldOut
hold_out_ratio = 0.2
validation = Extrapolation
extrapolation_columns = {"x2": 4}
folds = 4
y = "y"
hyperparameter_tuning = Hyperopt
hyperopt_max_evals = 10
hyperopt_save_interval = 5

[DataPreparation]
input_path = "inputs/simplest.csv"
inverse = [*]
log = [*]
product_max_degree = inf
product_interactions_only = True
use_columns = ["x1", "x2", "x4"]
skip_columns = ["x4"]
selected_products = ['x1 x1 x3 x1', 'x2 x3']
skip_rows = {'x2': 5}
normalization = True
ernest = True
rename_columns = {"x1": "cores", "x2": "datasize"}

[FeatureSelection]
method = "SFS"
max_features = 1
folds = 5
```
