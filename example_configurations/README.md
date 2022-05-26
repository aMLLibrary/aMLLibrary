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
| Option | type | description | notes |
| ------ | ---- | ----------- | ----- |
| `run_num`  | integer | number of runs for the given experiment campaign | |
| `techniques` | list of strings | list of regression techniques to be used | |
| `y` | string | name of the column which will be the regression target | |
| `hyperparameter_tuning` | string | hyperparameter tuning method to be used | default: grid search, set to `Hyperopt` to use Bayesian Optimization instead |
| `hyperopt_max_evals` | integer | maximum iterations for Bayesian Optimization | to be used with `Hyperopt` |
| `hyperopt_save_interval` | integer | number of iterations after which progress of Bayesian Optimization is saved to a checkpoint file (0 to deactivate) | to be used with `Hyperopt` |
| `hp_selection` | string | hyperparameter selection technique | see below |
| `validation` | string | model validation technique | see below |
| --------------------------- |

`hp_selection` and `validation` can have the following values: `Extrapolation`, `HoldOut`, `Interpolation`, `KFold`, `All`.
Depending on which ones are chosen, you also need the following options:

| Option | type | description | to be used with? |
| ------ | ---- | ----------- | ---------------- |
| `extrapolation_columns` | dictionary: string -> float | column names and lower bound for extrapolation, e.g. `extrapolation_columns = {"x2": 4}` | `Extrapolation` |
| `folds` | integer | number of Cross-Validation folds | `KFold` |
| `hold_out_ratio`  | float in (0,1) | fraction size of the hold-out set | `HoldOut` |
| `interpolation_columns` | dictionary: string -> float | column names and lower bound for interpolation, e.g. `interpolation_columns = {"x2": 4}` | `Interpolation` |
| --------------------------- |

```




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
