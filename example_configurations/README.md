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
For examples, please check the configuration files in this folder.


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
| `hp_selection` | string | hyperparameter selection technique | can be `HoldOut`, `KFold`, `All` |
| `validation` | string | model validation technique | can be `Extrapolation`, `HoldOut`, `Interpolation`, `KFold`, `All` |

Depending on which `hp_selection` and `validation` methods are chosen, you also need the following options:

| Option | type | description | notes |
| ------ | ---- | ----------- | ----- |
| `extrapolation_columns` | dictionary {string: float} | column names and lower bound for extrapolation | used with `Extrapolation` |
| `hold_out_ratio`  | float in (0,1) | fraction size of the hold-out set | used with `HoldOut` |
| `interpolation_columns` | dictionary {string: float} | column names and lower bound for interpolation | used with `Interpolation` |
| `folds` | integer | number of Cross-Validation folds | used with `KFold` |


### `DataPreparation` section
All options except `input_path` are not mandatory.

| Option | type | description | notes |
| ------ | ---- | ----------- | ----- |
| `input_path`  | string | path to the dataset file | mandatory argument |
| `inverse` | list of strings | list of column names to compute the inverse of | `[*]` indicates all columns |
| `log` | list of strings | list of column names to compute the natural logarithm of | `[*]` indicates all columns |
| `product_max_degree` | integer or `inf` | maximum degree of feature products to be computed | `inf` means the number of columns |
| `product_interactions_only` | set to `True` if power terms of a single feature should not be computed | used with `product_max_degree` |
| `selected_products`  | list of strings | TODO | |
| `use_columns` | list of strings | only consider the listed columns and ignore the rest | |
| `skip_columns` | list of strings | ignore the listed columns and consider all other ones | |
| `skip_rows` | dictionary {string: float} | column names and lower bound for the values of rows that will be ignored | |
| `ernest` | string | set to `True` if... TODO | |
| `normalization` | string | set to `True` if... TODO | |
| `rename_columns` | dictionary {string: string} | old and new names for the columns to be renamed | |

```
[FeatureSelection]
method = "SFS"
max_features = 1
folds = 5
```
