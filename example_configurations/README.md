# Documentation for configuration files
A configuration file is a text file containing all needed information to run `aMLLibrary`.
All files in this folder conventionally use the `.ini` extension, but in principle they can be of any type of plain text file.
The content of a configuration file looks something like this:
```
[SectionName1]
option1 = value1
option2 = value2  # This is a comment, it will be ignored in the parsing of the file

[SectionName2]
option3 = value3
option4 = [valueA, valueB]  # List of values
option5 = {keyA: valA, keyB: valB}  # Key-value dictionary
```
For usage examples of the listed options, please check the configuration files in the [current folder](.).


## List of options
### `General` section

| Option | Type | Description | Notes |
| ------ | ---- | ----------- | ----- |
| `run_num`  | integer | number of runs for the given experiment campaign | |
| `techniques` | list of strings | list of regression techniques to be used | currently supported: `DecisionTree`, `LRRidge`, `NeuralNetwork`, `NNLS`, `RandomForest`, `Stepwise`, `SVR`, `XGBoost` |
| `y` | string | name of the column which will be the regression target | |
| `hyperparameter_tuning` | string | hyperparameter tuning method to be used | default: grid search, set to `Hyperopt` to use Bayesian Optimization instead (see [below](#hyperopt)) |
| `hyperopt_max_evals` | integer | maximum iterations for Bayesian Optimization | used with `Hyperopt` |
| `hyperopt_save_interval` | integer | number of iterations after which progress of Bayesian Optimization is saved to a checkpoint file (0 to deactivate) | used with `Hyperopt` |
| `save_training_regressors` | bool | set to `True` to also save regressors from the training phase, before the full dataset is used |
| `hp_selection` | string | hyperparameter selection technique | can be `HoldOut`, `KFold`, `All` |
| `validation` | string | model validation technique | can be `Extrapolation`, `HoldOut`, `Interpolation`, `KFold`, `All` |

Depending on which `hp_selection` and `validation` methods are chosen, you also need the following options:

| Option | Type | Description | Notes |
| ------ | ---- | ----------- | ----- |
| `extrapolation_columns` | dictionary {string: float} | column names and lower bound for extrapolation (see below) | used with `Extrapolation` |
| `hold_out_ratio`  | float in (0,1) | fraction size of the hold-out set | used with `HoldOut` |
| `interpolation_columns` | dictionary {string: float} or {string: list} | column names and interpolation step, or column names and test-set values (see below) | used with `Interpolation` |
| `folds` | integer | number of Cross-Validation folds | used with `KFold` |

`Extrapolation` means that any data point (i.e. row) which have one or more features strictly above the indicated threshold(s) will be placed in the test set.

The `Interpolation` dictionary can either contain a float or a list:
* a *float* is interpreted as an interpolation step `h`. One every `h` values (sorted in increasing order) for the indicated column(s) will be destined to the test set. Any data point which has one or more feature values with the selected value(s) will be placed in the test set;
* a *list* is interpreted as a list of values which will be destined to the test set. Any data which has one or more feature values with the indicated values will be placed in the test set.


### `DataPreparation` section
All options except `input_path` are not mandatory.

| Option | Type | Description | Notes |
| ------ | ---- | ----------- | ----- |
| `input_path`  | string | path to the dataset file | mandatory argument |
| `normalization` | string | set to `True` to apply normalization on the dataset | |
| `inverse` | list of strings | list of column names to compute the inverse of | `[*]` indicates all columns |
| `log` | list of strings | list of column names to compute the natural logarithm of | `[*]` indicates all columns |
| `ernest` | string | set to `True` to compute the features in the Ernest model ([Venkataraman et al, 2016](https://dl.acm.org/doi/10.5555/2930611.2930635)) | requires `datasize` and `cores` columns |
| `product_max_degree` | integer or `inf` | maximum degree of feature products to be computed | `inf` means the number of columns |
| `product_interactions_only` | string | set to `True` if power terms of a single feature should not be computed | used with `product_max_degree` |
| `selected_products`  | list of strings | compute only the indicated products | |
| `use_columns` | list of strings | consider only the listed columns and ignore the rest | |
| `skip_columns` | list of strings | ignore the listed columns and consider all other ones | |
| `skip_rows` | dictionary {string: float} | column names and lower bound for the values of rows that will be ignored | |
| `rename_columns` | dictionary {string: string} | old and new names for the columns to be renamed | |


### `FeatureSelection` section
This section is mandatory, and should only be used if one wants to perform some form of feature selection.

| Option | Type | Description | Notes |
| ------ | ---- | ----------- | ----- |
| `method`  | string | feature selection method | can be `SFS` or `XGBoost` |
| `max_features` | integer | maximum number of features to be selected | used with `SFS` |
| `min_features` | integer | minimum number of features to keep | used with `SFS`. Optional (default 1) |
| `folds` | integer | number of Cross-Validation folds to be used | used with `SFS` |
| `XGBoost_tolerance` | float in (0, 1) | maximum cumulative feature weight to be kept | used with `XGBoost` |


### Regression models sections
Each regression model used (i.e. indicated in [`techniques`](#general-section)) has its own section which specifies the values for the model hyperparameters.
For instance:
```
[LRRidge]
alpha = [0.02, 0.1, 1.0]
```
For specific examples for each model, please check out the configuration files in this folder.
Values are always lists of integers/floats/strings, based on the type of hyperparameter.
One can also use strings that represent hyperpriors to be used in a Bayesian Optimization hyperparameter tuning procedure (see next section).


### Hyperopt
This library is integrated with the Hyperopt package for hyperparameter tuning via Bayesian Optimization.
As mentioned [earlier](#general-section), this search mode is activated by inserting the `hyperparameter_tuning = Hyperopt` flag in the `General` section, as well as appropriate `hyperopt_max_evals` and `hyperopt_save_interval` values.
When using Hyperopt, strings representing prior distributions, such as `'loguniform(0.01,1)'`, may be assigned to hyperparameters instead of the usual lists of values used in grid search mode.
Such strings refer to and are interpreted as Hyperopt prior objects, assuming they are appropriately formatted; please head to the [Hyperopt wiki](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for more information.

Note that logarithm-based distributions follow a different notation in `aMLLibrary` configuration files than in the Hyperopt library, for the sake of clarity.
For instance, the string `'loguniform(a,b)'` in a configuration file means a log-uniform distribution with support `[a,b]`, whereas an equivalent distribution in Hyperopt notation would be `'loguniform(e^a,e^b)'` instead.
(`aMLLibrary` performs this conversion of parameter notation automatically.)



## Prediction files
The prediction module (invoked with [`predict.py`](predict.py)) may also be used with a configuration file, which is not necessarily the same one used to produce the regression model which is being used.
Such configuration files only require the `y` option in the `General` section and the `input_path` option in the `DataPreparation` section as described earlier.
If you wish to invoke the prediction module in inline mode from a Python script, please check out the [`predict.py`](predict.py) file itself.
