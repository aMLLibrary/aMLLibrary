# a-MLLibrary
Library for the generation of regression models.

The main script of the library is `run.py`:

```
usage: run.py [-h] -c CONFIGURATION_FILE [-d] [-s SEED] [-o OUTPUT] [-j J]
              [-g] [-t] [-l]

Perform exploration of regression techniques

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIGURATION_FILE, --configuration-file CONFIGURATION_FILE
                        The configuration file for the infrastructure
  -d, --debug           Enable debug messages
  -s SEED, --seed SEED  The seed
  -o OUTPUT, --output OUTPUT
                        The output where all the models will be stored
  -j J                  The number of processes to be used
  -g, --generate-plots  Generate plots
  -t, --self-check      Predict the input data with the generate regressor
  -l, --details         Print results of the single experiments
```
Example of configuration files can be found under `example_configurations` directory.


## Tutorial
To run your first example job with this library, please issue the following command in your terminal:
```shell
python3 run.py -c example_configurations/simplest_example_1.ini -o output_example
```
This will extract the experiment configuration from the `simplest_example_1.ini` file and write any output file into the `output_example` folder.
If the `-o` argument is missing, the default name `output` will be used for the output folder.
Please note that if the output folder already exists, it will not be overwritten, and the execution will stop right away.

Results will be summarized in the `results.txt` file, as well as printed to screen during the execution of the experiment.


### Predicting module
This library also has a predicting module, in which you can use an output regressor in the form of a Pickle file to make predictions about new, previously-unseen data.
It is run via the `predict.py` file.
First of all, run the library to create a regression model similarly to what was indicated in the first part of the tutorial section:
```shell
python3 run.py -c example_configurations/faas_test.ini -o output_test
```
Then, you can apply the obtained regressor in the form of the `LRRidge.pickle` file by running:
```shell
python3 predict.py -c example_configurations/faas_predict.ini -r output_test/LRRidge.pickle -o output_test_predict
```
Please refer to the `predict.py` file itself for more information.


## Docker image
This section shows how to create and use the Docker container image for this library.
It is not strictly needed, but it ensures an environment in which dependencies have the correct version, and in which it is guaranteed that the library works correctly.
This Docker image can be built from the `Dockerfile` at the root folder of this repository by issuing the command line instruction
```shell
sudo docker build -t brunoguindani/a-mllibrary .
```
To run a container and mount a volume which includes the root folder of this repository, please use
```shell
sudo docker run --name aml --rm -v $(pwd):/a-MLlibrary -it brunoguindani/a-mllibrary
```
which defaults to a `bash` terminal unless a specific command is appended to the line.
In this terminal, you may run the same commands as in a regular terminal, including the ones from the Tutorial section.


## Hyperopt
This library is integrated with the Hyperopt package for hyperparameter tuning via Bayesian Optimization.
This search mode is activated by inserting the `hyperparameter_tuning = Hyperopt` flag in the "General" section of the configuration file, as well as appropriate `hyperopt_max_evals` and `hyperopt_save_interval` values.
When using Hyperopt, strings representing prior distributions, such as `'loguniform(0.01,1)'`, may be assigned to hyperparameters instead of the usual lists of values used in grid search mode.
Such strings refer to and are interpreted as Hyperopt prior objects, assuming they are appropriately formatted; please head to https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions for more information.

Note that logarithm-based distributions follow a different notation in `a-MLLibrary` configuration files than in the Hyperopt library, for the sake of clarity.
For instance, the string `'loguniform(a,b)'` in a configuration file means a log-uniform distribution with support `[a,b]`, whereas an equivalent distribution in Hyperopt notation would be `'loguniform(e^a,e^b)'` instead.
(`a-MLLibrary` performs this conversion of parameter notation automatically.)
