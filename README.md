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


## Docker image
The Docker container image for this library can be built from the `Dockerfile` at the root folder of this repository by issuing the command line instruction
```shell
sudo docker build -t brunoguindani/a-mllibrary .
```
Alternatively, the image can also be found at https://hub.docker.com/repository/docker/brunoguindani/a-mllibrary or retrieved via
```shell
sudo docker pull brunoguindani/a-mllibrary
```
To run a container and mount a volume which includes the root folder of this repository, please use
```shell
sudo docker run --name aml --rm -v $(pwd):/a-MLlibrary -it brunoguindani/a-mllibrary
```
which defaults to a `bash` terminal unless a specific command is appended to the line.


## Hyperopt
This library is integrated with the Hyperopt package for hyperparameter tuning via Bayesian Optimization.
This search mode is activated by inserting the `hyperparameter_tuning = Hyperopt` flag in the "General" section of the configuration file, as well as appropriate `hyperopt_max_evals` and `hyperopt_save_interval` values.
When using Hyperopt, strings representing prior distributions, such as `'loguniform(0.01,1)'`, may be assigned to hyperparameters instead of the usual lists of values used in grid search mode.
Such strings refer to and are interpreted as Hyperopt prior objects, assuming they are appropriately formatted; please head to https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions for more information.

Note that logarithm-based distributions follow a different notation in `a-MLLibrary` configuration files than in the Hyperopt library, for the sake of clarity.
For instance, the string `'loguniform(a,b)'` in a configuration file means a log-uniform distribution with support `[a,b]`, whereas an equivalent distribution in Hyperopt notation would be `'loguniform(e^a,e^b)'` instead.
(`a-MLLibrary` performs this conversion of parameter notation automatically.)
