# aMLLibrary
Library for the generation of regression models.

The main script of the library is `run.py`:

```
usage: run.py [-h] -c CONFIGURATION_FILE [-d] [-o OUTPUT] [-j J] [-g] [-l] [-k]

Perform exploration of regression techniques

options:
  -h, --help            show this help message and exit
  -c CONFIGURATION_FILE, --configuration-file CONFIGURATION_FILE
                        configuration file for the infrastructure
  -d, --debug           enable debug messages
  -o OUTPUT, --output OUTPUT
                        output folder where all the models will be stored
  -j J                  number of processes to be used
  -l, --details         print results of the single experiments
  -k, --keep-temp       do not remove temporary files after successful execution

```
Example of configuration files can be found under `example_configurations` directory.
See also the [`README.md`](example_configurations/README.md) file there.


## Installation
You can use `git clone` in the terminal to download the library.
Since it includes the HyperOpt submodule, you must also add the appropriate flag:
```shell
git clone <SSH/HTTPS url of the library> --recurse-submodules
```
Or, if you forgot the flag, you can still download the submodules even after cloning by running:
```shell
git submodule update --init --recursive
```


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
This library also has a predicting module, in which you can use an output model in the form of a Pickle file to make predictions about new, previously-unseen data.
It is run via the [`predict.py`](predict.py) file.
First of all, run the library to create a regression model (similarly to the first part of the section):
```shell
python3 run.py -c example_configurations/faas_test.ini -o output_test
```
Then, you can apply the obtained model in the form of the `LRRidge.pickle` file by running:
```shell
python3 predict.py -c example_configurations/faas_predict.ini -r output_test/LRRidge.pickle -o output_test_predict
```
For more information, please refer to the [`predict.py`](predict.py) file itself and to the [README.md](example_configurations/README.md#prediction-files) for configuration files.


### Web Services
The training and predicting modules mentioned above can be exposed as web services using a REST API implemented through Flask. To start the services, run:
```
python3 web_service.py
```
The train and predict modules will be accessible at `http://0.0.0.0:8888/amllibrary/train` and `http://0.0.0.0:8888/amllibrary/predict`, respectively. 
A `POST` request can be issued by passing the required data. Corresponding samples are provided in [test/web_service](test/web_service).
**Note:** The configuration/regressor/data files used by the modules should already be available to the train/predict services. Only the paths are sent as input to the requests.


## Docker image
This section shows how to create and use the Docker container image for this library.
It is not strictly needed, but it ensures an environment in which dependencies have the correct version, and in which it is guaranteed that the library works correctly.
This Docker image can be built from the `Dockerfile` at the root folder of this repository by using:
```shell
docker build -t amllibrary .
```
(You might need to add `sudo` if Docker requires root permissions on your machine.)
To run a container and mount a volume which includes the root folder of this repository, please use:
```shell
docker run --name aml --rm -v $(pwd):/aMLLibrary -it amllibrary
```
which defaults to a `bash` terminal unless a specific command is appended to the line.
In this terminal, you may run the same commands as in a regular terminal, including the ones from the [Tutorial](#tutorial) section.

**Note:** when using web services, remind to make the internal address and port visible from outside the container, by running:
```
docker run --name aml --rm -v $(pwd):/aMLLibrary -it -p 8888:8888 amllibrary
```


## Hyperopt
This library is integrated with the Hyperopt package for hyperparameter tuning via Bayesian Optimization.
For more information, please refer to the [`README.md`](example_configurations/README.md#hyperopt) for configuration files.


## Acknowledgments
This library is currently maintained by the LIGATE project, which was partially funded by the European Commission under the Horizon 2020 Grant Agreement number 956137, as part of the European High-Performance Computing (EuroHPC) Joint Undertaking program.

It was previously maintained by the ATMOSPHERE project, which was also funded by the European Union under the Horizon 2020 Cooperation Programme, with Grant Agreement number 777154.
