Library for the generation of regression models.
The main script of the library is run.py:

usage: run.py [-h] -c CONFIGURATION_FILE [-d] [-s SEED] [-o OUTPUT] [-j J]
              [-g] [-t]

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

Example of configuration files can be found under example_configurations directory
