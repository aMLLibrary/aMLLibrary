"""
Copyright 2021 Bruno Guindani

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import pandas as pd

from model_building.predictor import Predictor

def main():
    """
    Script to perform prediction on new data using old models

    The main arguments of this script are:
    -r, --regressor (required): specifies the location of the regressor to be used, in form of a Pickle binary file (e.g. LRRidge.pickle). This is usually given as output of a previous run of the library.
    -c, --config-file: specifies the configuration file which contains the dataset to perform prediction on, and the target variable to predict. Example of such configuration files, containing "predict" in their name, can be found in example_configurations directory.

    Other arguments are:
    -o, --output: specifies the output directory where logs and results will be put. If the directory already exists, the script fails. This behaviour has been designed to avoid unintentinal overwriting.
    -d, --debug: enables the debug printing.
    -m, --mape-to-file: enables printing the MAPE of the prediction to a text file called mape.txt.

    This file is used to perform prediction using a configuration file, but also contains an example of inline prediction using a Pandas DataFrame, which requires no additional files aside from the Pickle regressor one.

    Usage example:
    First, produce the original model with
    $ python3 run.py -c example_configurations/faas_test.ini -o output_test
    Then, use the predict() from file...
    $ python3 predict.py -c example_configurations/faas_predict.ini -r output_test/LRRidge.pickle -o output_test_predict
    ...or the inline predict_from_df():
    $ python3 predict.py -r output_test/LRRidge.pickle -o output_test_predict_2
    """
    parser = argparse.ArgumentParser(description="Perform prediction on new data using the provided models")
    parser.add_argument("-r", "--regressor",    help="binary regressor file to be used", required=True)
    parser.add_argument("-c", "--config-file",  help="configuration file for the infrastructure (optional, inline prediction available)")
    parser.add_argument("-o", "--output",       help="output folder where predictions will be stored", default='output_predict')
    parser.add_argument("-d", "--debug",        help="Enable debug messages", default=False, action='store_true')
    parser.add_argument("-m", "--mape-to-file", help="Write MAPE, if any, to text file", default=False, action='store_true')
    args = parser.parse_args()

    # Build object
    predictor_obj = Predictor(regressor_file=args.regressor, output_folder=args.output, debug=args.debug)

    # Perform prediction
    if args.config_file:
        # prediction reading from a config file
        predictor_obj.predict(config_file=args.config_file, mape_to_file=args.mape_to_file)
    else:
        # example of inline prediction on dataframe
        xx = pd.DataFrame(data=[[0.2224,2.0000,2.3852,600],
                                [0.2330,1.9669,2.3044,600]],
                          columns='Lambda,warm_service_time,cold_service_time,expiration_time'.split(',')
                          )
        yy = predictor_obj.predict_from_df(xx)
    print("End of predict.py")



if __name__ == '__main__':
    main()
