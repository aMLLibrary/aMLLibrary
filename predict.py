#!/usr/bin/env python3
import argparse

from model_building.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(
      description="Perform prediction on new data using the provided models")
    parser.add_argument('-c', "--config-file", help="configuration file for the infrastructure", required=True)
    parser.add_argument('-r', "--regressor", help="binary regressor file to be used", required=True)
    parser.add_argument('-o', "--output", help="output folder where predictions will be stored", required=True)
    #parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    args = parser.parse_args()

    predictor_obj = Predictor(config_file=args.config_file,
                              regressor_file=args.regressor,
                              output_folder=args.output)
    predictor_obj.predict()

if __name__ == '__main__':
    main()

# Example: pipenv run python predict.py -c example_configurations/00_predict.ini -r output_agora/blockscholes/blockscholes_itr1/LRRidge.pickle -o output_predict
