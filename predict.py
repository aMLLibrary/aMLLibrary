#!/usr/bin/env python3
import argparse
import pandas as pd

from model_building.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Perform prediction on new data using the provided models")
    parser.add_argument("-r", "--regressor",    help="binary regressor file to be used", required=True)
    parser.add_argument("-c", "--config-file",  help="configuration file for the infrastructure (optional, inline prediction available)")
    parser.add_argument("-o", "--output",       help="output folder where predictions will be stored", default='output_predict')
    parser.add_argument("-d", "--debug",        help="Enable debug messages", default=False, action='store_true')
    parser.add_argument("-m", "--mape-to-file", help="Write MAPE, if any, to text file", default=False, action='store_true')
    args = parser.parse_args()

    # Build object
    predictor_obj = Predictor(regressor_file=args.regressor,
                              output_folder=args.output,
                              debug=args.debug
                              )

    # For inline prediction on dataframe
    xx = pd.DataFrame(data=[[0.2224,2.0000,2.3852,600],
                            [0.2330,1.9669,2.3044,600]],
                      columns='Lambda,warm_service_time,cold_service_time,expiration_time'.split(',')
                      )
    yy = predictor_obj.predict_from_df(xx)

    # Prediction from file
    predictor_obj.predict(config_file=args.config_file,
                          mape_to_file=args.mape_to_file
                          )

if __name__ == '__main__':
    main()

## USAGE
## First, produce the original model with
# (pipenv run) python run.py -c example_configurations/faas.ini -o output_faas
## Then, use the predict() from file
# (pipenv run) python predict.py -c example_configurations/faas_predict.ini -r output_faas/LRRidge.pickle -o output_faas_predict
## or the inline predict_from_df()
# (pipenv run) python predict.py -r output_faas/LRRidge.pickle -o output_faas_predict
