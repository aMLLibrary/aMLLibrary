"""
Copyright 2023 Federica Filippini

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

from flask import Flask, request, jsonify
from waitress import serve
import pandas as pd
import logging
import os

from model_building.predictor import Predictor
import sequence_data_processing


# ----------------------------------------------------------------------------
# global definitions
# ----------------------------------------------------------------------------
app = Flask(__name__)

# online path
train_path = "/amllibrary/train"
predict_path = "/amllibrary/predict"

# exit codes
NOT_FOUND = 404
POST_SUCCESS = 201

# error messages
error_msg = {
    404: "ERROR: page not found",
    414: "ERROR: missing mandatory input `configuration_file`",
    424: "ERROR: missing mandatory input `regressor`",
    434: "ERROR: either `config_file` or `df` must be provided",
    444: "ERROR: both `config_file` and `df` provided --> ambiguous call",
    454: "ERROR: aMLLibrary called `sys.exit`"
}

# set basic logging level
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------------
# train service
# ----------------------------------------------------------------------------
@app.route(train_path, methods=["POST"])
def train():
    """
    Starts training service

    Returns
    -------
    Message and key denoting the training outcome (success or failure)
    """
    # get all data
    data = request.get_json()
    
    # check existence of mandatory fields:
    KEY_ERROR = 0
    if "configuration_file" not in data.keys():
        KEY_ERROR = 10
    else:
        # extract configuration parameters
        configuration_file = data["configuration_file"]
        debug = data.get("debug", False)
        output = data.get("output", "output")
        j = data.get("j", 1)
        generate_plots = data.get("generate_plots", False)
        details = data.get("details", False)
        keep_temp = data.get("keep_temp", False)

        # set logging level for debugging (if required)
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            # train
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            processor = sequence_data_processing.SequenceDataProcessing(
                configuration_file, 
                debug=debug, 
                output=output, 
                j=j, 
                generate_plots=generate_plots, 
                details=details, 
                keep_temp=keep_temp
            )
            processor.process()

            # define output
            output = ("DONE", POST_SUCCESS)
        
        # define appropriate key error if the training module fails
        except SystemExit:
            KEY_ERROR = 50
    
    # if any key error is defined, return original data and error code
    if KEY_ERROR > 0:
        output = (error_msg[NOT_FOUND + KEY_ERROR], NOT_FOUND + KEY_ERROR)
           
    return jsonify(output[0]), output[1]



# ----------------------------------------------------------------------------
# predict service
# ----------------------------------------------------------------------------
@app.route(predict_path, methods=["POST"])
def predict():
    """
    Starts predict service

    Returns
    -------
    Message and key denoting the predict outcome (success or failure)
    The message contains the list of predicted values if the prediction is 
    done on a dataframe instead of a file
    """
    # get all data
    data = request.get_json()
    
    # check existence of mandatory fields:
    KEY_ERROR = 0
    if "regressor" not in data.keys():
        KEY_ERROR = 20
    elif ("config_file" not in data.keys()) and ("df" not in data.keys()):
        KEY_ERROR = 30
    elif ("config_file" in data.keys()) and ("df" in data.keys()):
        KEY_ERROR = 40
    else:
        # get configuration parameters
        regressor_file = data["regressor"]
        config_file = data.get("config_file", None)
        output_folder = data.get("output", "output_predict")
        debug = data.get("debug", False)
        mape_to_file = data.get("mape_to_file", False)
        df = data.get("df", None)

        # set logging level for debugging (if required)
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # initialize predictor
            predictor_obj = Predictor(
                regressor_file=regressor_file, 
                output_folder=output_folder, 
                debug=debug
            )

            # if configuration file is provided, perform prediction from file
            if "config_file" in data.keys():
                predictor_obj.predict(
                    config_file=config_file, 
                    mape_to_file=mape_to_file
                )
                result = "DONE"
            else:
                # otherwise, perform prediction from dataframe
                yy = predictor_obj.predict_from_df(
                    xx=pd.DataFrame(df),
                    regressor_file=regressor_file
                )
                result = str(yy)
            
            # define output        
            output = (result, POST_SUCCESS)
        
        # define appropriate key error if the training module fails
        except SystemExit:
            KEY_ERROR = 50
    
    # if any key error is defined, return original data and error code
    if KEY_ERROR > 0:
        output = (error_msg[NOT_FOUND + KEY_ERROR], NOT_FOUND + KEY_ERROR)
           
    return jsonify(output[0]), output[1]



# ----------------------------------------------------------------------------
# start
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=8888)
    serve(app, host="0.0.0.0", port=8888)
