import requests


def main():
    sample_data = {
      "regressor": "outputs/faas_test_ws/best.pickle",
      "df": {
        "Lambda": [
          0.222486056568079,
          0.233038487465872,
          0.487490803901082
        ],
        "warm_service_time": [
          2.00009526463535,
          1.96695204745981,
          1.33774671968767
        ],
        "cold_service_time": [
          2.38527585140162,
          2.30448269192042,
          1.66541166626772
        ],
        "expiration_time": [
          600,
          600,
          600
        ],
        "ave_response_time": [
          2.00151865892106,
          1.96814934943309,
          1.33831639990086
        ]
      }
    }

    ## Note: working in Python, you can define a pandas dataframe and add it
    ## to the input data by running something as:
    # df = pd.DataFrame(
    #   data=[[0.222486056568079,2.00009526463535,2.38527585140162,600,2.00151865892106],
    #         [0.233038487465872,1.96695204745981,2.30448269192042,600,1.96814934943309],
    #         [0.487490803901082,1.33774671968767,1.66541166626772,600,1.33831639990086]],
    #   columns="Lambda,warm_service_time,cold_service_time,expiration_time,ave_response_time".split(',') 
    # ) 
    # df_dict = {key: vals.tolist() for key, vals in df.iteritems()}
    # sample_data["df"] = df_dict

    port = 8888
    url = f"http://0.0.0.0:{port}/amllibrary/predict"
    sample_result = requests.post(url = url, json = sample_data)
    print(sample_result)
    print(sample_result.json())


if __name__ == "__main__":
    main()
