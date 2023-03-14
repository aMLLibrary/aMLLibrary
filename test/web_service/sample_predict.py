import requests


def main():
    sample_data = {
      "regressor": "outputs/faas_test_ws/best.pickle",
      "config_file": "example_configurations/faas_predict.ini",
      "output": "outputs/faas_predict_ws"
    }

    port = 8888
    url = f"http://0.0.0.0:{port}/amllibrary/predict"
    sample_result = requests.post(url = url, json = sample_data)
    print(sample_result)
    print(sample_result.json())


if __name__ == "__main__":
    main()
