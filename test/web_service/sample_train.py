import requests


def main():
    sample_data = {
      "configuration_file": "example_configurations/faas_test.ini",
      "output": "OUTPUT/faas_test_2"
    }

    port = 8888
    url = f"http://0.0.0.0:{port}/amllibrary/train"
    sample_result = requests.post(url = url, json = sample_data)
    print(sample_result)
    print(sample_result.json())


if __name__ == "__main__":
    main()
