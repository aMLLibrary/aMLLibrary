## First of all, cd into the folder containing this file.
## Build the image with:
#  (sudo) docker build -t a-mllibrary:1 .
## Run container (bash) with:
#  (sudo) docker run --name aml --rm -v $(pwd):/a-MLlibrary -it a-mllibrary:1
## Example run:
#  pipenv run ./run.py --help
## Run FaaS files:
#  pipenv run ./run.py -c example_configurations/faas.ini -o output_faas
## !!! All-in-one command:
#  sudo docker run --name aml --rm -v $(pwd):/a-MLlibrary -it a-mllibrary:1 pipenv run ./run.py -c example_configurations/faas.ini -o output_faas | tee log_faas.txt
## Remove root protection from "output" folder:
#  chmod -R a+rw output

FROM python:3.7
ENV MY_DIR=/a-MLlibrary
WORKDIR ${MY_DIR}
RUN python -m pip install pipenv
COPY Pipfile .
RUN pipenv install
COPY . .
CMD bash
