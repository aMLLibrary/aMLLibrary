## First of all, cd into the folder containing this file.
## Build the image with:
#  sudo docker build -t brunoguindani/amllibrary .
## Run container (bash) with:
#  sudo docker run --name aml --rm -v $(pwd):/aMLlibrary -it brunoguindani/amllibrary
## Run small FaaS test:
#  python ./run.py -c example_configurations/faas_test.ini -o output_test
## Remove root permissions from "outputs" folder:
#  chmod -R a+rw outputs

FROM python:3.8
ENV MY_DIR=/aMLlibrary
WORKDIR ${MY_DIR}
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .
CMD bash
