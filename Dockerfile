## First of all, cd into the folder containing this file.
## Build the image with:
#  docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t a-mllibrary:1 .
## Run container with:
#  docker run --name aml --rm --user $(id -u):$(id -g) -v $(pwd):/a-MLlibrary -it a-mllibrary:1
## Example run:
#  pipenv run ./run.py --help
## Run FaaS files:
#  pipenv run ./run.py -c example_configurations/faas.ini -o output_faas

FROM python:3.7
ENV MY_DIR=/a-MLlibrary
WORKDIR ${MY_DIR}
RUN python -m pip install pipenv
COPY Pipfile .
RUN pipenv install
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
COPY . .
CMD bash
