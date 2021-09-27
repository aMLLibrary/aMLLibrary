# a-MLLibrary

The Docker container image for this library can be built from the `Dockerfile` at the root folder of this repository by issuing the command line instruction
```shell
sudo docker build -t brunoguindani/a-mllibrary .
```
Alternatively, the image can also be found at https://hub.docker.com/repository/docker/brunoguindani/a-mllibrary or retrieved via
```shell
sudo docker pull brunoguindani/a-mllibrary
```
To run a container and mount a volume which includes the root folder of this repository, please use
```shell
sudo docker run --name aml --rm -v $(pwd):/a-MLlibrary -it brunoguindani/a-mllibrary
```
which defaults to a `bash` terminal unless a specific command is appended to the line.
