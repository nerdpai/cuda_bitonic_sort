# CUDA Bitonic Sort

## Preparing

### Docker

After cloning the repo, the easiest way to setup environment is docker. \
So make docker pull:

```bash
docker pull docker pull nvidia/cuda:11.1.1-devel-ubuntu20.04 
```

(or any other version, here ya go: [nvidia/cuda-dockerhub])

### Run container

```bash
docker run --gpus all -it --rm -v "{project_path}":/cuda --net=host {docker_image}
```

And after that just connect to the container through vs code (or any other tool)

## Compile

In container run:

```bash
nvcc {path}/main.cu -o {path}/main.x
```

## Run

And run executable with:

```bash
{path}/main.x
```

[nvidia/cuda-dockerhub]: <https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md>
