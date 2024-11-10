# Docker

For installation:
1. [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
    * Don't forget the [post-installation instructions](https://docs.docker.com/engine/install/linux-postinstall/).
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)

### Create your dockerized code

Design your `Dockerfile`, then:

```shell
# build image
bash docker-build.sh

# run test script
bash docker-run.sh

# execute training
WKSPACE_DIR=$(dirname $(pwd))
docker run --rm --gpus all -v $WKSPACE_DIR:$WKSPACE_DIR jax-img \
    "cd $WKSPACE_DIR && python jax-nn-train.py --save_plot"
```
