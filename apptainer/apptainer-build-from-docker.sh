#!/bin/bash
# start a local docker registry (do once only)
# docker run -d -p 5000:5000 --restart=always --name registry registry:2

# tag and push local container to local docker registry
docker tag jax-img:latest localhost:5000/jax-img:latest
docker push localhost:5000/jax-img:latest

# build apptainer container from local docker registry
apptainer build jax.sif docker-daemon://localhost:5000/jax-img:latest