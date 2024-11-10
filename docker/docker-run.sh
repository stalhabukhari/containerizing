#!/bin/bash
docker run -it --rm --gpus all -v "$(dirname $(pwd))":"/code-dir" \
  jax-img "python test_script.py"