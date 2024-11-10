#!/bin/bash
apptainer run --nv jax.sif "cd $(dirname $(pwd)) && echo $(python --version) &&
        python test_script.py"