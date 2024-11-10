#!/bin/bash
apptainer run --nv -B "$(dirname $(pwd))":"/code-dir" jax.sif "cd /code-dir && python test_script.py"