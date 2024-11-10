#!/bin/bash
apptainer build -B $(dirname $(pwd)):/env-setup jax.sif ApptainerFile