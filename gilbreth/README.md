# Gilbreth

After following the [instructions for apptainer](../apptainer/README.md), a slurm job (e.g., [jax-job.sub](jax-job.sub)):

```shell
#!/bin/bash
sbatch -A <ACCOUNT-NAME> --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gpus-per-node=1 --time=0:05:00 jax-job.sub
```

### External References:

* [Gilbreth Knowledge Base](https://www.rcac.purdue.edu/knowledge/gilbreth/run)
