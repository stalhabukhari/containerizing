import numpy as np
import jax


if __name__ == '__main__':
    print("***** Docker test script *****")
    print("Numpy:", np.__version__)
    print("Jax:", jax.__version__)
    
    from jax.lib import xla_bridge
    gpu_check = xla_bridge.get_backend().platform=='gpu' and jax.local_device_count()
    if gpu_check:
        print("GPU check passed!")
    else:
        print("GPU check failed!")
    
    print("Script ran successfully!")
