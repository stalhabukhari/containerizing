"""
This serves as a training demo for JAX neural networks.
Taken from: https://github.com/stalhabukhari/JAX-DNN-MNIST/blob/main/JAX_DNN_MNIST.ipynb
"""
import time, itertools, argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax

from jax.lib import xla_bridge

print(f"JAX version: {jax.__version__}")
# GPU check
assert xla_bridge.get_backend().platform=='gpu' and jax.local_device_count()

num_classes = 10
reshape_args = [(-1, 28*28), (-1,)]
input_shape = reshape_args[0]

step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9
rng = random.PRNGKey(0)

# Training Data
ds_train = tfds.load('mnist', split='train', shuffle_files=True)
total_train_imgs = len(ds_train)
print(f'Training Set Images: {total_train_imgs}')
ds_train = ds_train.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds_train = tfds.as_numpy(ds_train)

# Testing Data
ds_test = tfds.load('mnist', split='test', shuffle_files=False)
total_test_imgs = len(ds_test)
print(f'Testing Set Images: {total_test_imgs}')
ds_test = ds_test.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = tfds.as_numpy(ds_test)

# {Dense(1024) -> ReLU}x2 -> Dense(10) -> LogSoftmax
init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(10), stax.LogSoftmax)

# Utility functions in jax

def one_hot_nojit(x, k, dtype=jnp.float32):
    """ Create a one-hot encoding of x of size k. """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
one_hot = jit(partial(one_hot_nojit, k=num_classes, dtype=jnp.float32))

@jit
def loss(params, batch):
    """ Cross-entropy loss over a minibatch. """
    inputs, targets = batch
    return jnp.mean(jnp.sum( -targets*predict(params, inputs), axis=1))

@jit
def pred_check(params, batch):
    """ Correct predictions over a minibatch. """
    inputs, targets = batch
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.sum(predicted_class == targets)

@jit
def update(i, opt_state, batch):
    """ Single optimization step over a minibatch. """
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)


class MetricAccumulator(object):
    """ Class for collecting and plotting training/testing metrics. """

    def __init__(self, metric_list):
        metric_dict = {}
        for met in metric_list:
            metric_dict[met] = []

        self.metric_dict = metric_dict

    def update(self, metric_dict):
        for met in self.metric_dict.keys():
            self.metric_dict[met].append(metric_dict[met])

    def plot(self, save_path=None):
        fig, ax = plt.subplots()
        fig.suptitle('Per-epoch Metrics')

        for met in self.metric_dict.keys():
            y_axis = self.metric_dict[met]
            x_axis = np.arange(len(y_axis))
            ax.plot(x_axis, y_axis, linewidth=2, label=met)

        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        ax.legend(fancybox=True)
        ax.grid()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_plot', action='store_true', default=False)
    args = parser.parse_args()
    
    # Initialize Network, Optimizer
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
    _, init_params = init_random_params(rng, input_shape)
    opt_state = opt_init(init_params)

    # Counter
    itercount = itertools.count()

    print("\nStarting training...\n")

    MetAcc = MetricAccumulator(metric_list=['TrainAcc', 'TestAcc'])

    for epoch in range(num_epochs):
        train_acc, test_acc = [], []

        # Training
        start_time = time.time()
        
        for batch_raw in ds_train:
            data = batch_raw["image"].reshape(*reshape_args[0])
            targets = one_hot(batch_raw["label"].reshape(*reshape_args[1]))
            opt_state = update(next(itercount), opt_state, (data, targets))
        
        epoch_time = time.time() - start_time
        print("\nEpoch {} in {:0.2f} sec".format(epoch+1, epoch_time))

        params = get_params(opt_state)

        # Train Acc
        correct_preds = 0.0
        for batch_raw in ds_train:
            data = batch_raw["image"].reshape(*reshape_args[0])
            targets = batch_raw["label"].reshape(*reshape_args[1])
            correct_preds += pred_check(params, (data, targets))
        train_acc.append(correct_preds/float(total_train_imgs))
        print(f"Training set accuracy: {train_acc}")
        
        # Test Acc
        correct_preds = 0.0
        for batch_raw in ds_test:
            data = batch_raw["image"].reshape(*reshape_args[0])
            targets = batch_raw["label"].reshape(*reshape_args[1])
            correct_preds += pred_check(params, (data, targets))
        test_acc.append(correct_preds/float(total_test_imgs))
        print(f"Test set accuracy: {test_acc}")

        MetAcc.update({'TrainAcc': train_acc, 'TestAcc': test_acc})
        
    # plotting
    if args.save_plot:
        MetAcc.plot(save_path='metrics.png')
    else:
        MetAcc.plot()
