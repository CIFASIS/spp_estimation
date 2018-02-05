import numpy as np
import lasagne


def save_network_values(network, filepath):
    np.savez(filepath, *lasagne.layers.get_all_param_values(network))


def load_network_values(network, filepath):
    with np.load(filepath) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

