from .brain import *
from .blocks import *




def add_to_weights(n, original_weights, weight_adds, shape_array):
    """ Adds numpy input to neuron weights """
    new_weights = np.array(original_weights) + np.array(weight_adds)

    set_weights_v2(n.neuronBlock.weights,  new_weights, shape_array)

