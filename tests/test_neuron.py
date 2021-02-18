import unittest

import numpy as np

from .context import zero


import tensorflow as tf
import logging

logging.getLogger().setLevel(logging.INFO)
class TestNeuron(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        neuron = zero.neuron.neuron(
            numSynapse=100,
            nucleus_id=1,
            neuron_radius=1,
            synapse_radius=1,
            neuronFields={'a':0},
            synapseFields={'a':0},
        )
        
        data_array = [1,2,3,4]
        input = tf.constant(
                data_array, shape=(1, len(data_array)), dtype=tf.float32
            )

        # This will be used to test consistency
        self.neuron_input = input 

        result_ = neuron.apply_neuron_block(input).numpy()
        self.result = result_
        #import pdb; pdb.set_trace()

        print('original weights : ', neuron.neuronBlock.weights)

        neuron.save('/tmp/test_neuron.json')

        self.original_neuron = neuron

    def test_neuron(self):

        neuron = zero.neuron.neuron_from_file('/tmp/test_neuron.json')

        for k in self.original_neuron.get_attributes().keys():
            #if k == 'neuron_weights': import pdb; pdb.set_trace()
            assert(self.original_neuron.get_attributes()[k] == neuron.get_attributes()[k])


if __name__ == '__main__':
    unittest.main(verbosity=1)