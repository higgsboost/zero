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
        

        neuron.save('/tmp/test_neuron.json')

    def test_neuron(self):

        neuron = zero.neuron.neuron_from_file('/tmp/test_neuron.json')
        # neuron = zero.neuron.neuron(
        #     numSynapse=100,
        #     nucleus_id=1,
        #     neuron_radius=1,
        #     synapse_radius=1,
        #     neuronFields={'a':0},
        #     synapseFields={'a':0},
        # )

        # print('-----', neuron)

        #import pdb; pdb.set_trace()




if __name__ == '__main__':
    unittest.main(verbosity=1)