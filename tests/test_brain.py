import unittest

import numpy as np

from .context import zero

import logging
logging.getLogger().setLevel(logging.INFO)
class TestBrain(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        pass

    def test_brain(self):
        b1 = zero.brain.brain(
            num_neuron=50,
            num_synapse=20,
            neuronFields={'a':0.0,'b':0.0},
            synapseFields={'a':0.0,'b':0.0},
            neuron_radius = 3,
            synapse_radius=1.5,
            block_limits=[-1,1,-1,1,-1,1],
            neuron_index_offset=0
            )
        b1.findClosestPoint()
        b1.one_step()
        #import pdb; pdb.set_trace()

        b1.save('/tmp/test_brain')




if __name__ == '__main__':
    unittest.main(verbosity=1)