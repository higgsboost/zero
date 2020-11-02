import unittest

import numpy as np

from .context import zero

import logging
logging.getLogger().setLevel(logging.INFO)
class TestBrain(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
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

        self.original_brain = b1

    def test_brain(self):
        loaded_brain = zero.brain.brain_from_file('/tmp/test_brain')

        for k in self.original_brain.get_attributes().keys():
            # print('key', k)
            # print('-before', self.original_brain.get_attributes()[k])
            # print('-after', loaded_brain.get_attributes()[k])
            eq = (self.original_brain.get_attributes()[k] == loaded_brain.get_attributes()[k])
            assert(eq == True)
            #import pdb; pdb.set_trace()

        assert(self.original_brain.get_attributes() == loaded_brain.get_attributes())
        


if __name__ == '__main__':
    unittest.main(verbosity=1)