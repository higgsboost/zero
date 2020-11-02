import unittest

import numpy as np

from .context import zero

import logging
logging.getLogger().setLevel(logging.INFO)
class TestZero(unittest.TestCase):


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

        b1.save('/tmp/test_zero')

        self.original_brain = b1

    def test_mutate_individual_neuron(self):

        b = zero.brain.brain_from_file('/tmp/test_zero')

        
        num_neuron = len(b.getNeuronArray()) # RUN ALL

        for n_id in range(num_neuron):

            n = b.getNeuronArray()[n_id]

            w, s, l = n.getWeights()   

            #total_length = sum([t[0]*t[1] for t in s])
            weight_to_add = np.random.rand(l) 
            
            zero.zero.add_to_weights(n, w, weight_to_add, s)

            # Actual new weights
            actual_weights = np.array(w) + weight_to_add

            assert(np.mean(np.array(n.get_attributes()['neuron_weights']) -  actual_weights)<0.0000001)




        

        
        


if __name__ == '__main__':
    unittest.main(verbosity=1)