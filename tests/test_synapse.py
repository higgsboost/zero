import unittest

import numpy as np

from .context import zero

import logging
logging.getLogger().setLevel(logging.DEBUG)
class TestSynapse(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        syn = zero.synapse.synapse(1, None, 0, 1, np.array([0,0,0]), 1, {'a':0.0,'b':0.0})

        syn.save('/tmp/test_syn_4123')

        self.syn = syn
 

    def test_loader(self):
        syn_loaded = zero.synapse.synapse_from_file('/tmp/test_syn_4123')
        assert(syn_loaded.get_attributes() == self.syn.get_attributes())




if __name__ == '__main__':
    unittest.main(verbosity=1)