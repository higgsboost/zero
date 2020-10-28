import unittest

import numpy as np

from .context import zero

class TestSynapse(unittest.TestCase):
    def test_setup(self):
        
        syn = zero.synapse.synapse(1, None, 0, 1, np.array([0,0,0]), 1, {'a':0.0,'b':0.0})

        syn.save('/tmp/test_syn_4123')

    def test_loader(self):
        zero.synapse.synapse_from_file('/tmp/test_syn_4123')



   




if __name__ == '__main__':
    unittest.main(verbosity=1)