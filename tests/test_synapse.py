import unittest

import numpy as np

from .context import src

class TestSynapse(unittest.TestCase):
    def test_1(self):
        
        syn = src.synapse(False, 1, None, 0, 1, np.array([0,0,0]), 1, {'a':0.0,'b':0.0})


   




if __name__ == '__main__':
    unittest.main(verbosity=1)