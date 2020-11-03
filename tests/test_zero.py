import unittest

import numpy as np

from .context import zero

import logging
logging.getLogger().setLevel(logging.INFO)
class TestZero(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        b1 = zero.brain.brain(
            num_neuron=20,
            num_synapse=15,
            neuronFields={'a':0.0,'b':0.0},
            synapseFields={'a':0.0,'b':0.0},
            neuron_radius = 1,
            synapse_radius=1.5,
            block_limits=[-1,1,-1,1,-1,1],
            neuron_index_offset=0
            )
        b1.findClosestPoint()
        b1.one_step()

        b1.getNeuronArray()[0].is_output=True
        b1.getNeuronArray()[1].is_output=True
        b1.getNeuronArray()[2].is_input=True
        b1.getNeuronArray()[3].is_input=True
        #import pdb; pdb.set_trace()

        b1.save('/tmp/test_zero')
        b1.save('/tmp/test_zero_get_to_the_point')

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


    def test_mutate_brain_with_noise(self):
        
        # for i in range(100):
        #     print(i)
        b1 = zero.brain.brain_from_file('/tmp/test_zero')
        b2 = zero.brain.brain_from_file('/tmp/test_zero')
        zero.zero.add_noise_to_brain(b1, 0)

        assert(b1.get_attributes() == b2.get_attributes())

        zero.zero.add_noise_to_brain(b1, 1)

        assert(b1.get_attributes() != b2.get_attributes())

        
    def test_get_to_the_point(self):

        b12 = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')
        #b22 = zero.brain.brain_from_file('/tmp/test_zero')

        #assert(b1.get_attributes() == b2.get_attributes())

        #import pdb;pdb.set_trace()
        #print(b12.neuronMapper)

        def get_pop():
            pop = []
            for i in range(5):
                p = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')
                zero.zero.add_noise_to_brain(p, 0.1)
                pop.append(p)
            return pop

        num_steps = 2

        pop = get_pop()

        target = 0.8
        while True:
            diff_array = []
            for p in pop:
                for s in range(10):
                    p.one_step()
                    out = p.get_output_values()[0]
                    
                    print('out', out)

                    if s % 10 == 0:

                        input_neurons = [n_ for n_ in p.getNeuronArray() if n_.is_input is True]
                        #print(input_neurons)
                        for input_neuron in input_neurons:
                            input_neuron.setField("a", np.random.randint(1,100))


                diff_array.append(np.absolute(out-target))
            print('loss', np.sum(diff_array))
            pop[np.argmin(diff_array)].save('/tmp/test_zero_get_to_the_point')

            pop = get_pop()

            
                    
        # import ray
        # ray.init()

        # import multiprocessing as mp

        # a = []
        # for _ in range(1000):
        #     p = mp.Process(target=zero.brain.brain_from_file, args=('/tmp/test_zero',))
        #     p.start()
        #     a.append(p)

        # for p in a:
        #     p.join()
        



        


        
        


if __name__ == '__main__':
    unittest.main(verbosity=1)