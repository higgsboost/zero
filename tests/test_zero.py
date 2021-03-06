import unittest

import numpy as np

from .context import zero

import logging
import copy
logging.getLogger().setLevel(logging.INFO)
class TestZero(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        num_neuron = 10
        b1 = zero.brain.brain(
            num_neuron=num_neuron,
            num_synapse=3,
            neuronFields={'a':0.0,'b':0.0},
            synapseFields={'a':0.0,'b':0.0},
            neuron_radius = 1,
            synapse_radius=1.4,
            block_limits=[-1,1,-1,1,-1,1],
            neuron_index_offset=0
            )
        b1.findClosestPoint()
        b1.one_step()
        # b1.one_step()
        # b1.one_step()
        # b1.one_step()

        num_inputs = 5
        self.num_inputs = num_inputs
        for i in range(0, num_inputs):
            b1.getNeuronArray()[i].is_output=True


        for i in range(num_neuron-num_inputs, num_neuron):
            b1.getNeuronArray()[i].is_input=True

        #b1.getNeuronArray()[3].is_input=True
        #import pdb; pdb.set_trace()

        b1.save('/tmp/test_zero')
        b1.save('/tmp/test_zero_get_to_the_point')

        self.original_brain = b1
        
        a=  b1.returnSynapses()             

        z = b1.getNeuronArray()[0].input_synapse_array[0] in a
        #import pdb;pdb.set_trace()

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

    @unittest.skip("skipping for now")
    def test_get_to_the_point(self):
        # Test if multiple outputs goes to one

        b12 = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')


        def get_pop(p):
            pop = []
            for i in range(20):
                p = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')
                zero.zero.add_noise_to_brain(p, 0.1)
                pop.append(p)
            return pop

        num_steps = 1

        pop = get_pop(b12)

        target = 0.8

        input_size = self.num_inputs
        for _ in range(200):
            diff_array = []
            for p in pop:
                diff_array_pop = []
                for i in range(1):
                    input_array = [0.0] *input_size

                    on_index = 0#np.random.randint(0, input_size)
                    input_array[int(on_index)] = 2.0
                    
                    
                    for s in range(num_steps):
                        if 0:

                            input_neurons = [n_ for n_ in p.getNeuronArray() if n_.is_input is True]
                    
 
                            for v_i, v in enumerate(input_array):

                                input_neurons[v_i].setField("a", v)
               
                        
            
                        
                        p.one_step()
                        out = p.get_output_values()

                        out = np.argmax(out)
                        
                        #print('{} + {} = {} ? '.format(a, b, out))

                        
                    #import pdb; pdb.set_trace()
                    if out==on_index:
                        diff =0 
                    else:
                        diff =1
                    diff_array_pop.append(diff)
                print(diff_array_pop)
                diff_array.append(np.sum(diff_array_pop))
            print('diff array ', diff_array)
            print('min', np.min(diff_array))
            print('total sum', np.sum(diff_array))
            #import pdb; pdb.set_trace()
            pop[np.argmin(diff_array)].save('/tmp/test_zero_get_to_the_point')

            pop = get_pop(pop[np.argmin(diff_array)])

   

    @unittest.skip("skipping for now")
    def test_get_to_the_point_with_time(self):
        # Test if multiple outputs goes to specified

        b12 = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')
        #b22 = zero.brain.brain_from_file('/tmp/test_zero')

        #assert(b1.get_attributes() == b2.get_attributes())

        #import pdb;pdb.set_trace()
        #print(b12.neuronMapper)

        def get_pop(p):
            pop = []
            for i in range(20):
                p = zero.brain.brain_from_file('/tmp/test_zero_get_to_the_point')
                zero.zero.add_noise_to_brain(p, 0.01)
                pop.append(p)
            return pop

        num_steps = 1

        pop = get_pop(b12)

        target = 0.8

        input_size = self.num_inputs
        while 1:
            diff_array = []
            for p in pop:
                diff_array_pop = []
                for i in range(10):
                    input_array = [0.0] *input_size

                    on_index = np.random.randint(0, input_size)
                    #input_array[int(on_index)] = 2.0
                    
                    
                    for s in range(num_steps):
                        if 1:

                            input_neurons = [n_ for n_ in p.getNeuronArray() if n_.is_input is True]
                    
 
                            for v_i, v in enumerate(input_array):
                                input_neurons[v_i].setField("a", -100)

                            input_neurons[on_index].setField("a", 200)
               
                        
            
                        
                        for _ in range(1): p.one_step()
                        out = p.get_output_values()

                        out = np.argmax(out)
                        
                        #print('{} + {} = {} ? '.format(a, b, out))

                        
                    #import pdb; pdb.set_trace()
                    #print('out', out)
                    if out==on_index:
                        diff =0 
                    else:
                        diff =1
                    diff_array_pop.append(diff)
                #print(diff_array_pop)
                diff_array.append(np.sum(diff_array_pop))
            print('diff array ', diff_array)
            print('min', np.min(diff_array))
            print('total sum', np.sum(diff_array))
            if np.sum(diff_array) == 0:
                break
            #import pdb; pdb.set_trace()
            pop[np.argmin(diff_array)].save('/tmp/test_zero_get_to_the_point')

            pop = get_pop(pop[np.argmin(diff_array)])

   
        


        
        


if __name__ == '__main__':
    unittest.main(verbosity=1)