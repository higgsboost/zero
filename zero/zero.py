from .brain import *
from .blocks import *




def add_to_weights(n, original_weights, weight_adds, shape_array):
    """ Adds numpy input to neuron weights """
    new_weights = np.array(original_weights) + np.array(weight_adds)

    set_weights_v2(n.neuronBlock.weights,  new_weights, shape_array)

def add_noise_to_brain(b, r):
    """ Add noise with min=-r and max=r """
    num_neuron = len(b.getNeuronArray()) # RUN ALL

    for n_id in range(num_neuron):
        
        if np.random.randint(0, 100) > 50: continue
        n = b.getNeuronArray()[n_id]

        w, s, l = n.getWeights()   

        #total_length = sum([t[0]*t[1] for t in s])
        weight_to_add = np.random.uniform(-r, r, l) 
        
        add_to_weights(n, w, weight_to_add, s)

        

def generate_population():
    pass

    

