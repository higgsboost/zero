import tensorflow as tf
import numpy as np


import logging

import numpy as np
from tqdm import tqdm

import random
import time


import json

import pickle
import os
import sys


from .synapse import *
from .blocks import *

class neuron:



    def __init__(
        self,
        numSynapse,
        nucleus_id,
        neuron_radius,
        synapse_radius,
        neuronFields,
        synapseFields,
        block_limits=[-1, 1, -1, 1, -1, 1],
        is_input=False,
        is_output=False,
        is_reward=False,
        firing_value=500.0,
        predetermined_position=np.array([]),
        modifiable=True, # Modifiable weights

        loading_from_file = False,
        # loading_synapse_from_file=False,
        # loading_input_synapse_from_file=False,
        # loading_neuron_weights_from_file=False,
        

    ):
        # Number of output synapses
        tf.random.set_seed(int(time.time()))

        np.random.seed(int(time.time()))

        self.numSynapse = numSynapse
        self.nucleus_id = nucleus_id

        self.neuron_radius = neuron_radius
        self.synapse_radius = synapse_radius

        self.fields = neuronFields.copy()

        self.fields_array = dict()

        # Assign fields_array
        for key in self.fields:
            self.fields_array[key] = []

        self.synapseFields = synapseFields.copy()

        # Input fields
        self.inputField = []

        # output fields
        self.outputField = []

        # Bool to determine if input from another but it is only for debugging as there is no such thing as an input
        # 
        self.is_input = is_input
        self.is_output = is_output
        self.is_reward = is_reward

        self.modifiable = modifiable

        # generate position based on a specified radius
        if len(predetermined_position) == 0:
            self.position = self.neuron_radius * np.array(
                [
                    (random.uniform(block_limits[0], block_limits[1])),
                    (random.uniform(block_limits[2], block_limits[3])),
                    (random.uniform(block_limits[4], block_limits[5])),
                ]
            )
        else:
            self.position = predetermined_position

        # This is for the outputs
        self.synapse_array = []

        if loading_from_file is False:
            for i in range(numSynapse):
                self.synapse_array.append(
                    synapse(
                        synapse_id=self.nucleus_id + i + 1,
                        weight=1,
                        source_id=self.nucleus_id,
                        target_id=None,
                        position=self.position,
                        radius=self.synapse_radius,
                        synapseFields=self.synapseFields
                    )
                )

            
            

        else:
            pass # TODO
        

        if loading_from_file is False:

            self.input_synapse_array = []
        else:
            self.input_synapse_array = []


        if self.is_input:
            self.input_synapse_array.append(
                synapse(
                    synapse_id=self.nucleus_id + numSynapse + 1,
                    weight=1,
                    source_id=self.nucleus_id,
                    target_id=None,
                    position=self.position,
                    radius=self.synapse_radius,
                    synapseFields=self.synapseFields
                )
            ) 

        # if loading_from_file is False:
        #     self.neuronBlock = neuronBlock()
        # else:
        #     pass

        self.neuronBlock = neuronBlock()

        self.modifierBlock = modifierBlock()
        self.prev_tf_input = None

        self.current_weight_tf = None

        self.current_input_tf = None


        # Default firing value if not set
        self.firing_value = firing_value

    def apply_neuron_block(self, tf_input):
        # print('prev', self.prev_tf_input)
        # print('Cur', tf_input)
        self.prev_tf_input = tf_input
        return self.neuronBlock(tf_input)

    def update_neuron_block_weights(self, current_input):
        # print()
        current_weight = get_weights(self.neuronBlock.weights)

        # modifier_input = current_weight+current_input

        # input_tf = tf.constant(modifier_input, shape=(1, len(modifier_input)), dtype=tf.float32)
        if self.current_weight_tf is None:
            self.current_weight_tf = tf.constant(
                current_weight, shape=(1, len(current_weight)), dtype=tf.float32
            )

        if self.current_input_tf is None:
            self.current_input_tf = tf.constant(
                current_input, shape=(1, len(current_input)), dtype=tf.float32
            )
        # print(self.current_weight_tf)

        output_weight = self.modifierBlock(
            self.current_weight_tf, self.current_input_tf
        ).numpy()[0]  # [0]
        # print(output_weight.shape)
        # print(dir(self.modifierBlock))
        # print('current tf weight ', self.current_weight_tf.shape)
        # print('current_input_tf ', self.current_input_tf.shape)
        #print('outpout_wiehgt', output_weight.shape)
        # print('current weight', len(current_weight))
        set_weights(output_weight, self.neuronBlock.weights)

    def append_synapse_array(self, syn_):
        self.synapse_array.append(syn_)

    def append_input_synapse_array(self, syn_):
        self.input_synapse_array.append(syn_)

    def get_input_synpase_array(self):
        return self.input_synapse_array

    def getPosition(self):
        return self.position

    def getId(self):
        return self.nucleus_id

    def getSynapse(self):
        return self.synapse_array

    def setSynapse(self, new_synapse_array):
        self.synapse_array = new_synapse_array

    def returnFieldString(self):
        string_ = ""
        for key, val in self.fields.items():
            string_ += str(val) + " "

        return string_

    def set_neuron_block(self):
        self.neuronBlock = neuronBlock()

    # gets entire field dict
    def getFieldDict(self):
        return self.fields

    # returns field given name
    def getField(self, field_name):
        return self.fields[field_name]

    # returns field given name
    def setField(self, field_name, new_value):
        self.fields[field_name] = new_value

    # adds new value to a given field
    def addField(self, field_name, add_value):
        self.fields[field_name] += add_value

    # Similar with add field but we append value to array
    def appendField(self, field_name, add_value):
        self.fields_array[field_name].append(add_value)

    # This function can apply functions to input fields.
    # This function is what should be evolved, and is very important.
    # def applyFunction(self, field_name, function_type=0):
    #     if function_type == 0:
    #         self.fields[field_name] = sigmoid(self.fields[field_name])

    # def applyFunctionArray(self, field_name, array_function_type=0):
    #     if array_function_type == 0:
    #         print(
    #             "Neuron {} has input field size {} ".format(
    #                 self.nucleus_id, len(self.fields_array[field_name])
    #             )
    #         )
    #         self.fields[field_name] = sigmoid(np.sum(self.fields_array[field_name]))

    def appendInputField(self, input_):
        self.inputField.append(input_)

    def clearInputField(self, input_):
        self.inputField = []

    def setIsInput(self, input_, firing_value_=2000.0):
        self.is_input = input_
        self.firing_value = firing_value_

    def setIsOutput(self, output_):
        self.is_output = output_

    def setFiringValue(self, firing_value_):
        self.firing_value = firing_value_

    def getWeights(self):
        """ wrapper around get_weights from .blocks """
        w,s =get_weights(self.neuronBlock.weights)

        total_length = 0

        for s_ in s:
            l,_ = shape_length(s_)
            total_length += l
        
        return w,s,total_length


    def get_attributes(self):

        weights, shapes = get_weights(self.neuronBlock.weights)

        # Convert np -> float
        weights = [float(w) for w in weights]
        
        attributes =  {
            "numSynapse": self.numSynapse ,
            "nucleus_id": self.nucleus_id ,
            "neuron_radius": self.neuron_radius ,
            "synapse_radius":self.synapse_radius,
            "fields": self.fields ,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "is_reward": self.is_reward,
            "modifiable": self.modifiable,
            "neuronFields":self.fields,
            "synapseFields":self.synapseFields,
            "position": list(self.position),
            "synapse_array": [s.get_attributes() for s in self.synapse_array],
            "input_synapse_array": [s.get_attributes() for s in self.input_synapse_array],
            "neuron_weights": weights,
            "neuron_weights_shapes": shapes,
            "firing_value": self.firing_value

            
        }

        
        # import pdb;pdb.set_trace()

        
        return attributes
    def save(self, location):
        data = self.get_attributes()
        # print(data)
        with open(location, 'w') as f:
            json.dump(data, f)



def neuron_from_json(data):
    synapse_array = [synapse_from_json(d) for d in data['synapse_array']]
    input_synapse_array = [synapse_from_json(d) for d in data['input_synapse_array']]


    n = neuron(
        numSynapse=data['numSynapse'],
        nucleus_id=data['nucleus_id'],
        neuron_radius=data['neuron_radius'],
        synapse_radius=data['synapse_radius'],
        neuronFields=data['neuronFields'],
        synapseFields=data['synapseFields'],
        is_input=data['is_input'],
        is_output=data['is_output'],
        is_reward=data['is_reward'],
        firing_value=data['firing_value'],
        predetermined_position=data['position'],
        modifiable=data['modifiable'],
        loading_from_file = True,
    )



    _ = [n.append_input_synapse_array(s) for s in input_synapse_array]
    _ = [n.append_synapse_array(s) for s in synapse_array]

    # load weights for block

    weights = data['neuron_weights']
    weights_shapes = data['neuron_weights_shapes']

    data_array = list(range(0, weights_shapes[0][0]))
    input = tf.constant(
            data_array, shape=(1, len(data_array)), dtype=tf.float32
        )

    n.apply_neuron_block(input).numpy()
    
    set_weights_v2(n.neuronBlock.weights, weights, weights_shapes)

    return n


def neuron_from_file(location):
    with open(location, 'r') as f:
        data = json.load(f)

    n = neuron_from_json(data)
    
    return n
