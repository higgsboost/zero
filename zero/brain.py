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
from .neuron import *

def distanceTwoNodes(n1, n2):
    distance_ = np.linalg.norm(n1 - n2)
    return distance_


def sigmoid(x):
    return 1.0 / 1.0 + np.exp(-x)


def tanh(x):
    return np.tanh(x)



class brain:
    """
    Brian class that has neurons and synapses.
    """

    def __init__(
        self,
        num_neuron=0,
        num_synapse=0,
        neuronFields=dict(),
        synapseFields=dict(),
        neuron_radius=35,
        synapse_radius=10,
        brainId="default",
        block_limits=[-1, 1, -1, 1, -1, 1],
        neuron_index_offset=0,
        generate_neurons=True,
        is_input_block=False,
        is_output_block=False,
        is_reward_block=False,
        modifiable=True
    ):
        
        self.neuron_array = []

        self.brainId = brainId

        # Fields for neurons
        self.neuronFields = neuronFields.copy()
        # Fields for synapses
        self.synapseFields = synapseFields.copy()

        # This array maps neuron ids to indices in 'neuron_array'
        
        self.neuronMapper = dict()

        # generation radius
        self.neuron_radius = neuron_radius

        self.synapse_radius = synapse_radius

        self.neuron_index_offset = neuron_index_offset


        self.num_neuron = num_neuron
        self.num_synapse = num_synapse
        self.block_limits = block_limits

        self.is_input_block = is_input_block
        self.is_output_block = is_output_block
        self.is_reward_block = is_reward_block

        # If synapse is modifiable from the neuron
        self.modifiable = modifiable

        # todo figure out logic
        # self.modifierBlock = modifierBlock()

        # generate neuron
        logging.debug("Generating neurons ...")


        if generate_neurons:
            self.generate_neurons()

    def max_id(self):
        ''' returns maximum id for neuron | synapse '''
        return self.num_neuron * (self.num_synapse + 1) + self.neuron_index_offset

    def set_neuron_mapper(self, mapper):
        self.neuronMapper = mapper
        
    def generate_neurons(self):

        for i in range(self.num_neuron):

            neuron_id = i * (self.num_synapse + 1) + self.neuron_index_offset
            n_ = neuron(
                self.num_synapse,
                neuron_id,
                neuron_radius=self.neuron_radius,
                synapse_radius=self.synapse_radius,
                neuronFields=self.neuronFields,
                synapseFields=self.synapseFields,
                block_limits=self.block_limits,
                is_input=self.is_input_block,
                is_output=self.is_output_block,
                is_reward=self.is_reward_block,
                modifiable=self.modifiable
            )
            self.neuronMapper[neuron_id] = i

            # self.cmissIO.addNode(2,'0 1 1 2')
            self.neuron_array.append(n_)

    def get_output_values(self):
        output_array = []
        for n in self.getNeuronArray():
            if n.is_output:
                output_array.append(n.getField('a'))

        return output_array

    def set_reward(self, value=1000.0):
   
        for n in self.getNeuronArray():
            if n.is_reward:

                n.setField('a', value)

    def set_neuron_array(self, neuron_array):
        self.neuron_array = neuron_array

    def one_step_new(self):
        pass


    def one_step(self):
        # Main guy takes care of one step of t

        s_array = self.returnSynapses()

        # Loop through all synapses in brain
        for s_ in s_array:
            # print(s_.getSourceNucleusId())
            #import pdb; pdb.set_trace()
            sourceN_ = self.getNeuron(s_.getSourceNucleusId())
            targetN_ = self.getNeuron(s_.getTargetNucleusId())

            # in_syn = targetN_.get_input_synpase_array()

            # Loop through all source neurons and put value in synapse
            for key, val in sourceN_.getFieldDict().items():
                f_ = sourceN_.getField(key)
                s_.setField(key, f_)

        neuron_array = self.getNeuronArray()
        #logging.info('Modifiable : {}'.format(self.modifiable))
        for n_ in neuron_array:

            if n_.is_input:
                # If neuron is input then don't allow other inputs.
                logging.debug('Skipping because it is an input : {}'.format(n_.getId()))
                continue

            # Loop through input synapse
            in_syn = n_.get_input_synpase_array()
            in_syn_values = []
            for syn_ in in_syn:
                in_syn_values.append(syn_.getField("a"))
                # j=0
                # for key,val in sourceN_.getFieldDict().items():

                #   if j==0:

                #       #print('Source syn id : {}, with value {}'.format(syn_.getId(), syn_.getField(key)))

                #   j+=1

            # print('insyn values', in_syn_values)
            # print(in_syn_values)

            input_tf = tf.constant(
                in_syn_values, shape=(1, len(in_syn_values)), dtype=tf.float32
            )

            # print('Prev in ', self.prev_in)
            # print('Cur in ', in_syn_values)
            # self.prev_in = in_syn_values

            result_ = n_.apply_neuron_block(input_tf).numpy()


            mutate_neuron_block = False #TODO ARGS
            if mutate_neuron_block:
                n_.update_neuron_block_weights(in_syn_values)

            # print(n_.apply_neuron_block(input_tf).numpy()[0])

            for key, val in sourceN_.getFieldDict().items():
         
                n_.setField(key, float(result_))

    def findClosestPoint(self):
        # for each synapse find and move that point to the clo sest neuron (not parent)
        print("Finding closest points ...")
        for n_ in tqdm((self.neuron_array)):
            self.findClosestNeuronSynapse(n_)

    def findClosestNeuronSynapse(self, n_):
        already_visited_neurons = []
        for syn_ in n_.getSynapse():
            min_dist = np.inf
            target_neuron = n_
            for nn_ in self.neuron_array:
                if nn_.is_input: continue
                if n_ is not nn_ and nn_ not in already_visited_neurons:
                    dist_ = distanceTwoNodes(nn_.getPosition(), syn_.getPosition())
                    if dist_ < min_dist:

                        min_dist = dist_
                        target_neuron = nn_

            syn_.setPosition(target_neuron.getPosition())
            syn_.setTargetNucleusId(target_neuron.getId())

            target_neuron.append_input_synapse_array(syn_)

            # UNCOMMENT FOR MULTIPLE SYNPASE TO GO FROM A -> B
            already_visited_neurons.append(target_neuron)

    # returns the synapse
    def returnSynapses(self):
        s_array = []
        for n_ in self.neuron_array:
            for s_ in n_.getSynapse():
                s_array.append(s_)

        return s_array

    # get neuron given id
    def getNeuron(self, neuron_id):

        return self.neuron_array[self.neuronMapper[neuron_id]]

    # get neuron given id
    def getNeuronIndex(self, index):
        return self.neuron_array[index]

    # set neuron input bool
    def setIsInputIndex(self, index, is_input):
        self.neuron_array[index].setIsInput(is_input)

    def getNeuronArray(self):
        return self.neuron_array

    def get_attributes(self):
        
        attributes = {
            "neuron_array": [n.get_attributes() for n in self.neuron_array],
            "brainId": self.brainId,
            "neuronFields": self.neuronFields,
            "synapseFields": self.synapseFields,
            "neuronMapper": self.neuronMapper,
            "neuron_radius": self.neuron_radius,
            "synapse_radius": self.synapse_radius,
            "neuron_index_offset": self.neuron_index_offset,
            "num_neuron": self.num_neuron,
            "num_synapse": self.num_synapse,
            "block_limits": self.block_limits,
            "is_input_block": self.is_input_block,
            "is_output_block": self.is_output_block,
            "is_reward_block": self.is_reward_block,
            "modifiable": self.modifiable,

        }

        return attributes

        

    def save(self, location):
        data = self.get_attributes()
        # print(data)
        with open(location, 'w') as f:
            json.dump(data, f)


def brain_from_file(location):
    with open(location, 'r') as f:
        data = json.load(f)

    neuron_array = [neuron_from_json(d) for d in data['neuron_array']]

    b = brain(
        num_neuron=data['num_neuron'],
        num_synapse=data['num_synapse'],
        neuronFields=data['neuronFields'],
        synapseFields=data['synapseFields'],
        neuron_radius=data['neuron_radius'],
        synapse_radius=data['synapse_radius'],
        brainId=data['brainId'],
        block_limits=data['block_limits'],
        neuron_index_offset=data['neuron_index_offset'],
        generate_neurons=False,
        is_input_block=data['is_input_block'],
        is_output_block=data['is_output_block'],
        is_reward_block=data['is_reward_block'],
        modifiable=data['modifiable'],
    )

    b.set_neuron_array(neuron_array)
    b.modifiable = data['modifiable']
    mapper = data['neuronMapper']

    for k in mapper.keys():
        mapper[int(k)] = mapper.pop(k)

    b.set_neuron_mapper(data['neuronMapper'])

    return b





    
