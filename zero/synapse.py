
"""
A<->B: worm is the "<->" a connection
"""

import logging

import numpy as np
from tqdm import tqdm

import random
import time


import json

import pickle
import os
import sys


# from opencmissIO import opencmissIO
# from neuron_math import distanceTwoNodes, sigmoid

# from tf_layer import *



class synapse:
    def __init__(
        self, synapse_id, weight, source_id, target_id, position, radius=1.0, synapseFields={'a':0}, generate_position=True
    ):

        self.synapse_id = synapse_id
        self.weight = weight
        self.source_id = source_id
        self.target_id = target_id

        # initially it will be dangling 
        if generate_position:
            self.position = position + radius * (
                np.array(
                    [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
                )
            )
        else:
            self.position = position

        self.ds1 = np.array(
            [
                random.uniform(-radius, radius),
                random.uniform(-radius, radius),
                random.uniform(-radius, radius),
            ]
        )

        self.fields = synapseFields
        logging.debug("Generating synapse")
    def getId(self):
        return self.synapse_id

    def getPosition(self):
        return self.position

    def setPosition(self, updated_position):
        self.position = updated_position

    def getSourceNucleusId(self):
        return self.source_id

    def getTargetNucleusId(self):
        return self.target_id

    def setTargetNucleusId(self, target_id):
        self.target_id = target_id

    # gets entire field dict
    def getFieldDict(self):
        return self.fields

    # returns field given name
    def getField(self, field_name):
        return self.fields[field_name]

    # returns field given name
    def setField(self, field_name, new_value):
        self.fields[field_name] = new_value

    def returnFieldString(self):
        string_ = ""
        for key, val in self.fields.items():
            string_ += str(val) + " "

        return string_

    def getds1(self):
        return self.ds1

    def get_attributes(self):
        return {
            "synapse_id": self.synapse_id ,
            "weight": self.weight ,
            "source_id": self.source_id ,
            "target_id": self.target_id ,
            "position": list(self.position)
        }
    def save(self, location):

        data = self.get_attributes()
        logging.debug("saving to {} \ndata: {}".format(location, data))
        # initially it will be dangling 
        #self.position 

        with open(location, 'w') as f:
            json.dump(data, f)



def synapse_from_json(data):
    temp = synapse(
        synapse_id=data['synapse_id'],
        weight=data['weight'],
        source_id=data['source_id'],
        target_id=data['target_id'],
        position=np.array(data['position']),
        generate_position=False
    )   
    return temp



def synapse_from_file(location):
    with open(location, 'r') as f:
        data = json.load(f)

    temp = synapse_from_json(data)
    
    logging.debug("Loading synapse from : {}\ndata:{}".format(location, data))
    return temp
