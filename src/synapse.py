
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

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
logging.getLogger().setLevel(logging.DEBUG)

class synapse:
    def __init__(
        self, load_from_file, synapse_id, weight, source_id, target_id, position, radius, synapseFields
    ):

        self.synapse_id = synapse_id
        self.weight = weight
        self.source_id = source_id
        self.target_id = target_id

        # initially it will be dangling 
        self.position = position + radius * (
            np.array(
                [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
            )
        )

        self.ds1 = np.array(
            [
                random.uniform(-radius, radius),
                random.uniform(-radius, radius),
                random.uniform(-radius, radius),
            ]
        )

        self.fields = synapseFields

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

    def save(self, location):
        pass
        

        