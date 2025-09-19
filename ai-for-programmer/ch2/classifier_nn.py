#!/usr/bin/env python
# encoding: utf-8
from random import random


## @n_inputs: number of neurons in input layer.
## @n_hidden: number of neurons in hidden layer.
## @n_outputs: numbers of neurons in output layer
def init_network(n_input, n_hidden, n_output):
    network = list()
    ## Actually, there is an 2-dimensional matrix in each hidden layer and output layer.
    hidden_layer = [ { 'weights' : [ random() for i in range(n_input + 1) ] } for i in range(n_hidden) ]
    output_layer = [ { 'weights' : [ random() for i in range(n_hidden) ] } for i in range(n_output) ]
    network.append(hidden_layer)
    network.append(output_layer)
    return network
