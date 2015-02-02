#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/2/2015
#
# Requires:
#   * matplotlib

import matplotlib.pyplot as plt

def fsc_ssc(data):
    '''Plot FSC v SSC.

    data - NxD numpy array (row=event), 1st column=FSC, 2nd column=SSC'''
    
    plt.scatter(data[:,0], data[:,1], marker='.')
    plt.axis([0,1023,0,1023])
    plt.xlabel('FSC')
    plt.ylabel('SSC')
    plt.show()
