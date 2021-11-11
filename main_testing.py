

import numpy as np

import logging

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from os import path as pth
import sys, os
import random
import copy
from sklearn.model_selection import train_test_split
from spn.data.metaData import *
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
from spn.algorithms.SPMN import SPMN
from spn.structure.Base import Sum
from spn.algorithms.MEU import meu
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.data.simulator import get_env
from spn.algorithms.MEU import best_next_decision
from spn.io.ProgressBar import printProgressBar
import numpy
import multiprocessing
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
from collections import Counter
import time
import pickle
from tqdm import tqdm






#TODO Not operating correctly. Does not catch all times when children have same weights
#Purely For Testing Purposes (Iterates through all children in a list of spmns printing similar weights
def child_parser(curr_node_list, depth = 0,counter=0):
    curr_node_parser = curr_node_list[0]
    if (not hasattr(curr_node_parser, "children")): return counter
    if (not curr_node_parser.children): return counter

    if isinstance(curr_node_parser, Sum):
        testList = curr_node_parser.weights
        for i in range(0,len(testList)):
            for x in curr_node_list[1:]:
                if x.weights[i] == testList[i]: counter=1+counter
                #print(x.weights[i],testList[i],counter)

    for i in range(0, len(curr_node_parser.children)):
        counter = child_parser([node.children[i] for node in curr_node_list],depth+1,counter)

    #print(counter)
    return counter

