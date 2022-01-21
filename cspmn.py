import math

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
from main_testing import child_parser

#TODO Dynamic recursion limit?  How is this handeled by Swaraj?
sys.setrecursionlimit(1000000000)


class contaminator():
    def __init__(self):
        self.rng = numpy.random.default_rng()

    #random "contamination" not really "e" contamination yet

    def e_contam(self, node,rangeW):
        # contaminate them
        print(node.weights,random.uniform(rangeW[0],rangeW[1]))
        node.weights = self.rng.dirichlet(alpha=[random.uniform(rangeW[0],rangeW[1]) for x in node.weights])
        return node.weights
cont = contaminator()


#parses tree finding minimum and maximum weights
def fixWeightRange(curr_node_list=[],bias=0,minW=0,maxW=0):
    curr_node_parser = curr_node_list[0]
    if (not hasattr(curr_node_parser, "children")): return True
    if (not curr_node_parser.children): return True

    if isinstance(curr_node_parser, Sum):

        for curr_node in curr_node_list:

            if min(curr_node.weights) < minW: minW = min(curr_node.weights)
            if max(curr_node.weights) > maxW: maxW = max(curr_node.weights)

    for i in range(0, len(curr_node_parser.children)):
        # print(f"learning child {i}")
        fixWeightRange([node.children[i] for node in curr_node_list],bias,minW,maxW)

    #quick test
    #this can go negative because the dirchlet distribution will normalize it.
    print(minW,bias)
    return max(0,minW-bias),min(maxW+bias,100)







def learnCSPMNs(curr_node_list=[],rangeW=[1,100]):
    curr_node_parser = curr_node_list[0]
    if(not hasattr(curr_node_parser,"children")): return True
    if( not curr_node_parser.children): return True
    #print(curr_node_parser.children)
    #if(hasattr(curr_node_parser,"weights")): print(curr_node_parser.weights)
    if isinstance(curr_node_parser,Sum):
        #print("prior",curr_node_parser.weights)
        for curr_node in curr_node_list:
            #print("prior",curr_node.weights)

            #TODO FIX THIS Duplication bug* ?

            #equal normalized weights
            #curr_node.weights = [1/len(curr_node.weights) for x in curr_node.weights]


            #Random normalized weights
            curr_node.weights=cont.e_contam(curr_node,rangeW)


            #Make one random node be 1.0 weighted and the rest 0
            #============================================================
            #curr_node.weights=[0 for x in curr_node.weights]
            #curr_node.weights[random.randint(0,len(curr_node.weights)-1)]=1.0
            #==============================================================
            #print("post",curr_node.weights)

        #print("post",curr_node_parser.weights)
    #Old method for single CSPMN generation
    #for child_node in curr_node.children: learnCSPMNs(child_node,n)

    for i in range(0,len(curr_node_parser.children)):
        #print(f"learning child {i}")
        learnCSPMNs([node.children[i] for node in curr_node_list],rangeW)

    return curr_node_list

def learner(spmn, n=10,bias=0):
    curr_node_list = [copy.deepcopy(spmn) for x in range(n)]
    rangeW = fixWeightRange(curr_node_list,bias)
    return learnCSPMNs(curr_node_list,rangeW)




def buildSPMN(dataset,ver):
    partial_order = get_partial_order(dataset)
    utility_node = get_utilityNode(dataset)
    decision_nodes = get_decNode(dataset)
    feature_names = get_feature_names(dataset)
    feature_labels = get_feature_labels(dataset)
    meta_types = [MetaType.DISCRETE] * (len(feature_names) - 1) + [MetaType.UTILITY]

    df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')

    df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence

    data = df.values
    train, test = data, np.array(random.sample(list(data), int(data.shape[0] * 0.02)))

    # print("Start Learning...")
    spmn = SPMN(partial_order, decision_nodes, utility_node, feature_names, meta_types,
                cluster_by_curr_information_set=True, util_to_bin=False,ver=ver)

    spmn = spmn.learn_spmn(train)

    return spmn


def credal_best_next_decision(cspmn_list,state):
    decisions = {}

    for cspmn in cspmn_list:
        spmn_output = best_next_decision(cspmn, state)
        action = spmn_output[0][0]

        if action in decisions:
            decisions[action] = decisions[action] + 1
        else:
            decisions[action] = 1

    dominant_action = None
    credal_value = 0
    curr_state_decisions = decisions
    for x in curr_state_decisions:
        if curr_state_decisions[x] > credal_value:
            dominant_action = x
            credal_value = curr_state_decisions[x]


    return dominant_action,credal_value





