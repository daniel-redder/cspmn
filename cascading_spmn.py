from main import learner, getSPMN

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




class caSpmn():

    #built from SPMN
    def __init__(self, dataset, number_of_sets=4, number_of_credals=10, weight=[42]):
        self.dataset = dataset
        self.cspmns = []
        self.number_of_sets = number_of_sets
        self.number_of_credals = number_of_credals

        if weight[0]==42:self.weight = [.5*number_of_credals for x in number_of_sets]
        else: self.weight = weight

        spmns = self.buildSpmns()
        self.sets = self.credalize(spmns)


    def credalize(self,spmns):
        sets = []
        for i in range(spmns):
            sets.append(learner(spmns[i],self.number_of_credals))


    def buildSpmns(self):
        spmn_bucket = []
        for i in tqdm(range(self.number_of_sets)):
            spmn_bucket.append(getSPMN(self.dataset))

        return spmn_bucket


    def cascading_best_next_decision(self,state):
        credal_values = []
        dominant_decisions = []
        for cspmn_list in self.sets:
            decision, value = self.credal_best_next_decision(cspmn_list,state)
            dominant_decisions.append(decision)
            credal_values.append(value)

        for x in range(len(credal_values)):
            if(credal_values[x] >= self.weight[x]): return dominant_decisions[x]

        #TODO for reality this would be replaced with a default decision, or alternative decision system
        return dominant_decisions[0]


    def credal_best_next_decision(self,cspmn_list,state):
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

























