
import numpy as np

import logging

from cspn import buildSPN, learnCSPNs

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
from cspmn import learnCSPMNs
from cspmn import buildSPMN, credal_best_next_decision




class caSpn():

    #built from SPMN
    def __init__(self, dataset, number_of_sets=2,vers=["naive","naive"] ,number_of_credals=10, weight=[42]):
        self.dataset = dataset
        self.cspmns = []
        self.number_of_sets = number_of_sets
        self.number_of_credals = number_of_credals
        self.vers = vers
        if weight[0]==42:self.weight = [.5*number_of_credals for x in range(number_of_sets)]
        else: self.weight = weight

        #spmns = self.buildSpmns()
        #self.sets = self.credalize(spmns)

    """
    inputs: force_make_new if true will forcibly generate new spmns rather than reading in pickles
    Output: will train the "CASPMN" model via sets of credalized SPMNs/CSPMNs
    """
    def learn(self,force_make_new=False):

        if not force_make_new:
            if self.getCascading(): return None

        spns = self.buildSpns()

        self.spns = spns.copy()

        with open("models/non_credal_spns.pickle", "wb") as f:
            pickle.dump(spns, f)

        self.sets = self.credalize()

        with open("models/credal_spns.pickle","wb") as f:
            pickle.dump(self.sets,f)


    """
    Input: Structure learned SPMN
    Output: List of Credalized SPMNs (through cspmn.py)
    """
    def learner(self,spn):
        curr_node_list = [copy.deepcopy(spn) for x in range(self.number_of_credals)]
        return learnCSPNs(curr_node_list)

    """
    Input: List of SPMNs 
    Output: a List of Sets of Credalized SPMNs
    """
    def credalize(self):
        sets = []
        for i in range(len(self.spns)):
            sets.append(self.learner(self.spns[i]))
        return sets

    """
    TODO move this?
    Input: list of datasets
    Output: List of built SPMNs
    """
    def buildSpns(self):
        spn_bucket = []
        for i in tqdm(range(self.number_of_sets)):
            spn_bucket.append(buildSPN(self.dataset))

        return spn_bucket

    """
    Retrieve the credal SPMNs if they already exist
    """
    def getCascading(self):
        if(not os.path.exists("credal_spns.pickle")): return False
        with open(f"credal_spns.pickle", "rb") as file:
            self.sets = pickle.load(file)
        return True

    """
    Implementation of the "Best Next Decision" function of SPMNs for CASPMNs
    Returns (for now) the first decision in the set with uncertainty greater than the "weight value" else returns the first decision, each decision, and the credal values of the decisions
    TODO consider changing this to the decision with the greates robustness...
    """
