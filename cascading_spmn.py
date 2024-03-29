import json

import numpy as np

import logging

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')

from copy import deepcopy
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
from cspmn import learner
from cspmn import buildSPMN, credal_best_next_decision




class caSpmn():

    #built from SPMN
    def __init__(self, dataset, number_of_sets=2,vers=["naive","naive"] ,number_of_credals=10, weight=[42],bias=0):
        self.dataset = dataset
        self.cspmns = []
        self.number_of_sets = number_of_sets
        self.bias = bias
        self.number_of_credals = number_of_credals
        self.vers = vers
        self.rangeW = []
        self.origin=[]
        if weight[0]==42:self.weight = [.5*number_of_credals for x in range(number_of_sets)]
        else: self.weight = weight

        #spmns = self.buildSpmns()
        #self.sets = self.credalize(spmns)


    def learn(self,force_make_new=False):

        if not force_make_new:
            if self.getCascasing(): return None

        spmns = self.buildSpmns()

        self.spmns = spmns.copy()

        with open("models/non_credal_spmns.pickle", "wb") as f:
            pickle.dump(spmns, f)

        self.sets = self.credalize(spmns)
        print("set length: ",len(self.sets))
        with open("credal_spmns.pickle","wb") as f:
            pickle.dump(self.sets,f)

    def credalize(self,spmns):
        sets = []
       
        for i in range(len(spmns)):
            learnedSPMN, count, origin = learner(spmns[i],self.number_of_credals,self.bias*(i))
            sets.append(learnedSPMN)

            #Adds original values for std calc
            self.origin=origin


            #TODO Remove rangeW deprecated
            self.rangeW.append(self.bias)
        return sets

    def buildSpmns(self):
        spmn_bucket = []
        spmn, buildingJson = buildSPMN(self.dataset, self.vers[0])

        for i in tqdm(range(1,self.number_of_sets)):
            spmn_bucket.append(deepcopy(spmn))

        #for i in tqdm(range(1,self.number_of_sets)):
            #spmn, buildingJson = buildSPMN(self.dataset,self.vers[i],buildingJson=buildingJson)
            #spmn_bucket.append(spmn)
        with open("output/cspmn_data.json","w+") as f:
            json.dump(buildingJson, f)

        return spmn_bucket


    def getCascasing(self):
        if(not os.path.exists("credal_spmns.pickle")): return False
        with open(f"credal_spmns.pickle", "rb") as file:
            self.sets = pickle.load(file)
        return True


    def cascading_best_next_decision(self,state):
        credal_values = []
        dominant_decisions = []
        full_decisions = []
        for cspmn_list in self.sets:
            decision, value, full_dec_list = credal_best_next_decision(cspmn_list,state)
            dominant_decisions.append(decision)
            credal_values.append(value)
            full_decisions.append(full_dec_list.copy())
        for x in range(len(credal_values)):
            if(credal_values[x] >= self.weight[x]): return dominant_decisions[x], dominant_decisions, credal_values, self.rangeW, full_decisions

        #TODO for reality this would be replaced with a default decision, or alternative decision system
        return dominant_decisions[0],dominant_decisions,credal_values, self.rangeW, full_decisions





























