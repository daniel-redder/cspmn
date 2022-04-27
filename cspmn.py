import math
from itertools import repeat
from sklearn import preprocessing as prepro
import numpy as np

import logging
from spn.algorithms.EM import EM_optimization
from spn.algorithms.MEU import meu
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
from spn.structure.Base import Sum, get_number_of_nodes
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
        print(random.uniform(rangeW[0],rangeW[1]))
        node.weights = self.rng.dirichlet(alpha=[random.uniform(rangeW[0],rangeW[1]) for x in node.weights])
        return node.weights
cont = contaminator()

#Deprecated
#parses tree finding minimum and maximum weights
def fixWeightRange(curr_node_list=[],bias=0,minW=50,maxW=50):
    curr_node_parser = curr_node_list[0]
    if (not hasattr(curr_node_parser, "children")): return True
    if (not curr_node_parser.children): return True

    if isinstance(curr_node_parser, Sum):

        for curr_node in curr_node_list:

            if min(curr_node.weights < minW):
                minW = min(curr_node.weights)
                print(minW," minW")
            if max(curr_node.weights) > maxW:
                maxW = max(curr_node.weights)
                print(maxW," maxW")
    for i in range(0, len(curr_node_parser.children)):
        # print(f"learning child {i}")
        fixWeightRange([node.children[i] for node in curr_node_list],bias,max(0,minW),min(100,maxW+bias))

    #quick test
    #this can go negative because the dirchlet distribution will normalize it.
    #print(minW,bias)
    return max(0,minW-bias),min(maxW+bias,100)



def normalize_weights(weights, bias):
    new_weights = []
    #handles potential 0s from normal sampling in normalization
    if len(weights) == 1: return [1.0]
    for x in weights:
        new_weights.append(np.random.normal(x,bias,1)[0])
    new_weights = np.array([x+abs(min(new_weights)) for x in new_weights])
    #assert sum(prepro.normalize(new_weights])) == 1.0, f"Error {((new_weights / np.linalg.norm(new_weights)).tolist())}"    
    print(new_weights)
    #Doesn't work
    #new_weights = prepro.normalize([new_weights]).tolist()[0]
    new_weights = new_weights.tolist()
    new_weights = [x / sum(new_weights) for x in new_weights]
    print(new_weights)
    #assert sum(new_weights) == 1.0, f'{new_weights}'
    return (new_weights)



def learnCSPMNs(curr_node_list=[],bias=0,count=0,origin=[]):
    curr_node_parser = curr_node_list[0]
    count+=1
    if(not hasattr(curr_node_parser,"children")): return None, count+1, origin
    if( not curr_node_parser.children): return None, count+1, origin

    if isinstance(curr_node_parser,Sum):
        for curr_node in curr_node_list:

            temp_test = curr_node.weights.copy()

            curr_node.weights=normalize_weights(curr_node.weights,bias)
            print(curr_node.weights,"vs old: ",temp_test,"  bias is: ",bias)
            origin.append(temp_test.copy().tolist())

    for i in range(0,len(curr_node_parser.children)):
        curr_node_hold, count, origin  = learnCSPMNs([node.children[i] for node in curr_node_list],bias,count+1,origin)

    return curr_node_list, count+1,origin




def learner(spmn, n=10,bias=0,origin=[]):
    curr_node_list = [copy.deepcopy(spmn) for x in range(n)]

    spmn, count, origin = learnCSPMNs(curr_node_list,bias)

    return spmn, count, origin




def buildSPMN(dataset,ver,buildingJson={"before":{"ll":[],"meu":[],"data":[],"rewards":[],"reward_dev":[]},"after":{"ll":[],"meu":[],"data":[],"rewards":[],"reward_dev":[]}}):
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
    meu_test = [[np.nan]*len(feature_names)]
    spmn = spmn.learn_spmn(train)
    print(get_number_of_nodes(spmn))
    print(get_structure_stats_dict(spmn)["nodes"],"  pizza")
    buildingJson["before"]["meu"].append(meu(spmn,meu_test)[0])
    buildingJson["before"]["ll"].append(log_likelihood(spmn,test)[0][0])
    #rewards, reward_dev = test_rewards(spmn,dataset, test)
    buildingJson["before"]["data"].append(dataset)
    #buildingJson["before"]["rewards"].append(rewards)
    #buildingJson["before"]["reward_dev"].append(reward_dev)
    EM_optimization(spmn,train)

    buildingJson["after"]["meu"].append(meu(spmn,meu_test)[0])
    #rewards, reward_dev = test_rewards(spmn,dataset, test)
    buildingJson["after"]["ll"].append(log_likelihood(spmn,test)[0][0])
    #buildingJson["after"]["rewards"].append(rewards)
    #buildingJson["after"]["reward_dev"].append(reward_dev)

    return spmn, buildingJson


def get_reward(ids,spmn,env):

	#policy = ""
	state = env.reset()
	while(True):
		output = best_next_decision(spmn, state)
		action = output[0][0]
		#policy += f"{action}  "
		state, reward, done = env.step(action)
		if done:
			return reward
			#return policy

def test_rewards(spmn, dataset, test):
    env = get_env(dataset)
    total_reward = 0
    batch_count = 25
    batch_size = 20000
    batch = list()

    pool = multiprocessing.Pool()
    policy_set = list()

    for z in range(batch_count):
        ids = [None for x in range(batch_size)]
        rewards = pool.starmap(get_reward,zip(ids,repeat(spmn),repeat(env)))

        # policies = pool.map(get_reward, ids)
        # policy_set += policies
        # print(Counter(policy_set))

        batch.append(sum(rewards) / batch_size)
        # print(batch[-1])
        printProgressBar(z + 1, batch_count, prefix=f'Average Reward Evaluation :', suffix='Complete', length=50)

    avg_rewards = np.mean(batch)
    reward_dev = np.std(batch)
    return avg_rewards, reward_dev



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


    return dominant_action,credal_value, curr_state_decisions





