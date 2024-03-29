import json

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
#from spn.algorithms.MEU import meu
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
import sqlite3 as sql




#class contaminator():
 #   def __init__(self):
  #      self.rng = numpy.random.default_rng()

    #random "contamination" not really "e" contamination yet

   # def e_contam(self, node):
    #    # contaminate them
     #   node.weights = self.rng.dirichlet(alpha=[random.randint(1,100) for x in node.weights])
      #  return node.weights
#cont = contaminator()

#input a node





# def learnCSPMNs(curr_node_list=[]):
#     curr_node_parser = curr_node_list[0]
#     if(not hasattr(curr_node_parser,"children")): return True
#     if( not curr_node_parser.children): return True
#     #print(curr_node_parser.children)
#     #if(hasattr(curr_node_parser,"weights")): print(curr_node_parser.weights)
#     if isinstance(curr_node_parser,Sum):
#         #print("prior",curr_node_parser.weights)
#         for curr_node in curr_node_list:
#             #print("prior",curr_node.weights)
#
#             #TODO FIX THIS Duplication bug* ?
#
#             #equal normalized weights
#             #curr_node.weights = [1/len(curr_node.weights) for x in curr_node.weights]
#
#
#             #Random normalized weights
#            # curr_node.weights=cont.e_contam(curr_node)
#
#
#             #Make one random node be 1.0 weighted and the rest 0
#             #============================================================
#             #curr_node.weights=[0 for x in curr_node.weights]
#             #curr_node.weights[random.randint(0,len(curr_node.weights)-1)]=1.0
#             #==============================================================
#             #print("post",curr_node.weights)
#
#         #print("post",curr_node_parser.weights)
#     #Old method for single CSPMN generation
#     #for child_node in curr_node.children: learnCSPMNs(child_node,n)
#
#     for i in range(0,len(curr_node_parser.children)):
#         #print(f"learning child {i}")
#         learnCSPMNs([node.children[i] for node in curr_node_list])
#
#     return curr_node_list
#
# def learner(spmn, n=10):
#     curr_node_list = [copy.deepcopy(spmn) for x in range(n)]
#     return learnCSPMNs(curr_node_list)


#
# #plot_spn(valus[0],'spmn.pdf', feature_labels=feature_labels)
#
# from collections import defaultdict
# def credal_best_next_decisions(cspmn_list, state,env):
#     decisions = {}
#     state_iterator = 0
#     env.reset()
#     while(True):
#         decisions[state_iterator] = {}
#         for cspmn in cspmn_list:
#             spmn_output = best_next_decision(cspmn,state)
#             action = spmn_output[0][0]
#             #print(action)
#
#             if action in decisions[state_iterator]: decisions[state_iterator][action] = decisions[state_iterator][action]+1
#             else: decisions[state_iterator][action]=1
#
#         dominant_action = None
#         credal_value = 0
#         curr_state_decisions=decisions[state_iterator]
#         for x in curr_state_decisions:
#             if curr_state_decisions[x]>credal_value:
#                 dominant_action = x
#                 credal_value = curr_state_decisions[x]
#
#         curr_state, reward, done=env.step(dominant_action)
#         state = curr_state
#         state_iterator+=1
#         if done:
#             break
#
#     return decisions
#
#
#
# def get_reward(dataset,spmn):
#     # policy = ""
#     env = get_env(dataset)
#     state = env.reset()
#     while (True):
#         output = best_next_decision(spmn, state)
#         print(output)
#         action = output[0][0]
#         # policy += f"{action}  "
#         state, reward, done = env.step(action)
#         if done:
#             return reward
#         # return policy
#
#
#
#
# #dataset = "Elevators"
#
#
#
#
#
#
#
#
# #Postprocessing Step to turn a spmn into a credal spmn
# #TODO Thread it
#
# #TODO turn it into real e_contamination
# #TODO fix RDC infinite Loop error, and g-test weirdness
#
#
#
# def buildSPMN(dataset):
#     partial_order = get_partial_order(dataset)
#     utility_node = get_utilityNode(dataset)
#     decision_nodes = get_decNode(dataset)
#     feature_names = get_feature_names(dataset)
#     feature_labels = get_feature_labels(dataset)
#     meta_types = [MetaType.DISCRETE] * (len(feature_names) - 1) + [MetaType.UTILITY]
#
#     df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')
#
#     df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
#
#     data = df.values
#     train, test = data, np.array(random.sample(list(data), int(data.shape[0] * 0.02)))
#
#     #print("Start Learning...")
#     spmn = SPMN(partial_order, decision_nodes, utility_node, feature_names, meta_types,
#                 cluster_by_curr_information_set=True, util_to_bin=False)
#
#     spmn = spmn.learn_spmn(train)
#
#     return spmn
#
#
# def getSPMN(dataset):
#     with open(f"spn/data/original_new/{dataset}/spmn_original.pkle","rb") as file:
#         spmn = pickle.load(file)
#     return spmn
#
#
# def credal_tester(datas):
#     for dataset in datas:
#         # buildSPMN(dataset)
#         print(f"Dataset: {dataset}")
#         spmn = getSPMN(dataset)
#         print("node count: ", get_structure_stats_dict(spmn)["nodes"])
#         valus = learner(spmn, 100)
#
#         env = get_env(dataset)
#
#         state = env.reset()
#
#         spmn_dec_list = []
#         while (True):
#             action = best_next_decision(spmn, state)[0][0]
#             spmn_dec_list.append(action)
#             state, reward, done = env.step(action)
#             if done:
#                 break
#
#         print("spmn Decision List: ", spmn_dec_list)
#         state = env.reset()
#
#         print("cspmn Decision Dictionary: ", credal_best_next_decisions(valus, state,env))
#         print("Number of Same nodes across CSPMN: ", child_parser(valus) / len(valus))
#
#         feature_names = get_feature_names(dataset)
#         test_data = [[np.nan] * len(feature_names)]
#
#         m = meu(spmn, test_data)
#         q = sum([meu(p, test_data)[0] for p in valus]) / len(valus)
#
#         meus = (m[0])
#
#         print("meus: ", meus)
#         print("Cmeus: ", q, "\n\n====================\n")
#
#         feature_labels = get_feature_labels(dataset)
#         if dataset == "Export_Textiles":
#             plot_spn(valus[4], 'cspmn.pdf', feature_labels=feature_labels)
#             plot_spn(valus[3], 'spmn.pdf', feature_labels=feature_labels)
#
#



datas=["Export_Textiles",'Powerplant_Airpollution', 'HIV_Screening', 'Computer_Diagnostician', 'Test_Strep']
print(datas)

#credal_tester(datas)
from cascading_spmn import caSpmn


def createCredalSPMNSets():
    for dataset in datas:
        cascading = caSpmn(dataset)
        print(dataset)
        cascading.learn()

        env = get_env(dataset)
        state = env.reset()

        print(cascading.cascading_best_next_decision(state))

        with open(f"models_credal_{dataset}.pickle","wb+") as fp:
            pickle.dump(cascading.sets[0],fp)

def caspmn_new_full_test(datas):
    print(datas)

    json_out = {}
    #IT was a good thought :)
    #conn=sql.connect("caspmn_test")
    #curse = conn.cursor()
    # curse.execute("CREATE TABLE IF NOT EXISTS mainOut(referenceID int , dataset string, decisionNumber int, "
    #               +"trainedDec int, cascadingDec int, PRIMARY KEY(referenceID, dataset, decisionNumber));")
    # curse.execute("CREATE TABLE IF NOT EXISTS credalWeight(referenceID int, dataset string, decisionNumber int, indexVal int, credalList int,"
    #               +" weightList int, decisionList, PRIMARY KEY(indexVal, referenceID, dataset,decisionNumber),"
    #               +"FOREIGN KEY(referenceID) REFERENCES mainOut(referenceID),"
    #               +"FOREIGN KEY(dataset) REFERENCES mainOut(dataset),"
    #               +"FOREIGN KEY(decisionNumber) REFERENCES mainOut(decisionNumber));")
    # Insert main table datapoints
    # curse.execute(f"INSERT INTO mainOut(referenceID, dataset string, decisionNumber, "
    # +"trainedDec, cascadingDec) VALUES({id},{dataset},{x},{best_next_decision(cascading.spmns[0],state)[0][0]},{decision})")

    # commit
    # conn.commit()

    # Insert child table datapoints
    # for ind in range(len(credalList)):
    #   curse.execute("INSERT INTO credalWeight(referenceID, dataset, decisionNumber, indexVal, credalList,"
    ###conn.commit()


    for dataset in datas:
        print(get_partial_order(dataset))
        cascading = caSpmn(dataset,number_of_credals=1000,number_of_sets=100,vers=["RDC" for x in range(100)],bias=.01)
        print(dataset)
        cascading.learn(force_make_new=True)

        #Was I tempted to just making a sqlite database to commit this to, perhaps.
        #Composite Attribute problems sigh you know what thats what I'm doing
        # -- hey its me but later remember json is easy json is nice


        json_out[dataset] = {"std":[],"data":[]}
        json_nest = []
        for id in range(0,3):
            env = get_env(dataset)
            state = env.reset()
            isDone = False
            x=1



            while(not isDone):
                decision, decisionList, credalList, weightList, full_dec_list =  cascading.cascading_best_next_decision(state)

                print(decisionList, credalList," state: ",state)


                feature_labels = get_feature_labels(dataset)

                json_feather = {"decisionNumber":x,"trainedDec":best_next_decision(cascading.spmns[0],state)[0][0],
                             "cascadingDec":decision, "decList":decisionList,"credalList":credalList,"weightList":weightList,"full_dec_list":full_dec_list}



                x=x+1
                state, reward, isDone = env.step(decision)

            json_nest.append(json_feather.copy())

        json_out[dataset]["data"].append(json_nest.copy())
        json_out[dataset]["std"]=cascading.origin[0]
        
        #print(json_out,type(json_out))
        with open("output/credaling_across_decisions.json","w+") as fp:
            json.dump(json_out,fp)


        #plot_spn(cascading.sets[0][0],f"graphs/naive_cspmn{dataset}.png",feature_labels=feature_labels)
        #plot_spn(cascading.spmns[0],f"graphs/naive_spmn{dataset}.png",feature_labels=feature_labels)


#print(get_reward(dataset,spmn))



#datas=['Export_Textiles']
caspmn_new_full_test(datas)




