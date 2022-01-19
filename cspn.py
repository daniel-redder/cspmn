import math
import multiprocessing

import numpy as np

import logging

from spn.algorithms.EM import EM_optimization
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.io.ProgressBar import printProgressBar
from spn.algorithms.Inference import log_likelihood
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.Base import Context
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import random

from spn.structure.StatisticalTypes import MetaType

from spn.structure.Base import Sum

from spn.algorithms.MEU import best_next_decision

import numpy
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans

import time


#TODO Dynamic recursion limit?  How is this handeled by Swaraj?
#inplace for bug, doesn't solve bug though......
#sys.setrecursionlimit(1000000000)

cols = "rdc"
rows = "kmeans"
min_instances_slice = 200
threshold = 0.3
ohe = False
leaves = create_histogram_leaf
rand_gen = None
cpus = -1



class contaminator():
    def __init__(self):
        self.rng = numpy.random.default_rng()

    #random "contamination" not really "e" contamination yet
    #currently performs normalized randoms through dirchlet distribution
    def e_contam(self, node):
        # contaminate them
        node.weights = self.rng.dirichlet(alpha=[random.randint(1,100) for x in node.weights])
        return node.weights

#Generates contaminator object on file load
cont = contaminator()


"""
Parser that visits every node in the SPMN checks if it is a sum node then contaminates the weights of sum nodes
input: SPMN
output: credalized spmn
"""
def learnCSPNs(curr_node_list=[]):
    curr_node_parser = curr_node_list[0]
    if(not hasattr(curr_node_parser,"children")): return True
    if( not curr_node_parser.children): return True

    if isinstance(curr_node_parser,Sum):

        for curr_node in curr_node_list:

            curr_node.weights=cont.e_contam(curr_node)

    for i in range(0,len(curr_node_parser.children)):
        #print(f"learning child {i}")
        learnCSPNs([node.children[i] for node in curr_node_list])

    return curr_node_list



#TODO maybe add independence testing change
"""
Generates the SPN given the independence testing method(?) and the dataset
"""
def buildSPN(dataset):
    df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
    train = df.values
    df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
    test = df2.values
    data = np.concatenate((train, test))
    var = data.shape[1]
    train, test = data, data

    df3 = pd.read_csv(f"spn/data/binary/{dataset}.valid.data", sep=',')
    valid = df3.values

    ds_context = Context(meta_types=[MetaType.DISCRETE] * train.shape[1])
    ds_context.add_domains(train)

    # Initialize splitting functions and next operation
    split_cols = get_split_cols_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
    split_rows = get_split_rows_KMeans()
    nextop = get_next_operation(min_instances_slice)

    # Learn the spn
    print('\nStart Learning...')
    start = time.time()
    spn = learn_structure(train, ds_context, split_rows, split_cols, leaves, nextop)
    end = time.time()
    print('SPN Learned!\n')

    print(f'Runtime: {end - start}')

    # inplace hardEM operation
    EM_optimization(spn, valid)

    #TODO cleanup the returns and printing script
    return spn, test, train, var, start, end






def testSPN(spn, test, train, dataset, path, var,start,end):
    nodes = get_structure_stats_dict(spn)["nodes"]

    def get_loglikelihood(instance):
        test_data = np.array(instance).reshape(-1, var)
        return log_likelihood(spn, test_data)[0][0]


    batches = 10
    pool = multiprocessing.Pool()
    batch_size = int(len(test) / batches)
    batch = list()
    total_ll = 0
    for j in range(batches):
        test_slice = test[j * batch_size:(j + 1) * batch_size]
        lls = pool.map(get_loglikelihood, test_slice)
        total_ll += sum(lls)
        printProgressBar(j + 1, batches, prefix=f'Evaluation Progress:', suffix='Complete', length=50)

    ll = total_ll / len(test)

    # Print and save stats
    print("\n\n\n\n\n")
    print("#Nodes: ", nodes)
    print("Log-likelihood: ", ll)

    print("\n\n\n\n\n")
    with open(f"{path}/{dataset}_stats.txt", "w") as f:
        f.write(f"\n\n\n{dataset}\n\n")
        f.write(f"\n#Nodes: {nodes}")
        f.write(f"\nLog-likelihood: {ll}")
        f.write(f"\nTime: {end - start}")
        f.write("\n\n\n\n\n")


