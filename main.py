'''

This Code is used to learn and evaluate
the SPN models for the given datasets
using the LearnSPN algorithm


'''

import numpy as np

from cspn import testSPN
from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.algorithms.EM import EM_optimization

import logging
from spn.algorithms.Inference import log_likelihood

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from os import path as pth
import sys, os
import time

import pickle

from caspn import caSpn

# Initialize parameters

cols = "rdc"
rows = "kmeans"
min_instances_slice = 200
threshold = 0.3
ohe = False
leaves = create_histogram_leaf
rand_gen = None
cpus = -1

datasets = ["nltcs", "msnbc", "kdd", "baudio", "jester", "bnetflix"]
datasets = ["nltcs"]
# datasets = ['bnetflix']
path = "original_new_opt"

# Create output directory



# Get log-likelihood for the instance
def get_loglikelihood(instance):
    test_data = np.array(instance).reshape(-1, var)
    return log_likelihood(spn, test_data)[0][0]




def buildSPN(dataset):
    # Read training and test datasets

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

    print(f'Runtime: {end-start}')

    #inplace hardEM operation
    EM_optimization(spn, valid)

    with open(f"{path}/models/spn_{dataset}.pkle", 'wb') as file:
        pickle.dump(spn, file)

    return spn, train, test, var

# Learn SPNs form each dataset
for dataset in datasets:

    print(f"\n\n\n{dataset}\n\n\n")
    if not pth.exists(f'{path}/models'):
        try:
            os.makedirs(f'{path}/models')
        except OSError:
            print("Creation of the directory %s failed" % path)
            sys.exit()

    credal = caSpn(dataset=dataset)

    credal.learn(force_make_new=True)
    context = credal.context_bucket[0]
    otherContext = credal.context_bucket[1]
    #spn, test, train, dataset, path, var,start,end
    testSPN(credal.spns[0],context[0],context[1],context[2],context[3],context[4],context[5],credal.vers[0])
    testSPN(credal.spns[1],otherContext[0],otherContext[1],otherContext[2],otherContext[3],otherContext[4],otherContext[5],credal.vers[1])
    with open("original_new_opt/models/caspn_stats.txt","a+") as f:
        f.write(f"\n {credal.testCASPN()}")
