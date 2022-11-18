import pandas as pd
import pickle
import numpy as np

import torch
import json
from sklearn.preprocessing import LabelEncoder
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_BIC
from causallearn.utils.GraphUtils import GraphUtils
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pydot 

from torch.utils.data import DataLoader
import torch
#Import for modification of ges 
from typing import Optional, List, Dict, Any
from numpy import ndarray
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

#import causalGenTool
from fairness.Max.CGT.core.utils import show, handle_dcausation, graph2adj, check_cycles
from fairness.Max.CGT.patch.causalLearnPatch import ges_updated
from fairness.Max.CGT.core.generator import CausalGenerator, norm_col, KL
from fairness.Max.CGT.core.discovery import Discoverer
from fairness.Max.CGT.core.bias import Observer
from fairness.Max.CGT.core.assessment import Assessor
from fairness.Max.CGT.core.mwcf_train.trainer import Model
from fairness.Max.CGT.core.mwcf_train.utils import MWCF_loss, dataset
from fairness.Max.CGT.core.fitter import LinearSCM



for p in tqdm(range(100)):

    adj = np.array([[0,0,-4,0,0,-6,0,0,0,-5],\
                    [-2,0,-7,0,0,1,-6,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [-2,-4,-2,0,0,-5,6,0,-6,0],
                    [0,-5,5,0,0,0,0,0,0,0],
                    [0,0,5,0,0,0,0,0,0,0],
                    [2,0,0,0,0,0,0,0,0,-6],
                    [3,3,0,4,0,-1,0,0,0,0],
                    [6,0,0,0,0,-3,0,0,0,4],
                    [0,0,0,0,0,0,0,0,0,0]])

    test = CausalGenerator(adj, var_names=None, size=100000, normalize=False)
    test.graph_from_adjacency_matrix(directed=True)
    test.generate_data()
    test.compute_CPDs()
    test.allEdgesCausalStrength()
    test.generate_counterfactual_worlds('x0')


    probs = {
        'keys':['x0','x'+str(adj.shape[0]-1)],
        'values': (1,1),
        'prob': 0.7
    }
    alg ='pc'
    pc_param = {'alpha' : [0.05], 'indep_test':[gsq]}
    conds = [probs]
    results = {}
    scms = []

    while len(scms) ==0:
        observer = Observer(test.data, 'x0')
        sample = observer.observe(bias=conds, n_data=10000, debug=False)

            
        discoverer = Discoverer()
        discoverer.grid_search(sample.to_numpy(), alg, pc_param, test.graph)
        true_graph = discoverer.grid_search_results()
        adj_  = graph2adj(discoverer.discovered_graphs[-1].graph)
        #/!\ True adj 
        #adj_ = (test.adjency_matrix != 0).astype(float)
        adj_list = handle_dcausation(adj_)
        adj_list = [adj for adj in adj_list if not check_cycles(adj)[0]]
        scm_fitter = [LinearSCM(adj) for adj in adj_list]
        scms = [fitter.fit(sample) for fitter in scm_fitter]

    counterfactuals = []
    for idx, scm in enumerate(scms):
        #Empty
        empty = pd.DataFrame(0, index=range(len(sample)), columns=sample.columns)
        new_gen = CausalGenerator(adjency_matrix=scm, normalize=True)
        new_gen.graph_from_adjacency_matrix(directed=True)
        new_gen.generate_data(store=True)
        #ynew_gen.noise = scm_fitter[idx].noises[:,sample.index]
        sample = sample.reset_index(drop=True)
        new_gen.noise = scm_fitter[idx].noises[:,sample.index]
        new_gen.data = sample
        new_gen.generate_counterfactual_worlds('x0')
        #counterfactual = pd.concat([new_gen.counterfactual_world_1.iloc[sample[sample['x0'] == 0].index], new_gen.counterfactual_world_0.iloc[sample[sample['x0'] == 1].index]], axis=0)
        empty.update(new_gen.counterfactual_world_1.iloc[sample[sample['x0'] == 0].index])
        empty.update(new_gen.counterfactual_world_0.iloc[sample[sample['x0'] == 1].index])
        counterfactuals.append(empty)
    
    #Train test split
    X_train, X_test, y_train, y_test= train_test_split(sample.iloc[:,:-1], sample.iloc[:,-1], test_size=0.2)
    counterfactuals_train = [counterfactual.iloc[X_train.index] for counterfactual in counterfactuals]
    counterfactuals_test = [counterfactual.iloc[X_test.index] for counterfactual in counterfactuals]

    #Datasets
    trainset = dataset(X_train.values, y_train.values, counterfactuals_train)
    testset = dataset(X_test.values, y_test.values, counterfactuals_test)

    #DataLoader
    trainloader = DataLoader(trainset, batch_size=128, shuffle=False)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    cf_accus = []
    f1s = []
    precisions = []
    recalls = []
    lambdas = np.arange(start=0,stop=7, step=0.2)
    fouts_0 = []
    fouts_1 = []
    for lam in tqdm(lambdas):

        epochs = 200
        model = Model(adj.shape[0]-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=5e-4)
        loss = MWCF_loss(lam=lam, eps=0.1)
        model.train()
        for i in range(epochs):
            trainloss_total = []
            valloss_total = []
            model.train(True)
            for j, (x_train, y_train, cf_train) in enumerate(trainloader):
                output = model(x_train)
                cf_output = torch.hstack([model(counter_data) for counter_data in cf_train])
                error = loss(output, y_train, cf_output)
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
                trainloss_total.append(error)

            if (i % 100) == 0:
                model.train(False)
                for k, (x_test, y_test, cf_test) in enumerate(testloader):
                    out_test = model(x_test)
                    cf_out_test = torch.hstack([model(counter_data_test) for counter_data_test in cf_test])
                    error = loss(out_test, y_test, cf_out_test)
                    valloss_total.append(error)

        cf0 = torch.from_numpy(test.counterfactual_world_0.values[:,:-1]).float()
        cf1 = torch.from_numpy(test.counterfactual_world_1.values[:,:-1]).float()
        gp0 = test.data.values[:,:-1][test.data.values[:,0] == 0]
        gp1 = test.data.values[:,:-1][test.data.values[:,0] == 1]

        f1, precision, recall = model.evalmodified(testloader)
        cf_accu = model.eval_cf(cf0,cf1)
        fouts_0.append(model(torch.from_numpy(gp0).float()))
        fouts_1.append(model(torch.from_numpy(gp1).float()))
        cf_accus.append(cf_accu)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    KLs = []
    for o0, o1 in zip(fouts_0, fouts_1):
        v0 = plt.hist(o0.detach().numpy(), bins=30, alpha = 1, label='Group 0', weights = np.zeros_like(o0.detach().numpy()) + 1. / len(o0.detach().numpy()), density=True)
        v1 = plt.hist(o1.detach().numpy(), bins=v0[1], alpha = 0.7, label='Group1', weights = np.zeros_like(o1.detach().numpy()) + 1. / len(o1.detach().numpy()), density=True)
        v0[0][np.where(v0[0] == 0)] = 1
        v1[0][np.where(v1[0] == 0)] = 1
        KLs.append(KL(v0[0]/sum(v0[0]),v1[1]/sum(v1[0])))
    with open("./script_results/examp3/cf_accus"+str(p)+".txt", "w") as f1:
        for s in cf_accus:
            f1.write(str(s.numpy()[0]) +"\n")
        
    with open("./script_results/examp3/precs"+str(p)+".txt", "w") as f2:
        for s in precisions:
            f2.write(str(s.numpy()[0]) +"\n")
    
    with open("./script_results/examp3/KLs"+str(p)+".txt", "w") as f3:
        f3.write(str(KLs) +"\n")