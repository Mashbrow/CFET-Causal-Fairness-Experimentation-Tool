import pydot
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import pickle
import json
from .utils import KL

def bern(p):
    """
        Bernoulli law with parameter p
    """
    x1 = np.random.random()
    if x1 >= p:
        return 0
    else:
        return 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_col(matrix):

    new_matrix = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[1]):
        vec = matrix[:,i]
        if not np.all(vec==0):
            vec = vec/np.linalg.norm(vec)
        new_matrix[:,i] = vec

    return new_matrix

class CausalGenerator:
    """
        Given a adjency matrix, compute bayesian network and generate data
    
    """

    __slot__ = ["adjency_matrix"]

    def __init__(self, adjency_matrix, var_names=None, size=10000, normalize=False):

        self.adjency_matrix = adjency_matrix if not normalize else norm_col(adjency_matrix)

        if var_names == None:
            self.var_names = ["x"+str(i) for i in range(adjency_matrix.shape[0])]
        else: 
            self.var_names = var_names

        self.sigmas = np.random.random_sample(len(self.var_names))
        self.size = size

        self.noise = np.array([np.random.normal(loc=0, scale=self.sigmas[j], size=size)\
                                 for j in range(len(self.sigmas))])

        self.CPDs = {}

    def graph_from_adjacency_matrix(self, node_prefix="", directed=False):
        """
            Modified version from pydot taking into account variable
            names for display purposes.
            
            Creates a basic graph out of an adjacency matrix.
            The matrix has to be a list of rows of values
            representing an adjacency matrix.
            The values can be anything: bool, int, float, as long
            as they can evaluate to True or False.
        """

        node_orig = 1

        if directed:
            graph = pydot.Dot(graph_type="digraph")
        else:
            graph = pydot.Dot(graph_type="graph")

        for row in self.adjency_matrix:
            if not directed:
                skip = self.adjency_matrixmatrix.index(row)
                r = row[skip:]
            else:
                skip = 0
                r = row
            node_dest = skip + 1

            for idx, e in enumerate(r):
                if e:
                    graph.add_edge(
                        pydot.Edge(
                            "%s%s" % (node_prefix, self.var_names[node_orig-1]),
                            "%s%s" % (node_prefix, self.var_names[node_dest-1]),
                        )
                    )
                node_dest += 1
            node_orig += 1

        self.graph = graph

    def get_children(self, node):
        """
            Get children of node
            Argument:
                - node : int or str
            Returns:
                - list of names of children
                - list of index of children in self.var_names
        """

        if isinstance(node, int):
            names = [self.var_names[value] for value in np.nonzero(self.adjency_matrix[node-1,:])[0]]
            index = [value for value in np.nonzero(self.adjency_matrix[node-1,:])[0]]
            return names, index
        elif isinstance(node, str):
            idx = self.var_names.index(node)
            names = [self.var_names[value] for value in np.nonzero(self.adjency_matrix[idx,:])[0]]
            index = [value for value in np.nonzero(self.adjency_matrix[idx,:])[0]]
            return names, index
        else:
            raise TypeError 
        
    def get_parents(self, node):
        """
            Get parents of node
            Argument:
                - node : int or str
            Returns:
                - list of names of parents
                - list of index of parents in self.var_names
        """
        
        if isinstance(node, int):
            names = [self.var_names[value] for value in np.nonzero(self.adjency_matrix[:,node-1])[0]]
            index = [value for value in np.nonzero(self.adjency_matrix[:,node-1])[0]]
            return names, index
        elif isinstance(node, str):
            idx = self.var_names.index(node)
            names = [self.var_names[value] for value in np.nonzero(self.adjency_matrix[:,idx])[0]]
            index = [value for value in np.nonzero(self.adjency_matrix[:,idx])[0]]
            return names, index
        else:
            raise TypeError 


    def generate_data(self, store=True, data=None):
        """
            To re-verify (changes)
            Generate data according to the Graph
            Argument:
                - store : Bool, to store in instance class or not
                - data : pd.DataFrame, dataframe on which generate missing values

        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(columns=self.var_names, index=np.arange(start=0, stop=self.size, step=1))

        def __values(node):
            """
                Reccursive function to generate data of node given values of its parents
                Argument:
                    - node : node name in str
            """
            
            assert isinstance(node, str), 'Expected str'

            #Initial w0 
            value_ = np.array([0.0 for i in range(len(data))])

            current_node_id = self.var_names.index(node)
            parents , idx = self.get_parents(node)
            if data[node].isnull().values.any():
                if len(parents) == 0:
                    #Changer param bern
                    data[node] = [bern(0.5) for i in range(self.size)]
                elif len(parents) > 0:
                    for parent, parent_id in zip(parents, idx):
                        if data[parent].isnull().values.any():
                            __values(parent)
                        else:
                            value_ += np.array([self.noise[current_node_id][i] +\
                                     (self.adjency_matrix[parent_id][current_node_id]*data[parent][i]) for i in data.index])
                    data[node] = [bern(sigmoid(value)) for value in value_]
            
        for var in data.columns:
            __values(var)
        
        if store:
            self.data = data.copy()

        return data
    
    def generate_data_v2(self, store=True, data=None):
        """
            Try broadcasting
            Generate data according to the Graph
            Argument:
                - store : Bool, to store in instance class or not
                - data : pd.DataFrame, dataframe on which generate missing values

        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(columns=self.var_names, index=np.arange(start=0, stop=self.size, step=1))

        def __values(node):
            """
                Reccursive function to generate data of node given values of its parents
                Argument:
                    - node : node name in str
            """
            
            assert isinstance(node, str), 'Expected str'

            #Initial w0 
            value_ = np.zeros(len(data))

            current_node_id = self.var_names.index(node)
            parents , idx = self.get_parents(node)
            if data[node].isnull().values.any():
                if len(parents) == 0:
                    #Changer param bern
                    data[node] = [bern(0.5) for i in range(self.size)]
                elif len(parents) > 0:
                    for parent, parent_id in zip(parents, idx):
                        if data[parent].isnull().values.any():
                            __values(parent)
                        else:
                            value_ += self.noise[current_node_id] +\
                                     (self.adjency_matrix[parent_id][current_node_id]*np.array(data[parent]))
                    data[node] = [bern(sigmoid(value)) for value in value_]
            
        for var in data.columns:
            __values(var)
        
        if store:
            self.data = data.copy()

        return data

    def show_graph(self):
        """
            Display graph according to adjency matrix
        """
        tmp_png = self.graph.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.figure(figsize=(20,120))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    
    def CPD(self, node, node_values):
        """
            Compute CPD of given node according to a value of state (A0,...,An)
            Argument:
                - node: node name in str
                - node_values: the value of the tuple (A0,...,An) for example (1,0,1,0,1)
            Returns:
                - The computed CPD.
        """

        assert isinstance(node_values, tuple), 'Tuple expected'
        num = []
        quo = []

        parents, idx = self.get_parents(node)
        if len(parents)>0:
            num.append(node_values[self.var_names.index(node)])
            for id in idx:
                quo.append(node_values[id])
                num.append(node_values[id])

            tuple_num = tuple(num)
            tuple_quo = tuple(quo)
            key = 'P(' + node + '=' + str(num[0]) + '|'
            for idx_, value in enumerate(quo):
                key += parents[idx_] + '=' + str(value) + ', '
            key = key[:-2] + ')'
            
            quotient = self.data.value_counts(subset=parents, normalize = True)[tuple_quo]
            parents.insert(0, node)
            numerator = self.data.value_counts(subset=parents, normalize = True)[tuple_num]
            self.CPDs[key] = (numerator/quotient)
            return (numerator/quotient)
        else:
            id = node_values[self.var_names.index(node)]
            return self.data.value_counts(subset=node, normalize=True)[id]

    def compute_CPDs(self):

        joined_prob = self.data.value_counts(normalize=True)
        for node_values in joined_prob.index:
            for var in self.var_names:
                _ = self.CPD(var, node_values)

    def counterfactuals(self, sensible_node, value, data):
        """
            Generate Counterfactuals of data given the new value of a sensible node
            Arguments:
                - sensible_node: str, the sensible node name
                - value: int, the value to be given to the sensible parameter
                - data: pd.Dataframe, the data on which compute the counterfactuals
            Returns:
                - pd.DataFrame, counterfactual data
        """

        to_counterfactual = data[data[sensible_node] != value].copy()
        to_keep = data[data[sensible_node] == value].copy()
        to_counterfactual[sensible_node] = np.array([value for i in range(len(to_counterfactual.index))])

        children, idx = self.get_children(sensible_node)
        none_values = np.array([None for i in range(len(to_counterfactual.index))])

        for child in children:
            to_counterfactual[child] = none_values

        counterfactuals_ = self.generate_data(store=False, data=to_counterfactual)

        counterfactuals_data = pd.concat([to_keep, counterfactuals_], axis=0)
        counterfactuals_data.sort_index(axis=0, ascending=True, inplace=True)
        
        return counterfactuals_data
    
    def generate_counterfactual_worlds(self,sensible_node):
        """
            Generate two counterfactual worlds corresponding each to a different value of the sensible parameter
            Argument:
                - sensible_node: str, the sensible node name
        """

        self.counterfactual_world_0 = self.counterfactuals(sensible_node, value=0, data=self.data)
        self.counterfactual_world_1 = self.counterfactuals(sensible_node=sensible_node, value=1, data=self.data)

    def _P_S_cond(self, edge, node_values):

        source = edge.get_source()
        target = edge.get_destination()
        value_t = node_values[self.var_names.index(target)]
        parents, index_p = self.get_parents(target)
        P_s_cond = 0
        for i in range(2):
            key = 'P('+target+'='+str(value_t)+'|'
            for parent, idx in zip(parents, index_p):
                if parent == source:
                    key+=parent+'='+ str(i) +', '
                else:
                    key+=parent+'='+ str(node_values[idx])+', '
            key = key[:-2] + ')'
        
            P_s_cond += self.CPDs[key]*self.data.value_counts(subset=[source],normalize=True)[i] if key in list(self.CPDs.keys()) else 0
        
        return P_s_cond

    def _P_S(self, edge, node_values):

        target = edge.get_destination()
        value_t = node_values[self.var_names.index(target)]
        parents, index_p = self.get_parents(target)

        key = 'P('+target+'='+str(value_t)+'|'
        for parent, idx in zip(parents, index_p):
                key+=parent+'='+ str(node_values[idx])+', '
        key = key[:-2] + ')'

        P_s_cond = self._P_S_cond(edge, node_values)
        num = self.data.value_counts(normalize=True)[node_values]
        P_S = (num/self.CPDs[key])*P_s_cond

        return P_S
    
    def causalStrength(self, edge, debug=False):
        
        P_As = []
        P_Ss = []

        for idx, value in self.data.value_counts(normalize=True).items():
            P_As.append(value)
            P_S = self._P_S(edge, idx)
            P_Ss.append(P_S)
            if debug:
                print(idx, P_S)
        
        causalStrength = KL(p=P_As,q=P_Ss)
        return causalStrength
    
    def allEdgesCausalStrength(self):
        
        self.causal_strengths = []
        for edge in self.graph.get_edges():
            causal_strength = self.causalStrength(edge)
            edge.set_label('CS: ' + str(np.round(causal_strength, decimals=5)))
            self.causal_strengths.append(causal_strength)
    
    def save(self):

        with open('cond_probs.pkl', 'wb') as f:
            pickle.dump(self.CPDs, f)

        with open('cond_probs.json', 'w') as fp:
            json.dump(self.CPDs, fp)

        self.data.to_csv('synthetic_data.csv')