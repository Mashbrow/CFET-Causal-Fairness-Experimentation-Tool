import itertools
import numpy as np
import pydot

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

class Discoverer:
    """
        Discoverer tool to recreate graph from data.
        Provides stability score after performing a grid search over a set of parameters

        self.penalties: the score to be given to each edge depending of its correctness
        self.labels: the colors of the edges depending of its correcteness 
    
    """

    def __init__(self):

        self.discovered_graphs = []
        self.discovered_graphs_pyd = []
        self.translate_dict = {}
        self.edge_scores = {}
        self.penalties = {
            'correct': 1,
            'misdirected': -1,
            'undiscovered' : 0
        }

        self.labels = {'correct': 'green', 'misdirected' : 'yellow', 'unexistant' : 'red'}
    
    def areEqual(self, edge1, edge2):
        """
            Unfortunately, pydot does not provide any __eq__ method for comparing to edges
            that is relevant in our case.
            The aim of this function is to assess the equalty between two edges in terms of
            skeleton (are two edges linking the same nodes?) and in terms of orientation.
            (-->, <--, --, <->)

            Arguments:
                - edge1, edge2 : pydot.Edge, the two edges to be compared
            Returns:
                skeleton: Bool, True if edges are linking the same nodes, else False
                orientation: Bool, True if edges have the same orientation, else False
        
        """

        dir1 = edge1.obj_dict['attributes']['dir']
        dir2 = edge2.obj_dict['attributes']['dir']
        if ((edge1.get_source() == edge2.get_source()) &
            (edge1.get_destination() == edge2.get_destination())) | \
            ((edge1.get_source() == edge2.get_destination()) & 
            (edge1.get_destination() == edge2.get_source())):

            skeleton = True
            if ((dir1 == 'both') | (dir1 == 'none')) & (dir1 == dir2):
                orientation = True
            elif (dir1 == dir2) & (dir1 == 'forward'):
                orientation = True if edge1.get_source() == edge2.get_source() else False
            else:
                orientation = False
        else:
            skeleton, orientation = False, False
            
        return skeleton, orientation


    def from_discovered_get_edges(self, discovered):
        """
            Graphs made from adjency as in the Generator class do not have the same syntax
            as the ones that are discovered. Thus this function aims at extracting the edges
            from a discovered graph and re-write them so that they can then be compared.

            Argument:
                - discovered: causallearn.CausalGraph, a discovered graph.
            Returns:
                - edges: list[pydot.Edge], a list of edges re-written from the discovered graph
        """

        adjency = discovered.graph
        edges = []
        indices = np.argwhere(np.triu(adjency) !=0)
        for coordinate in indices:
            if adjency[coordinate[0], coordinate[1]] == 1:
                if adjency[coordinate[1], coordinate[0]] == 1:
                    edges.append(pydot.Edge('x'+str(coordinate[0]),'x' + str(coordinate[1]), dir='both'))
                if adjency[coordinate[1], coordinate[0]] == -1:
                    edges.append(pydot.Edge('x'+str(coordinate[1]), 'x' + str(coordinate[0]), dir='forward'))
            elif adjency[coordinate[0], coordinate[1]] == -1:
                if adjency[coordinate[1],coordinate[0]] == 1:
                    edges.append(pydot.Edge('x'+str(coordinate[0]),'x' + str(coordinate[1]), dir='forward'))
                else:
                    edges.append(pydot.Edge('x'+str(coordinate[0]),'x' + str(coordinate[1]), dir='none'))
        
        return edges
    
    def from_true_get_edges(self, true_graph):
        """
            Graphs made from adjency as in the Generator class do not have the same syntax
            as the ones that are discovered. Thus this function aims at extracting the edges
            from the true graph and re-write them so that they can then be compared.

            Argument:
                - true_graph: pydot.Graph, the true graph given by the generation process
            Returns:
                - transformed_edges: list[pydot.Edge], a list of edges re-written from the
                true graph
        """

        transformed_edges = []
        for edge in true_graph.get_edges():
            if '->' in edge.to_string():
                transformed_edges.append(pydot.Edge(edge.get_source(), edge.get_destination(), dir='forward'))
            elif '<-' in edge.to_string():
                transformed_edges.append(pydot.Edge(edge.get_destination(), edge.get_source(), dir='forward'))
            elif '--' in edge.to_string():
                transformed_edges.append(pydot.Edge(edge.get_source(), edge.get_destination(), dir='none'))
            elif '<->' in edge.to_string():
                transformed_edges.append(pydot.Edge(edge.get_source(), edge.get_destination(), dir='both'), graph_type='digraph')
            
        return transformed_edges
    
    def true_edge_in_discovered(self, edge, discovered_edges):
        """
            Check if an edge from the true graph is in the edges of a discovered one
            and returns its score based on its correctness

            Arguments:
                - edge: pydot.Edge, an edge from the true graph
                - discovered_edges: list[pydot.Edge], list of edges from the discovered graph
                note that the edges must be in the same format as the output of from_true_get_edges
                and from_discovered_get_edges.
            Returns:
                - the score based on penalties given in self.penalties
        """

        results = []
        for discovered in discovered_edges:
            skeleton, orientation = self.areEqual(edge, discovered)
            results.append((skeleton, orientation))
            if results[-1][0] | results[-1][1]:
                break
        
        skeleton, orientation = results[-1]

        if skeleton:
            if orientation:
                return self.penalties['correct']
            else:
                return self.penalties['misdirected']
        else:
            return self.penalties['undiscovered']
        
    def discovered_edge_in_true(self, edge, true_edges):
        """
            Check if an edge from the discovered graph is in the edges of a true one
            and returns its score based on its correctness

            Arguments:
                - edge: pydot.Edge, an edge from the discovered graph
                - discovered_edges: list[pydot.Edge], list of edges from the true graph
                note that the edges must be in the same format as the output of from_true_get_edges
                and from_discovered_get_edges.
            Returns:
                - the color corresponding based on labels given in self.labels
        """

        results = []
        for true in true_edges:
            skeleton, orientation = self.areEqual(true, edge)
            results.append((skeleton, orientation))
            if results[-1][0] | results[-1][1]:
                break
        skeleton, orientation = results[-1]
        if skeleton:
            if orientation:
                return self.labels['correct']
            else:
                return self.labels['misdirected']
        else:
            return self.labels['unexistant']
            
    def discover(self, data, alg, param_dict):
        """
            Process discovery step

            Arguments:
                - data: np.array, the data on which discover the graph
                - alg: str, currently 'pc' or 'ges', the algorithm with which perform
                the discovery
                - param_dict: dict, in format {'param_name1' : value1, 'param_name2': ...}
                a dictionnary of parameters to be used when performing discovery
            
            Returns:
                graph: causallearn.CausalGraph, the discovered graph
        """

        cg = pc(data, **param_dict, uc_rule=0, uc_priority=1) if alg =='pc' else ges(data, **param_dict)
        graph = cg.G if alg =='pc' else cg['G']
    
        return graph
    
    def compare(self, true_graph, discovered_graph):
        """
            Compare the discovered graph with the true one.
            Compute the score of each edges.

            Arguments:
                - true_graph: pydot.Graph, the true graph given by the generation process
                - discovered_graph: causallearn.CausalGraph, the discovered graph
            Returns:
                - pyd: pydot.Graph, the modified discovered graph with labels added
        """

        discovered_edges = self.from_discovered_get_edges(discovered_graph)
        true_edges = self.from_true_get_edges(true_graph)
        pyd = GraphUtils.to_pydot(discovered_graph)

        for index, edge in enumerate(self.from_true_get_edges(true_graph)):
            score = self.true_edge_in_discovered(edge, discovered_edges)
            if index not in list(self.edge_scores.keys()):
                self.edge_scores[index] = [score]
            else:
                self.edge_scores[index].append(score)
        
        for index, edge in enumerate(self.from_discovered_get_edges(discovered_graph)):
            color = self.discovered_edge_in_true(edge, true_edges=true_edges)
            pyd.get_edges()[index].set_color(color)
        
        return pyd
    
    def discovered_ratio(self, discovered_pyd_graph):
        """
            Compute Recall and Precision of the graph reconstruction
            Argument:
                - discovered_pyd_graph: pyd.Graph, the discovered graph
            Returns:
                - record: dict, the results in dict where the difference
                between global and dir is that global only consider if an edge
                has been discovered or not while dir also check if it was well
                directed.
        """

        global_rec = {'tp' : 0, 'fp': 0, 'fn':0, 'tn': 0}
        dir_rec = {'tp' : 0, 'fp': 0, 'fn':0, 'tn': 0}
        total = len(self.gs_true_graph.get_edges())

        for edge in discovered_pyd_graph.get_edges():
            color = edge.obj_dict['attributes']['color']

            if color == self.labels['correct']:
                global_rec['tp']+=1
                dir_rec['tp']+=1
            elif color == self.labels['misdirected']:
                global_rec['tp']+=1
                dir_rec['fp']+=1
            elif color == self.labels['unexistant']:
                global_rec['fp']+=1
                dir_rec['fp']+=1

        recall_g = global_rec['tp']/total
        precision_g = global_rec['tp']/(global_rec['tp']+global_rec['fp'])
        recall_dir = dir_rec['tp']/total
        precision_dir = dir_rec['tp']/(dir_rec['tp']+dir_rec['fp'])
        record = {
            'recall_g': recall_g,
            'precision_g': precision_g,
            'recall_dir': recall_dir,
            'precision_dir': precision_dir
            }

        return record
    
    def _create_caption(self, discovered_pyd, record):
        """
            Create Legend/Caption to be displayed next to the node
        """

        cluster = pydot.Cluster('caption')
        caption = pydot.Node('inner_caption')
        caption.set_shape('box')

        caption.set_label("Legend:\lCorrect: "+ self.labels['correct']+"\lMisdirected: "+\
            self.labels['misdirected']+"\lUnexistant: "+ self.labels['unexistant']+\
            "\l\lRecovering Metrics:\lGlobal Recall: " + str(np.round(record['recall_g'], decimals=3)) \
            +"\lGlobal Precision: " + str(np.round(record["precision_g"], decimals=3)) +\
            "\lCorrect & Directed Recall: " + str(np.round(record['recall_dir'], decimals=3)) +\
            "\lCorrect & Directed Precision: " + str(np.round(record['precision_dir'], decimals=3)))
            
        cluster.add_node(caption)
        discovered_pyd.add_subgraph(cluster)
        
        return discovered_pyd
            
    def grid_search(self, data, alg, param_grid, true_graph):
        """
            Perform grid search on the discovery step.

            Arguments:
                - data: np.array, the data on which perform the discovery task
                - alg: str, currently 'pc' or 'ges', the algorithm used to perform
                the discovery task
                - param_grid: dict, format : {'param_name1': [value1,...], 'param_name2':...}
                dictionnary of parameters with which perform the grid search
                - true_graph: pydot.Graph, the true graph.
        """

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.gs_true_graph = true_graph
        for combination in combinations:
            print('Current parameters used: ', combination)
            graph = self.discover(data, alg, combination)
            self.discovered_graphs.append(graph)

    def _gs_compare(self):

        for discovered in self.discovered_graphs:
            pyd = self.compare(self.gs_true_graph, discovered)
            record = self.discovered_ratio(pyd)
            pyd = self._create_caption(pyd, record)
            self.discovered_graphs_pyd.append(pyd)

    def grid_search_results(self):
        """
        To fix should work without allEdgesCausalStrength
            Get results and add them to the labels in the true graph
            Returns:
                - self.gs_true_graph, the true graph with added labels
        """

        self._gs_compare()
        edges = self.gs_true_graph.get_edges()
        for key in list(self.edge_scores.keys()):

            if 'label' in list(edges[int(key)].obj_dict['attributes'].keys()):
                edges[int(key)].obj_dict['attributes']['label'] += ' | SS: ' +\
                    str(sum(self.edge_scores[key])/len(self.edge_scores[key]))
            else: 
                edges[int(key)].obj_dict['attributes']['label'] = ' | SS: ' +\
                    str(sum(self.edge_scores[key])/len(self.edge_scores[key]))
                    
        return self.gs_true_graph
