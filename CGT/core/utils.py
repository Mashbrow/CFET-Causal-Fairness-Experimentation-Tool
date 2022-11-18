import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg
import numpy as np

def show(graph):
    """
        Display graph according to adjency matrix
    """
    tmp_png = graph.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.figure(figsize=(20,120))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def KL(p : list, q : list):
        
    div = [p[i] * np.log((p[i]/q[i])) for i in range(len(p))]

    result = sum(div)

    return result


def graph2adj(graph):
    """
        Retrieve the adjency matrix that corresponds to the causal graph.
        Where adjency[i,j] = 1 corresponds to the link x_i -> x_j
        If an edge is of type x -- y or x <-> y then  adjency[i,j] = 2.
        In that case it may be required to use the handle_dcausation function.

        Argument:
            - graph: causallearn.CausalGraph, the graph on which get the adjency matrix
        Returns:
            - adjency: np.array, the adjency matrix.
    
    """

    triu = np.triu(graph)
    indexes = [(x,y) for x, y in zip(np.where(triu != 0)[0],np.where(triu != 0)[1])]
    adjency= np.zeros(graph.shape)
    for id in indexes:
        if (graph[id] == -1) & (graph[id[::-1]] == 1):
            adjency[id] = 1
        if (graph[id] == -1) & (graph[id[::-1]] == -1):
            adjency[id] = 2
        if (graph[id] == 1) & (graph[id[::-1]] == 1):
            adjency[id] = 2
        if (graph[id] == 1) & (graph[id[::-1]] == -1):
            adjency[id[::-1]] = 1
        
    return adjency

def handle_dcausation(adjency):
    """
        Handle double causation links, i.e. those with x -- y and x <-> y, by 
        creating two adjency matrix for each double causation such that
        one contains x -> y and another contains y -> x.

        Argument:
            - adjency: np.array, the adjency matrix to handle double causation the adjency
                                 matrix should have its problematic edges encoded with a 2
                                 as in graph2adj function
        Returns:
            - adjency_list: list[np.array()], the list of adjency matrix without double causations. 

    """
    adjency_list = []
    adjency_list.append(adjency)
    i=0
    while i < len(adjency_list):
        while 2 in adjency_list[i]:
            adj = adjency_list.pop(i)
            dcaus_idx = [(x,y) for x, y in zip(np.where(adj == 2)[0],np.where(adj == 2)[1])][0]
            copy1 = adj.copy()
            copy2 = adj.copy()
            copy1[dcaus_idx] = 1
            copy2[dcaus_idx[::-1]] = 1
            copy2[dcaus_idx] = 0
            adjency_list.append(copy1)
            adjency_list.append(copy2)
        i+=1
    return adjency_list


def check_cycles(adjency):
    """
        Check if the given adjency matrix contains cycles.

        Argument:
            - adjency: np.array(), an adjency matrix where adjency[i,j] = 1 corresponds to
                                   the edje x_i -> x_j
            - debug: bool, if True the function returns the length of the cycle that was found

        Returns:
            - bool, True if the adjency matrix contains a cycle, False otherwise.
    """

    A_i = np.eye(adjency.shape[0], adjency.shape[1])
    for i in range(adjency.shape[0]):
        A_i = A_i @ adjency
        if sum(np.diag(A_i)) != 0:
            return True, i
    
    return False, 0