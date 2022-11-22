import pydot
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz

from core.discovery import Discoverer
from core.generator import CausalGenerator

class test_Discoverer:
    ### /!\ need to add Generator to work

    def __init__(self, data):

        self.discoverer = Discoverer()
        pc_param = {'alpha' : 0.05, 'indep_test':gsq}
        self.dgraph = self.discoverer.discover(data, 'pc', pc_param)
        self.dgraph_pyd = GraphUtils.to_pydot(self.dgraph)

    def test_areEqual(self):

        edge1 = pydot.Edge('x0','x1', dir='forward')
        edge2 = pydot.Edge('x1','x0', dir='forward')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True,False)

        edge1 = pydot.Edge('x0','x1', dir='forward')
        edge2 = pydot.Edge('x0','x1', dir='forward')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, True)

        edge1 = pydot.Edge('x0','x1', dir='forward')
        edge2 = pydot.Edge('x0','x1', dir='none')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, False)

        edge1 = pydot.Edge('x0','x1', dir='forward')
        edge2 = pydot.Edge('x0','x1', dir='both')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, False)

        edge1 = pydot.Edge('x0','x1', dir='none')
        edge2 = pydot.Edge('x0','x1', dir='none')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, True)

        edge1 = pydot.Edge('x0','x1', dir='none')
        edge2 = pydot.Edge('x1','x0', dir='none')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, True)

        edge1 = pydot.Edge('x0','x1', dir='both')
        edge2 = pydot.Edge('x0','x1', dir='both')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, True)

        edge1 = pydot.Edge('x1','x0', dir='both')
        edge2 = pydot.Edge('x0','x1', dir='both')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (True, True)

        edge1 = pydot.Edge('x2','x1', dir='none')
        edge2 = pydot.Edge('x0','x1', dir='none')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (False, False)

        edge1 = pydot.Edge('x2','x1', dir='both')
        edge2 = pydot.Edge('x0','x1', dir='both')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (False, False)

        edge1 = pydot.Edge('x2','x1', dir='forward')
        edge2 = pydot.Edge('x0','x1', dir='forward')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (False, False)

        edge1 = pydot.Edge('x2','x1', dir='none')
        edge2 = pydot.Edge('x0','x1', dir='forward')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (False, False)

        edge1 = pydot.Edge('x2','x1', dir='none')
        edge2 = pydot.Edge('x0','x1', dir='both')
        result = self.discoverer.areEqual(edge1,edge2)
        assert result == (False, False)

    def test_from_discovered_get_edges(self):

        expected_results = [
            'x0 -- x1  [dir=both]',
            'x2 -- x0 [dir=forward]',
            'x1 -- x3 [dir=both]',
            'x2 -- x3 [dir=forward]',
            'x3 -- x4 [dir=forward]']

        edges = self.discoverer.from_discovered_get_edges(self.dgraph)

        for idx, edge in enumerate(edges):
            print(edge.to_string(), expected_results[idx])


