import numpy as np
import sys
import unittest

sys.path.append("..")
from core.generator import CausalGenerator


class TestCausalGenerator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCausalGenerator, self).__init__(*args, **kwargs)

        self.adj = np.array([[0, 0.3, 0.7, 0, 0], \
                        [0, 0, 0, 1, 0], \
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])
        self.generator = CausalGenerator(self.adj, var_names=None, size=100000)
        self.generator.generate_data()
    
    def test_init(self):
        self.assertTrue(np.all(self.generator.sigmas <1) & np.all(self.generator.sigmas>=0))
        
    
    def test_getChildren(self):
        children, idx = self.generator.get_children('x0')
        self.assertTrue(children == ['x1', 'x2'])
        self.assertTrue(idx == [1, 2])
    
    def test_getParents(self):
        parents, idx = self.generator.get_parents('x3')

        self.assertTrue(parents == ['x1','x2'])
        self.assertTrue(idx == [1,2])

    def test_generateData(self):
        
        self.assertFalse(self.generator.data.isnull().values.any())
        self.assertTrue(len(self.generator.data.columns) == self.adj.shape[0])

    def test_CPD(self):
        
        tuple_values = (0,1,0,0,1)
        num = self.generator.data.loc[(self.generator.data['x3'] == 0) &\
                                    (self.generator.data['x1'] == 1) &\
                                    (self.generator.data['x2'] == 0)]
        quo = self.generator.data.loc[(self.generator.data['x1'] == 1) & (self.generator.data['x2'] == 0)]
        ratio = len(num.index)/len(quo.index)
        result = self.generator.CPD('x3', tuple_values)
        self.assertTrue(np.allclose([ratio], [result], atol=0.001))

    def test_isBayesianNetwork(self):

        tol = 0.005
        joined_prob = self.generator.data.value_counts(normalize=True)
        computed_by_fact = []

        def __compute_bayes_fact(node_values):
            result = 1
            for var in self.generator.var_names:
                result *= self.generator.CPD(var, node_values)
            return result
        
        for node_values in joined_prob.index:
            computed_by_fact.append(__compute_bayes_fact(node_values=node_values))
            
        self.assertTrue(np.allclose(joined_prob.values, np.array(computed_by_fact), atol=tol))
    
    
if __name__ == '__main__':
    unittest.main()  