import pandas as pd


class Observer:
    """
        Reproduce Observation Bias Process from Real World Data distribution.
    """

    def __init__(self, data, sensible_parameter):

        self.data = data
        self.sensible_parameter = sensible_parameter
    
    def statistical_parity(self, data):
        """
            To be moved to utils
            Assuming that output variable is the last column. Compute Statistical Parity
            Argument:
                - data: pandas.Dataframe, the dataset on which compute the SP
            Returns:
                - Statistical Parity
        """

        output = list(data.columns)[-1]
        values_AnY = data.value_counts(subset=[self.sensible_parameter, output],
                                        normalize=True)
        values_A = data.value_counts(subset=[self.sensible_parameter], normalize= True)
        p_AnY_1= values_AnY[(1,1)]
        p_AnY_0= values_AnY[(0,1)]
        p_A_0 = values_A[0]
        p_A_1 = values_A[1]

        p_cond_1 = p_AnY_1/p_A_1
        p_cond_0 = p_AnY_0/p_A_0

        return (p_cond_1 - p_cond_0)
    
    def CSP(self, data, csp):
        """
            To be moved to utils
            Assuming that the last column is the output variable.
            Compute Conditionnal Statistical Parities on the given
            dataset.

            i.e if csp is [{
                            'keys':['x0','x1','x2','x8'],
                            'values': (1,1,1,1),
                            'prob': 0.7
                        }]
            
            Then the following CSP is computed: 
            P(x8 = 1 | x1 = 1, x2 = 1, x3 = 1, x0 = 1) - P(x8 = 1 | x1 = 1, x2 = 1, x3 = 1, x0 = 0)

            Arguments:
                - data: pandas.Dataframe, the data on which compute the CSP
                - csp: list of dict, csp is as required in describe 
            Returns:
                - the value of the corresponding CSP
        """
        
        sc = list(csp['values'])
        sc[0] = int(not sc[0])
        sc = tuple(sc)
        values_AnCnY_1 = data.value_counts(subset=csp['keys'], normalize=True)[csp['values']]
        values_AnC_1 = data.value_counts(subset=csp['keys'][:-1], normalize=True)[csp['values'][:-1]]

        values_AnCnY_0 = data.value_counts(subset=csp['keys'], normalize=True)[sc]
        values_AnC_0 = data.value_counts(subset=csp['keys'][:-1], normalize=True)[sc[:-1]]

        p_cond_1 = values_AnCnY_1/values_AnC_1
        p_cond_0 = values_AnCnY_0/values_AnC_0

        return (p_cond_1-p_cond_0)
        
    def _sample_CSP(self, sample, C, debug=False):

        assert isinstance(C, list), 'C should be a list'

        for c in C:
            while (sample.value_counts(subset=c['keys'], normalize= True)[c['values']]/\
                sample.value_counts(subset=c['keys'][:-1], normalize=True)[c['values'][:-1]])\
                    <= c['prob']:
                    
                if debug: print((sample.value_counts(subset=c['keys'], normalize= True)[c['values']]/\
                sample.value_counts(subset=c['keys'][:-1], normalize=True)[c['values'][:-1]]))

                new_row = self.data.groupby(c['keys']).get_group(c['values']).sample(n=1)
                complementary_l = list(c['values'])[:-1]
                complementary_l.append(int(not list(c['values'])[-1]))
                complementary = tuple(complementary_l)
                row_rm = sample.groupby(c['keys']).get_group(complementary).index[0] 
                if (sample.index != new_row.index[0]).all():
                    sample.drop(row_rm, inplace=True)
                    sample = pd.concat([sample,new_row])
            print(sample.value_counts(subset=c['keys'], normalize= True)[c['values']]/\
                sample.value_counts(subset=c['keys'][:-1], normalize=True)[c['values'][:-1]])
        
        return sample
    
    def describe(self, sample, CSP):
        """
            Print statistical measures corresponding to the dataset such as: SP, CSP, etc.

            Arguments:
                - sample: pandas.Dataframe, the dataframe on which print the statistical measures
                - CSP: list of dict containing the CSP to be computed and displayed
            
            CSP should look like this: [{
                            'keys':['x0','x1','x2','x8'],
                            'values': (1,1,1,1),
                            'prob': 0.7
                        }]
                Where 'keys' is a list containing variable names
                'values' is a tuple containing variable values
                'prob' key isn't used in this function
        
        """
        print("SP: Real World: ",self.statistical_parity(self.data))
        print("SP: Observed World :", self.statistical_parity(sample))
        
        if CSP != None:
            assert isinstance(CSP, list), "CSP should be a list"
            for csp in CSP:
                toprint = 'P(Y='+str(csp['values'][-1])+'|'
                for idx, key in enumerate(csp['keys'][:-1]):
                    toprint+= key+'='+str(csp['values'][idx]) + ','
                toprint = toprint[:-1] + ')'
                toprint2= toprint[:9] +str(int(not csp['values'][0]))+ toprint[10:]
                print(toprint,'-',toprint2, ': Real World: ' , self.CSP(self.data, csp))
                print(toprint,'-',toprint2, ': Observed World: ' , self.CSP(sample, csp))
            
    def observe(self, bias, n_data = 10000, debug=False):
        """
            Observation process on the real world data distribution.

            Arguments:
                - bias: list of dict, containing the different csp and sp and the
                corresponding probabilities that the observation sample should fit with
                - n_data: int, number of observation in the output sample 

                bias should look like this: [{
                            'keys':['x0','x1','x2','x8'],
                            'values': (1,1,1,1),
                            'prob': 0.7
                        }]
                Where 'keys' are the variable names
                'values' their values
                and prob the probability the sample as to fit with
                i.e the sample should have :
                P(x8 = 1 | x1 = 1, x2 = 1, x0 = 1 ) = 0.7
            
            Returns:
                - sample: pandas.Dataframe, the observed sample       
        """

        sample = self.data.sample(n=n_data)
        #print("SP: Real World: ",self.statistical_parity(self.data))
        #print("SP: Observed World :", self.statistical_parity(sample))

        if bias != None:
            sample = self._sample_CSP(sample,bias, debug=debug)
            #print("SP: Biased World :", self.statistical_parity(sample))
        self.sample_index = sample.index
        return sample