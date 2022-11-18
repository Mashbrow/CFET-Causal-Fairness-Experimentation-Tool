from sklearn.metrics import confusion_matrix
from .utils import KL

class Assessor():

    def __init__(self, sensible_parameter, output_variable, data, bias):

        self.sensible_parameter = sensible_parameter
        self.output = output_variable
        self.data = data
        self.bias = bias

        self.data0 = self.data[self.data[self.sensible_parameter] == 0]
        self.data1 = self.data[self.data[self.sensible_parameter] == 1]

        self.tn0, self.fp0, self.fn0, self.tp0 = confusion_matrix(self.data0.iloc[:,-1],\
                                                                 self.data0.iloc[:,-2]).ravel()
        self.tn1, self.fp1, self.fn1, self.tp1 = confusion_matrix(self.data1.iloc[:,-1],\
                                                                 self.data1.iloc[:,-2]).ravel()

    def statistical_parity(self, data):
        """
            To be moved to utils
            Assuming that output variable is the last column. Compute Statistical Parity
            Argument:
                - data: pandas.Dataframe, the dataset on which compute the SP
            Returns:
                - Statistical Parity
        """

        #output = list(data.columns)[-2]
        values_AnY = data.value_counts(subset=[self.sensible_parameter, self.output],
                                        normalize=True)
        values_A = data.value_counts(subset=[self.sensible_parameter], normalize= True)
        p_AnY_1= values_AnY[(1,1)] if (1,1) in values_AnY.index else 0
        p_AnY_0= values_AnY[(0,1)] if (0,1) in values_AnY.index else 0
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
        if (csp['values'] in data.value_counts(subset=csp['keys']).index) &\
             (csp['values'][:-1] in data.value_counts(subset=csp['keys'][:-1]).index):
            values_AnCnY_1 = data.value_counts(subset=csp['keys'], normalize=True)[csp['values']]
            values_AnC_1 = data.value_counts(subset=csp['keys'][:-1], normalize=True)[csp['values'][:-1]]
            p_cond_1 = values_AnCnY_1/values_AnC_1
        else:
            p_cond_1 = 0
        
        if (sc in data.value_counts(subset=csp['keys']).index) &\
             (sc[:-1] in data.value_counts(subset=csp['keys'][:-1]).index):

            values_AnCnY_0 = data.value_counts(subset=csp['keys'], normalize=True)[sc]
            values_AnC_0 = data.value_counts(subset=csp['keys'][:-1], normalize=True)[sc[:-1]]
            p_cond_0 = values_AnCnY_0/values_AnC_0
        else:
            p_cond_0 =0

        return (p_cond_1-p_cond_0)
    
    def overall_accuracy(self):
        """
            Compute Overall Accuracy Metric as:
                OA = Accuracy_0 - Accuracy_1
        """

        accu0 = (self.tp0 + self.tn0)/(self.tp0 + self.tn0 + self.fp0 + self.fn0)
        accu1 = (self.tp1 + self.tn1)/(self.tp1 + self.tn1 + self.fp1 + self.fn1)

        return accu0-accu1
    
    def predictive_parity(self):
        """
            Compute Predictive Parity Metric as:
                PP = PPV_0 - PPV_1
        """
        
        ppv0 = self.tp0/(self.tp0 + self.fp0)
        ppv1 = self.tp1/(self.tp1 + self.fp1)

        return ppv0-ppv1
    
    def predictive_equality(self):
        """
            Compute Predictive Equality Metric as:
                PE = FPR_0 - FPR_1
        """
        
        fpr0 = self.fp0/(self.fp0 + self.tn0)
        fpr1 = self.fp1/(self.fp1 + self.tn1)

        return fpr0-fpr1

    def equal_opportunity(self):
        """
            Compute Equal Opportunity Metric as:
                EO = FNR_0 - FNR_1
        """

        fnr0 = self.fn0/(self.fn0 + self.tp0)
        fnr1 = self.fn1/(self.fn1 + self.tp1)

        return fnr0 - fnr1

    def treatment_equality(self):
        """
            Compute Treatment Equality as:
                TE = FN_0/FP_0 - FN_1/FP_1
        """

        treat0 = self.fn0/self.fp0
        treat1 = self.fn1/self.fp1
        
        return treat0 - treat1

    def baserates(self):
        """
            Compute baserates
        """
        
        baserate0 = (self.tp0 + self.fn0)/(self.tp0 + self.fn0 + self.tn0 + self.fp0)
        baserate1 = (self.tp1 + self.fn1)/(self.tp1 + self.fn1 + self.tn1 + self.fp1)

        return baserate0 - baserate1
    
    def accuracy(self):

        accu = (self.tp0 + self.tp1 + self.tn1 + self.tn0)/\
                (self.tp0 + self.tp1 + self.tn1 + self.tn0 + self.fp1 +  self.fp0 + self.fn1 + self.fn0)
            
        return accu
    
    def KL_fairness(self, extracted_var):
        """
            Compute KL between two groups corresponding to the sensible parameter
            Argument:
                - extracted_var: list of str, the var that has been kept for in the dataset.
            Returns:
                KL Divergence
        
        """
        
        if self.output not in extracted_var: extracted_var.append(self.output)
        
        indices = list(set(self.data0.value_counts(subset=extracted_var, normalize=True).index) &\
                         set(self.data1.value_counts(subset=extracted_var, normalize=True).index))

        dens_0 = self.data0.value_counts(subset=extracted_var, normalize=True)[indices].values
        dens_1 = self.data1.value_counts(subset=extracted_var, normalize=True)[indices].values
        
        return KL(dens_0, dens_1)

    def compute_fairness(self, metrics, extracted_var=None):
        """
            Compute fairness on the dataset.
            
            Arguments:
                - metrics : list of str, the metrics to be computed
                - extracted_var : the var that has been kept in the dataset
            
            Returns:
                - record : dict containing the results for each metric
        
        """

        record = {}

        for metric in metrics:
            score_=0
            if metric == 'statistical_parity':
                score_ = self.statistical_parity(self.data)
            elif metric == 'CSP':
                score_ = {}
                for csp in self.bias:
                    score_k =''
                    csp_score = self.CSP(self.data, csp)
                    for key, value in zip(csp['keys'], csp['values']):
                        score_k+= str(key) + '=' + str(value) + ''
                    score_[score_k] = csp_score
            elif metric == 'overall_accuracy':
                score_ = self.overall_accuracy()
            elif metric == 'predictive_parity':
                score_ = self.predictive_parity()
            elif metric == 'equal_opportunity':
                score_ = self.equal_opportunity()
            elif metric == 'treatment_equality':
                score_ = self.treatment_equality()
            elif metric == 'baserates':
                score_ = self.baserates()
            elif metric == 'KL_fairness':
                assert not isinstance(extracted_var, type(None)), 'require extracted_var'
                score_ = self.KL_fairness(extracted_var=extracted_var)
            elif metric == 'accuracy':
                score_ = self.accuracy()
            else:
                raise 'metric {} not yet implemented'.format(metric)

            record[metric] = score_
        
        return record