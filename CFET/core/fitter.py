import numpy as np
from sklearn.linear_model import LogisticRegression


class LinearSCM():

    def __init__(self, adj):

        self.adj = adj

    
    def fit(self, data):
        noises_col = []
        scm = np.zeros(self.adj.shape)
        for idx, col in enumerate(self.adj.T):
            indices = np.where(col!=0)[0]
            if len(indices) >0:
                inner_data = data.iloc[:,indices]
                model = LogisticRegression()
                model.fit(inner_data, data.iloc[:,[idx]])
                noises = data.iloc[:,[idx]].values - model.predict_proba(inner_data)
                scm[indices,idx] = model.coef_[0]
                noises_col.append(noises)
        
        noises_col = np.hstack(noises_col).T
        self.noises = noises_col
        return np.round(scm, decimals=1)
