from torch.utils.data import Dataset
import torch
import torch.nn as nn

class dataset(Dataset):
    """
        Extends Dataset to take into account counterfactuals.
    """
    
    def __init__(self,factual,target, counterfactuals):

        self.factual = torch.tensor(factual,dtype=torch.float32)
        self.target = torch.tensor(target,dtype=torch.float32).unsqueeze(dim=1)
        self.counterfactuals = [torch.from_numpy(counterfactual.values[:,:-1]).float() for counterfactual in counterfactuals]
        self.length = self.factual.shape[0]
 
    def __getitem__(self,idx):
        return self.factual[idx],self.target[idx],[cf[idx] for cf in self.counterfactuals]
    
    def __len__(self):
        return self.length

class MWCF_loss(nn.Module):
    """
        Deterministic Loss from the following paper:
        
        Russell, C; Kusner, M; Loftus, C; Silva, R; (2017) When Worlds Collide: 
        Integrating Different Counterfactual Assumptions in Fairness. 
        In: Guyon, I and Luxburg, UV and Bengio, S and Wallach, 
        H and Fergus, R and Vishwanathan, S and Garnett, R, 
        (eds.) Advances in Neural Information Processing Systems 30 (NIPS 2017).
        NIPS Proceedings: Long Beach, CA, USA.

    """
    
    def __init__(self, lam=1, eps=0.01):
        super(MWCF_loss, self).__init__()
        self.lam = lam
        self.eps = eps
    
    def forward(self, output, target, counterfactuals_output):
        
        outputs = output.repeat((1,counterfactuals_output.shape[1]))
        zeros = torch.zeros(outputs.shape)
        loss_first = nn.BCELoss()
        o_loss_first = loss_first(output, target)
        inner_ = torch.mean(torch.max((torch.linalg.norm(outputs - counterfactuals_output) - self.eps), zeros), dim=0)
        o_loss_second = torch.sum(inner_)

        return (o_loss_first + (self.lam*o_loss_second))
