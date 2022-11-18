from sklearn.metrics import log_loss
import torch
import torch.nn as nn

## Pytorch

class Model(torch.nn.Module):

    def __init__(self, in_var):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(in_var, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x= self.linear2(x)
        x = torch.sigmoid(x)
        return x
    
    def eval_cf(self, cf0, cf1):

        out0 = self.forward(cf0)
        out1 = self.forward(cf1)
        bin_out0 = (out0 >0.5).int()
        bin_out1 = (out1 >0.5).int()

        tps = sum((bin_out0 == bin_out1) & (bin_out1 == 1))
        tns = sum((bin_out0 == bin_out1) & (bin_out1 == 0))
        fns = sum((bin_out0 != bin_out1) & (bin_out1 == 0))
        fps = sum((bin_out0 != bin_out1) & (bin_out1 == 1))

        cf_accu = (tps+tns)/(tps+tns+fps+fns)

        return cf_accu
    
    def evalmodified(self, testloader):

        tps=0
        tns=0
        fps=0
        fns=0

        for j, (x_test, y_test, cf_test) in enumerate(testloader):
            output = self.forward(x_test)
            bin_out = (output>0.5).int()
            tps += sum((bin_out == y_test) & (bin_out == 1))
            tns += sum((bin_out == y_test) & (bin_out == 0))
            fns += sum((bin_out != y_test) & (bin_out == 0))
            fps += sum((bin_out != y_test) & (bin_out == 1))
            

        f1 = tps/(tps + (0.5*(fns+fps)))
        precision = tps/(tps+fps)
        recall = tps/(tps+fns)
    
        return f1, precision, recall

