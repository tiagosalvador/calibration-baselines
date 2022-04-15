import torch
from sklearn.isotonic import IsotonicRegression

class IROvA():
    
    def __init__(self):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        
    # Find the temperature
    def fit(self, logits, labels):
        logits = logits.double()
        softmaxes = torch.nn.Softmax(dim=1)(logits)
        labels_onehot = torch.DoubleTensor(logits.shape[0], logits.shape[1])
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)

        self.ir = {}
        for ii in range(logits.shape[1]):
            self.ir[ii] = IsotonicRegression(out_of_bounds='clip')
            self.ir[ii].fit_transform(softmaxes[:,ii], labels_onehot[:,ii])
        
        return self.ir
        
    def predict_proba(self, logits):
        """
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        logits = logits.double()
        softmaxes = torch.nn.Softmax(dim=1)(logits)
        for ii in range(logits.shape[1]):
            softmaxes[:,ii] = torch.from_numpy(self.ir[ii].predict(softmaxes[:,ii]))+1e-9*softmaxes[:,ii]
        return softmaxes