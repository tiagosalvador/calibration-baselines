import torch
from sklearn.isotonic import IsotonicRegression

class IRM():
    
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

        self.ir = IsotonicRegression(out_of_bounds='clip')
        return self.ir.fit_transform(softmaxes.flatten(), (labels_onehot.flatten()))

        
    def predict_proba(self, logits):
        """
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        logits = logits.double()
        softmaxes = torch.nn.Softmax(dim=1)(logits)
        output = torch.from_numpy(self.ir.predict(softmaxes.flatten()))
        output = output.reshape(logits.shape)+1e-9*softmaxes
        return output
