import torch
import numpy as np
from scipy.optimize import minimize
from calibration_methods.temperature_scaling import TemperatureScaling

class EnsembleTemperatureScaling():
    
    def __init__(self, loss='mse'):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.loss = loss
    
    def _ce_loss_fun(self, w, p0, p1, p2, labels):
        # Calculates the cross-entropy loss
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        loss = torch.nn.CrossEntropyLoss()(p, labels)
        return loss
    
    def _mse_loss_fun(self, w, *args):
        # Calculates the MSE loss
        p0, p1, p2, labels = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        p = torch.nn.functional.normalize(p,p=1)
        
        labels_onehot = torch.DoubleTensor(p0.shape[0], p0.shape[1])
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)

        loss = torch.mean(torch.sum((p - labels_onehot) ** 2, dim=1,keepdim = True))
        return loss
        
    # Find the temperature
    def fit(self, logits, labels, verbose=False):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        
        # First, we compute temperature scaling
        ts_calibrator = TemperatureScaling(loss=self.loss)
        ts_calibrator.fit(logits, labels)
        self.temp = ts_calibrator.temp
        
        p0 = torch.nn.Softmax(dim=1)(logits/self.temp)
        p1 = torch.nn.Softmax(dim=1)(logits)
        p2 = torch.ones_like(logits) / logits.shape[1]

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        if self.loss == 'ce':
            opt = minimize(self._ce_loss_fun, (1.0, 0.0, 0.0) , args = (p0,p1,p2,labels), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': verbose})
        elif self.loss == 'mse':
            opt = minimize(self._mse_loss_fun, (1.0, 0.0, 0.0) , args = (p0,p1,p2,labels), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-10, options={'disp': verbose})
        self.w = opt.x
        if verbose:
            print(f'Temperature: {self.temp:1.2f}'+" Weights = " +str(self.w))

        return opt


    def predict_proba(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        p0 = torch.nn.Softmax(dim=1)(logits/self.temp)
        p1 = torch.nn.Softmax(dim=1)(logits)
        p2 = torch.ones_like(logits) / logits.shape[1]
        
        return self.w[0]*p0 + self.w[1]*p1 + self.w[2]*p2    

