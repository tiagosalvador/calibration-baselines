import torch
from scipy.optimize import minimize 

class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, T, logits, labels):
        # Calculates the cross-entropy loss
        loss = torch.nn.CrossEntropyLoss()(logits/T, labels)
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
        
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, labels), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        if verbose:
            print("Temperature:", self.temp)
        
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
        
        if not temp:
            return torch.nn.Softmax(dim=1)(logits/self.temp)
        else:
            return torch.nn.Softmax(dim=1)(logits/temp)


    def predict(self, logits, temp = None):
        """
        Scales inputs based on the model and returns calibrator outputs (logit scale)
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrator outputs (logit scale) (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return logits/self.temp
        else:
            return logits/temp