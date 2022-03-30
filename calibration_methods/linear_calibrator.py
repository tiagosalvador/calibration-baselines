import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
             
def get_off_diagonal_elements(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res

def cross_entropy_regularization(logits, labels, parameters, reg_lambda, reg_mu, odir):
    """
        logits
        labels
        parameters
        reg_lambda (float): lambda regularization parameter for off-diag terms.
        reg_mu (float): mu regularization parameter for bias terms.
        odir (bool): whether use complementary ODIR regularization or not.
    """

    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    num_classes = logits.shape[1]    
    if odir:
        for a_name, a in parameters:
            if a_name == 'W':
                loss += reg_lambda*get_off_diagonal_elements(a.pow(2)).sum()/(num_classes * (num_classes + 1))
            elif a_name == 'b':
                k = logits.shape[1]
                loss += reg_mu*a.pow(2).sum()/num_classes
            else:
                raise(ValueError('Parameter {} was not expected'.format(a_name)))
    else:
        for a_name, a in parameters:
            if a_name == 'W':
                k = logits.shape[1]
                loss += reg_lambda*a.pow(2).sum()/(num_classes * num_classes + num_classes)
            elif a_name == 'b':
                k = logits.shape[1]
                loss += reg_mu*a.pow(2).sum()/(num_classes * num_classes + num_classes)
            else:
                raise(ValueError('Parameter {} was not expected'.format(a_name)))
    return loss


class LogitsDataset(Dataset):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]

class LinearModel(torch.nn.Module):
    """
    Create a linear model

    Params:
        classes (int): number of classes, used for input layer shape and output shape
        use_logits (bool): Using logits as input of model, leave out conversion to logarithmic scale.        
    Returns:
        model (object): Pytorch model
    """

    def __init__(self, num_classes, W_init=[], b_init=[], use_logits=True):
        super(LinearModel, self).__init__()
        self.use_logits = use_logits
        if len(W_init) != 0:
            self.W = torch.nn.Parameter(W_init)
        else:
            self.W = torch.nn.Parameter(torch.eye(num_classes))
        if len(b_init) != 0:
            self.b = torch.nn.parameter.Parameter(b_init)
        else:
            self.b = torch.nn.parameter.Parameter(torch.zeros(num_classes))

    def forward(self, inputs):
        if self.use_logits:
            logits = inputs
        else:
            eps = np.finfo(float).eps  # 1e-16
            logits = torch.log(torch.clip(inputs, eps, 1 - eps))
        out = torch.matmul(logits, self.W) + self.b
        return out


class LinearCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda = 0., reg_mu = None, max_epochs = 500, odir = True,
                 patience = 15, lr = 0.001, random_state = 15,
                 double_fit = True, device='cpu', use_logits = False):
        """
        Initialize class
        
        Params:
            reg_lambda (float): lambda regularization parameter for L2 or ODIR regularization.
            mu (float): mu regularization parameter for bias. (If None, then it is set equal to lambda of reg_lambda)
            max_epochs (int): maximum number of epochs done by optimizer.
            odir (bool): whether use ODIR regularization or not.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and PyTorch
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)
            device: device where to train the model (default: cpu)
            use_logits (bool): Using logits as input of model, leave out conversion to logarithmic scale.
        """
        
        self.reg_lambda = reg_lambda
        if reg_mu is None:
            self.reg_mu = reg_lambda
        else:
            self.reg_mu = reg_mu
        self.max_epochs = max_epochs
        self.odir = odir
        self.patience = patience
        self.lr = lr
        self.weights = []
        self.random_state = random_state
        self.double_fit = double_fit
        self.device = device
        self.use_logits = use_logits
        self.min_delta = 0.0
        
        # Setting random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, inputs, labels, W_init=[], b_init=[], verbose = False, double_fit=None, batch_size = 128):#, *args, **kwargs):
        """
        Trains the model and finds optimal parameters
        
        Params:
            inputs: the output from neural network for each class (shape [samples, classes])
            labels: true labels.
            verbose (bool): whether to print out anything or not
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)
            
        Returns:
            self: 
        """
        if verbose:
            print('Fitting lambda=%1.2f, mu=%1.2f'%(self.reg_lambda, self.reg_mu))
        num_classes = inputs.shape[1]
        self.model = LinearModel(num_classes=num_classes, W_init=W_init, b_init=b_init)
        logits_dataloader = DataLoader(LogitsDataset(inputs, labels), batch_size=batch_size, shuffle=True)

        if double_fit == None:
            double_fit = self.double_fit
        
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss = None
        patience_counter = 0
        for epoch in range(self.max_epochs):
            if patience_counter == self.patience:
                break
            loss_epoch = []
            for batch, (X, y) in enumerate(logits_dataloader):
                optimizer.zero_grad()
                out = self.model(X.to(self.device))
                loss = cross_entropy_regularization(out, y.to(self.device), self.model.named_parameters(), self.reg_lambda, self.reg_mu, self.odir)
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())

            # Early Stopping if for the last patience epochs the loss has not decreased by at least min_delta
            loss = np.mean(loss_epoch)
            if best_loss is None:
                best_loss = loss
            else:
                if loss-best_loss  < -self.min_delta:
                    patience_counter = 0
                    best_loss = loss
                else:
                    patience_counter += 1    
            if verbose:
                if epoch % 10 == 0:
                    print(f"loss: {loss:>7f}  [{epoch:>3d}/{self.max_epochs:>3d}] [{patience_counter:>3d}/{self.patience:>3d}]")
        self.model.eval()
        self.model.cpu()
        
        if double_fit:
            if verbose:
                print("Fit with 10x smaller learning rate")
            self.lr = self.lr/10
            return self.fit(inputs, labels, W_init=self.model.W, b_init=self.model.b, verbose=verbose, double_fit=False, batch_size = batch_size)  # Fit 2 times
        
        return self

    def predict_proba(self, inputs):
        """
        Scales inputs based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        with torch.no_grad():
            return torch.nn.Softmax(dim=1)(self.model(inputs))

    def predict(self, inputs):
        """
        Scales inputs based on the model and returns calibrator outputs (logit scale)
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrator outputs (logit scale) (nd.array with shape [samples, classes])
        """
        with torch.no_grad():
            return self.model(inputs)

    @property
    def coef_(self):
        """
        Matrix weights
        """
        return self.model.W

    @property
    def intercept_(self):
        """
        Bias weights
        """
        return self.model.b