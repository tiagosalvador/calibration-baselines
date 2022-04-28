import torch
import numpy as np
from scipy.optimize import minimize 
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader

class LogitsDataset(Dataset):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]

class DiagonalModel(torch.nn.Module):
    """
    Create a diagonal model

    Params:
        classes (int): number of classes, used for input layer shape and output shape
        use_logits (bool): Using logits as input of model, leave out conversion to logarithmic scale.        
    Returns:
        model (object): Pytorch model
    """

    def __init__(self, num_classes, W_init=[], b_init=[], use_logits=True):
        super(DiagonalModel, self).__init__()
        self.use_logits = use_logits
        if len(W_init) != 0:
            self.W = torch.nn.Parameter(W_init)
        else:
            self.W = torch.nn.Parameter(torch.ones(num_classes))
        if len(b_init) != 0:
            self.b = torch.nn.parameter.Parameter(b_init)
        else:
            self.b = torch.nn.parameter.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        out = torch.matmul(logits, torch.diag(self.W)) + self.b
        return out

class VectorScaling():
    
    def __init__(self, max_epochs = 500, patience = 50, lr = 0.01, random_state = 15,
                 double_fit = True, device='cpu'):
        """
        Initialize class
        
        Params:
            max_epochs (int): maximum number of epochs done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and PyTorch
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.01), and the second time 10x lower lr (lr/10)
            device: device where to train the model (default: cpu)
        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weights = []
        self.random_state = random_state
        self.double_fit = double_fit
        self.device = device
        self.min_delta = 0.0
        
        # Setting random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    
    def fit(self, logits, labels, W_init=[], b_init=[], verbose = False, double_fit=None, batch_size = 128):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        num_classes = logits.shape[1]
        self.model = DiagonalModel(num_classes=num_classes, W_init=W_init, b_init=b_init)
        logits_dataloader = DataLoader(LogitsDataset(logits, labels), batch_size=batch_size, shuffle=True)

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
                loss = torch.nn.CrossEntropyLoss()(out, y.to(self.device))
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
            return self.fit(logits, labels, W_init=self.model.W, b_init=self.model.b, verbose=verbose, double_fit=False, batch_size = batch_size)  # Fit 2 times
        
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