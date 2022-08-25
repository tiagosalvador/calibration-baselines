import torch
import numpy as np
# import os
from tqdm import tqdm
# import pandas as pd

# from networks import load_net
# from datasets import load_ds_info
# from utils import get_features_logits_labels

# import torch
# import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer

from uncertainty_measures import get_ece

def CCAC(method):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    calibrator = CCACbase(method, device='cuda')
    lambdas = [5, 10, 15, 20, 25, 50]
    gscv = GridSearchCV(calibrator,
                        param_grid={'lambda_1':  lambdas,'lambda_2': lambdas},
                        cv=skf,
                        scoring=make_scorer(get_ece_cv, greater_is_better=False),
                        verbose=0, refit=True,
                        n_jobs=3,error_score='raise')
    return gscv


def get_ece_cv(labels_true, logits_pred):
    ground_truth = (logits_pred[:,:-1].argmax(dim=1) == labels_true).cpu().numpy()    
    softmaxes = torch.nn.Softmax(dim=1)(logits_pred)
    prob_top1 = softmaxes[:,:-1].max(dim=1).values
    prob_misclassified = softmaxes[:,-1]
    confidence = 1-torch.sqrt((1-prob_top1)*prob_misclassified)
    return get_ece(confidence.numpy(), ground_truth, 15)*100

class LossCCAC(torch.nn.Module):
    def __init__(self, lambda_1=0, lambda_2=0, num_classes=10):
        super(LossCCAC, self).__init__()
        self.num_classes = num_classes
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, logits, labels, ground_truth):
        labels_ccac = labels.clone()
        labels_ccac[torch.logical_not(ground_truth)] = self.num_classes
        softmaxes = torch.nn.Softmax(dim=1)(logits)
        loss_1 = -torch.log(softmaxes[np.arange(len(labels_ccac))[ground_truth],labels_ccac[ground_truth]]).sum()
        loss_2 = -self.lambda_1*torch.log(1-softmaxes[ground_truth,-1]).sum()
        loss_3 = -self.lambda_2*torch.log(softmaxes[torch.logical_not(ground_truth),-1]).sum()
        loss = (loss_1+loss_2+loss_3)/len(logits)
        return loss

class ModelCCAC(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ModelCCAC, self).__init__()
        if num_classes==10:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(10, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 11),
            )
        elif num_classes==100:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 101),
            )

    def forward(self, x):
        x = self.model(x)
        return x

    
class ModelCCACS(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ModelCCACS, self).__init__()
        if num_classes==10:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(10, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1),
            )
        elif num_classes==100:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 1),
            )
        self.temperature = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        
    def forward(self, x):
        return torch.hstack([x/self.temperature,self.model(x)])



class CCACbase(BaseEstimator, RegressorMixin):
    def __init__(self, method, lambda_1 = 0.0, lambda_2 = 0.0, epoch_check=5, max_epochs = 150, batch_size=128,
                 lr = 1e-2, random_state = 15, device='cpu'):
        
        self.method = method
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epoch_check = epoch_check
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = device
        self.patience = 15
        self.min_delta = 0.0
        
        # Setting random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, logits, labels, verbose = False):
        if verbose:
            tqdm.write(f'Fitting lambda_1={self.lambda_1:1.2f}, lambda_2={self.lambda_2:1.2f}')
            
        train_ds = torch.utils.data.TensorDataset(logits, labels)
        train_set_size = int(len(train_ds) * 0.8)
        valid_set_size = len(train_ds) - train_set_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_set_size, valid_set_size])

        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, multiprocessing_context='fork')
        logits_val = logits[np.array(val_ds.indices)]
        labels_val = labels[np.array(val_ds.indices)]
        # Initialize model
        if self.method == 'CCAC':
            self.model = ModelCCAC(num_classes=logits.shape[1])
            best_model = ModelCCAC(num_classes=logits.shape[1])
        elif self.method == 'CCAC-S':
            self.model = ModelCCACS(num_classes=logits.shape[1])
            best_model = ModelCCACS(num_classes=logits.shape[1])
        # Initialize the loss function
        loss_fn = LossCCAC(lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        # Initialize optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 125])            
        
        self.model.to(self.device)
        self.model.train()
        best_loss = None
        best_ece = 100.0

        patience_counter = 0
        for epoch in range(self.max_epochs):
            if patience_counter == self.patience:
                break
            loss_epoch = []
            for batch, (logits, labels) in enumerate(train_dataloader):
                logits = logits.to(self.device)
                labels = labels.to(self.device)
                # Compute prediction and loss
                new_logits = self.model(logits)
                ground_truth = (logits.argmax(dim=1) == labels).cpu()
                loss = loss_fn(new_logits, labels, ground_truth)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())
            
            lr_scheduler.step()
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
            
            ground_truth = (self.model(logits_val.to(self.device))[:,:-1].argmax(dim=1) == labels_val.to(self.device)).cpu().numpy()
            with torch.no_grad():
                confidences = self.predict_proba(logits_val.to(self.device)).cpu().max(dim=1).values.numpy()
                ece_val = get_ece(confidences, ground_truth, 15)*100
                
            if ece_val < best_ece:
                best_ece = ece_val
                best_model.load_state_dict(self.model.state_dict())
            if verbose:
                if epoch % 25 == 0:
                    print(f"Loss: {loss:>7f} ECE (val):{ece_val:1.2f} [{epoch:>3d}/{self.max_epochs:>3d}] [{patience_counter:>3d}/{self.patience:>3d}]")

        self.model.eval()
        self.model.cpu()
        self.model = best_model.eval().cpu()

        return self

    def predict_proba(self, inputs):
        with torch.no_grad():
            softmaxes = torch.nn.Softmax(dim=1)(self.model(inputs))
            prob_top1 = softmaxes[:,:-1].max(dim=1).values
            prob_misclassified = softmaxes[:,-1]
            temp = torch.sqrt(softmaxes[:,:-1]*(1-prob_misclassified.unsqueeze(1)))            
            probs = temp*((1-temp.max(dim=1).values)/(temp.sum(dim=1)-temp.max(dim=1).values)).unsqueeze(1)
            probs[np.arange(len(temp)),temp.argmax(dim=1)] = temp.max(dim=1).values
            return probs

    def predict(self, inputs):
        with torch.no_grad():
            return self.model(inputs)
