import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from calibration_methods.linear_calibrator import LinearCalibrator

def DirichletODIR(num_classes):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    calibrator = LinearCalibrator(use_logits=False, odir=True)
    num_classes=10
    if num_classes== 10:
        start_from = -5.0
    else:
        start_from = -2.0
    # Set regularisation parameters to check through
    lambdas = np.array([10**i for i in np.arange(start_from, 7)])
    lambdas = sorted(np.concatenate([lambdas, lambdas*0.25, lambdas*0.5]))
    mus = np.array([10**i for i in np.arange(start_from, 7)])
    gscv = GridSearchCV(calibrator,
                        param_grid={'reg_lambda':  lambdas,'reg_mu': mus},
                        cv=skf, scoring='neg_log_loss', refit=True, verbose=0, n_jobs=5)
    return gscv

def DirichletL2(num_classes):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    calibrator = LinearCalibrator(use_logits=False, odir=False)
    if num_classes== 10:
        start_from = -5.0
    else:
        start_from = -2.0
    # Set regularisation parameters to check through
    lambdas = np.array([10**i for i in np.arange(start_from, 7)])
    lambdas = sorted(np.concatenate([lambdas, lambdas*0.25, lambdas*0.5]))
    gscv = GridSearchCV(calibrator,
                        param_grid={'reg_lambda':  lambdas},
                        cv=skf, scoring='neg_log_loss', refit=True, verbose=2, n_jobs=5)
    return gscv