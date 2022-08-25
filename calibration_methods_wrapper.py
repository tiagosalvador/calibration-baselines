from pytorchcv.model_provider import get_model as ptcv_get_model
from tqdm import tqdm
from imgclsmob.pytorch.utils import prepare_model as prepare_model_pt

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

from utils import create_folder, get_features_logits_labels
from uncertainty_measures import get_uncertainty_measures

from scipy import optimize
from scipy.optimize import minimize
from sklearn import linear_model
import scipy.stats

from calibration_methods.temperature_scaling import TemperatureScaling
from calibration_methods.vector_scaling import VectorScaling
from calibration_methods.matrix_scaling import MatrixScaling, MatrixScalingODIR
from calibration_methods.dirichlet_calibration import DirichletL2, DirichletODIR
from calibration_methods.ensemble_temperature_scaling import EnsembleTemperatureScaling
from calibration_methods.irova import IROvA
from calibration_methods.irova_ts import IROvATS
from calibration_methods.irm import IRM
from calibration_methods.irm_ts import IRMTS
from calibration_methods.ccac import CCAC

from calibration_methods_ood.perturbed import Perturbed
from calibration_methods_ood.adaptive_perturbed import AdaptivePerturbed
from calibration_methods_ood.forged import Forged_v1

def get_calibrators(methods, net, ds_info):
    experiments_folder = ds_info['folder']
    logits_clean_cal = None
    calibrators = {}
    for method in tqdm(methods, desc="Prepping Methods", leave=False):
        file = os.path.join(experiments_folder, 'calibration_methods', method+'.npy')
        if os.path.exists(file):
            calibrators[method] = np.load(file, allow_pickle=True).item()
        else:
            if method == 'Vanilla':
                calibrators[method] = None
            else:
                if logits_clean_cal is None:
                    features_clean_cal, logits_clean_cal, labels_clean_cal = get_features_logits_labels(net, 'cal', ds_info)
                if 'TemperatureScaling' == method:
                    calibrator_temp = TemperatureScaling()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'TemperatureScalingMSE' == method:
                    calibrator_temp = TemperatureScaling(loss='mse')
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'VectorScaling' == method:
                    calibrator_temp = VectorScaling()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'MatrixScaling' == method:
                    calibrator_temp = MatrixScaling()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'MatrixScalingODIR' == method:
                    calibrator_temp = MatrixScalingODIR(logits_clean_cal.shape[1])
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'DirichletL2' == method:
                    calibrator_temp = DirichletL2(logits_clean_cal.shape[1])
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'DirichletODIR' == method:
                    calibrator_temp = DirichletODIR(logits_clean_cal.shape[1])
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'EnsembleTemperatureScaling' == method:
                    calibrator_temp = EnsembleTemperatureScaling()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'EnsembleTemperatureScalingCE' == method:
                    calibrator_temp = EnsembleTemperatureScaling(loss='ce')
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'CCAC' == method:
                    calibrator_temp = CCAC(method='CCAC')
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'CCAC-S' == method:
                    calibrator_temp = CCAC(method='CCAC-S')
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'IRM' == method:
                    calibrator_temp = IRM()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'IRMTS' == method:
                    calibrator_temp = IRMTS()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'IROvA' == method:
                    calibrator_temp = IROvA()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif 'IROvATS' == method:
                    calibrator_temp = IROvATS()
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                elif ('-P' in method) and ('-PMAX' not in method):
                    iid_method, _ = method.split('-')
                    calibrator_temp = Perturbed(method=iid_method)
                    calibrator_temp.fit(net, ds_info)
                elif '-AP' in method:
                    iid_method, _, metric = method.split('-')
                    calibrator_temp = AdaptivePerturbed(method=iid_method, metric=metric)
                    calibrator_temp.fit(net, ds_info)
                    for aux in ['ATCPMAX', 'ATCNE', 'ATCNGY']:
                        calibrator_temp.metric = aux
                        np.save(os.path.join(experiments_folder, 'calibration_methods', f'{iid_method}-AP-{aux}.npy'), calibrator_temp)
                    calibrator_temp.metric = metric    

                elif '-FV1' in method:
                    iid_method, _, metric = method.split('-')
                    calibrator_temp = Forged_v1(method=iid_method, metric=metric)
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                    for aux in ['PMAX', 'NE', 'NGY', 'ATCPMAX', 'ATCNE', 'ATCNGY']:
                        calibrator_temp.metric = aux
                        np.save(os.path.join(experiments_folder, 'calibration_methods', f'{iid_method}-FV1-{aux}.npy'), calibrator_temp)
                    calibrator_temp.metric = metric    
                elif '-FV2' in method:
                    iid_method, _, metric = method.split('-')
                    calibrator_temp = Forged_v1(method=iid_method, metric=metric, nsets=25)
                    calibrator_temp.fit(logits_clean_cal, labels_clean_cal)
                    for aux in ['PMAX', 'NE', 'NGY', 'ATCPMAX', 'ATCNE', 'ATCNGY']:
                        calibrator_temp.metric = aux
                        np.save(os.path.join(experiments_folder, 'calibration_methods', f'{iid_method}-FV2-{aux}.npy'), calibrator_temp)
                    calibrator_temp.metric = metric    
                calibrators[method] = calibrator_temp
                np.save(file, calibrator_temp)
    return calibrators

def get_results(method, features_test, logits_test, labels_test, calibrator, net, ds_info):
    if method == 'Vanilla':
        results = get_uncertainty_measures(torch.nn.Softmax(dim=1)(logits_test), labels_test)
    else:
        results = get_uncertainty_measures(calibrator.predict_proba(logits_test), labels_test)
    return results