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

def get_calibrators(methods, net, ds_info):
    experiments_folder = ds_info['folder']
    logits_clean_cal = None
    calibrators = {}
    for method in tqdm(methods, desc="Prepping Methods", leave=False):
        file = os.path.join(experiments_folder,'calibration_methods', method+'.npy')
        if os.path.exists(file):
            calibrators[method] = np.load(file, allow_pickle=True).item()
        else:
            if logits_clean_cal is None:
                features_clean_cal, logits_clean_cal, labels_clean_cal = get_features_logits_labels(net, 'cal', ds_info)
            if method == 'Vanilla':
                calibrators[method] = None
            else:
                if 'TemperatureScaling' == method:
                    calibrator_temp = TemperatureScaling()
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
                calibrators[method] = calibrator_temp
                np.save(file, calibrator_temp)
    return calibrators

def get_results(method, features_test, logits_test, labels_test, calibrator, net, ds_info):
    if method == 'Vanilla':
        results = get_uncertainty_measures(logits_test, labels_test)
    else:
        results = get_uncertainty_measures(calibrator.predict(logits_test), labels_test)
    return results