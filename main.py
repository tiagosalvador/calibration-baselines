root_folder_experiments = 'experiments_temp'
root_folder_data = 'data'

from tqdm import tqdm

import torch

import numpy as np
import os
import pandas as pd

from utils import create_folder, small_large_split, get_features_logits_labels
from calibration_methods_wrapper import get_calibrators, get_results
from networks import load_net
from datasets import load_ds_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experiment_name = 'standard'

create_folder(root_folder_experiments)
experiments_folder_base = os.path.join(root_folder_experiments, experiment_name)
create_folder(experiments_folder_base)

def calibrate(datasets, architectures, methods, splitIDs):
    for dataset in tqdm(datasets, desc='Datasets'):
        experiments_folder = os.path.join(experiments_folder_base, dataset)
        create_folder(experiments_folder)
        for architecture in tqdm(architectures, desc='Architectures', leave=False):
            create_folder(os.path.join(experiments_folder, architecture))
            net = load_net(dataset, architecture, device)
            ds_info = load_ds_info(dataset, net)
            for splitID in tqdm(splitIDs, desc='Splits', leave=False):
                ds_info['indices_cal'] =  np.load(os.path.join(root_folder_data, experiment_name, dataset, f'val_test_{splitID}.npy'), allow_pickle=True).item()['val']
                ds_info['indices_test'] =  np.load(os.path.join(root_folder_data, experiment_name, dataset, f'val_test_{splitID}.npy'), allow_pickle=True).item()['test']
                base_folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}', dataset)
                create_folder(base_folder_path_temp)
                create_folder(os.path.join(base_folder_path_temp,'features_logits_labels'))
                create_folder(os.path.join(base_folder_path_temp, 'calibration_methods'))
                ds_info['folder'] = base_folder_path_temp
                calibrators = get_calibrators(methods, net, ds_info)
                features_test = None
                logits_test = None
                labels_test = None
                create_folder(os.path.join(base_folder_path_temp,'results'))
                for method in tqdm(methods, desc="Method", leave=False):
                    saveto = os.path.join(base_folder_path_temp,'results', f'{method}.npy')           
                    if os.path.exists(saveto):
                        pass
                    else:
                        if features_test is None:
                            features_test, logits_test, labels_test = get_features_logits_labels(net, 'test', ds_info)
                        results = get_results(method, 
                                              features_test, 
                                              logits_test, 
                                              labels_test, 
                                              calibrators[method], 
                                              net, 
                                              ds_info)
                        np.save(saveto, results)


def evaluate_ood(iid_dataset, datasets, architectures, methods, splitIDs):
    experiments_folder = os.path.join(experiments_folder_base, iid_dataset)
    create_folder(experiments_folder)
    for architecture in tqdm(architectures, desc='Architectures'):
        create_folder(os.path.join(experiments_folder, architecture))
        net = load_net(iid_dataset, architecture, device)
        ds_info = load_ds_info(iid_dataset, net)
        for splitID in tqdm(splitIDs, desc='Splits', leave=False):
            ds_info['folder'] = os.path.join(experiments_folder, architecture, f'splitID_{splitID}', iid_dataset)
            calibrators = get_calibrators(methods, net, ds_info)
            folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
            create_folder(folder_path_temp)
            for dataset in datasets:
                ds_info = load_ds_info(dataset, net)
                create_folder(os.path.join(folder_path_temp, dataset))
                create_folder(os.path.join(folder_path_temp, dataset, 'features_logits_labels'))
                ds_info['folder'] = os.path.join(os.path.join(folder_path_temp, dataset))
                features_test = None
                logits_test = None
                labels_test = None
                create_folder(os.path.join(folder_path_temp, dataset, 'results'))
                for method in tqdm(methods, desc="Method", leave=False):
                    saveto = os.path.join(folder_path_temp, dataset, 'results', f'{method}.npy')
                    if os.path.exists(saveto):
                        pass
                    else:
                        if features_test is None:
                            features_test, logits_test, labels_test = get_features_logits_labels(net, 'ood', ds_info)
                        results = get_results(method, 
                                              features_test, 
                                              logits_test, 
                                              labels_test, 
                                              calibrators[method], 
                                              net, 
                                              ds_info)
                        np.save(saveto, results)