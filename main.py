root_folder_experiments = 'experiments'
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
        for architecture in tqdm(architectures, desc='Architectures'):
            create_folder(os.path.join(experiments_folder, architecture))
            net = load_net(dataset, architecture, device)
            ds_info = load_ds_info(dataset, net)
            for splitID in tqdm(splitIDs, desc='Splits', leave=False):
                ds_info['indices_cal'] =  np.load(os.path.join(root_folder_data, experiment_name, dataset, f'val_test_{splitID}.npy'), allow_pickle=True).item()['val']
                ds_info['indices_test'] =  np.load(os.path.join(root_folder_data, experiment_name, dataset, f'val_test_{splitID}.npy'), allow_pickle=True).item()['test']

                create_folder(os.path.join(experiments_folder, architecture, f'splitID_{splitID}'))
                create_folder(os.path.join(experiments_folder, architecture, f'splitID_{splitID}','features_logits_labels'))
                create_folder(os.path.join(experiments_folder, architecture, f'splitID_{splitID}','calibration_methods'))
                ds_info['folder'] = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')

                calibrators = get_calibrators(methods, net, ds_info)
                features_test = None
                logits_test = None
                labels_test = None
                create_folder(os.path.join(experiments_folder, architecture, f'splitID_{splitID}','results'))
                for method in tqdm(methods, desc="Method", leave=False):
                    saveto = os.path.join(experiments_folder, architecture, f'splitID_{splitID}','results', f'{method}.npy')           
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