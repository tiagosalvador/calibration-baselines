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
        for architecture in tqdm(architectures[dataset], desc='Architectures', leave=False):
            create_folder(os.path.join(experiments_folder, architecture))
            net = load_net(dataset, architecture, device)
            ds_info = load_ds_info(dataset, net)
            for splitID in tqdm(splitIDs, desc='Splits', leave=False):
                path_temp = os.path.join(root_folder_data, experiment_name, dataset)
                indices = np.load(os.path.join(path_temp, f'val_test_{splitID}.npy'), allow_pickle=True).item()
                ds_info['indices_cal'] =  indices['val']
                ds_info['indices_test'] =  indices['test']
                base_folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
                create_folder(os.path.join(base_folder_path_temp, 'calibration_methods'))
                ds_info['folder'] = base_folder_path_temp
                create_folder(os.path.join(root_folder_data, dataset, architecture, dataset))
                ds_info['folder_outputs'] = os.path.join(root_folder_data, dataset, architecture, dataset)
                calibrators = get_calibrators(methods, net, ds_info)
                features_test = None
                logits_test = None
                labels_test = None
                create_folder(os.path.join(base_folder_path_temp,'results', dataset))
                for method in tqdm(methods, desc="Method", leave=False):
                    saveto = os.path.join(base_folder_path_temp,'results', dataset, f'{method}.npy')           
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


def evaluate_ood(iid_dataset, ood_datasets, architectures, methods, splitIDs):
    experiments_folder = os.path.join(experiments_folder_base, iid_dataset)
    for architecture in tqdm(architectures, desc='Architectures'):
        create_folder(os.path.join(experiments_folder, architecture))
        net = load_net(iid_dataset, architecture, device)
        ds_info = load_ds_info(iid_dataset, net)
        for splitID in tqdm(splitIDs, desc='Splits', leave=False):
            base_folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
            create_folder(os.path.join(base_folder_path_temp, 'calibration_methods'))
            ds_info['folder'] = base_folder_path_temp
            calibrators = get_calibrators(methods, net, ds_info)
            for ood_dataset in tqdm(ood_datasets, desc='OOD Datasets', leave=False):
                ds_info = load_ds_info(ood_dataset, net)
                folder_outputs = os.path.join(root_folder_data, iid_dataset, architecture, ood_dataset)
                create_folder(folder_outputs)
                ds_info['folder_outputs'] = folder_outputs
                features_test = None
                logits_test = None
                labels_test = None
                create_folder(os.path.join(base_folder_path_temp, 'results', ood_dataset))
                for method in tqdm(methods, desc="Method", leave=False):
                    saveto = os.path.join(base_folder_path_temp, 'results', ood_dataset, f'{method}.npy')
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
                        
def evaluate_corrupted(iid_dataset, corruptions, intensities, architectures, methods, splitIDs):
    dataset_corrupted = iid_dataset+'-c'
    experiments_folder = os.path.join(experiments_folder_base, iid_dataset)
    for architecture in tqdm(architectures, desc='Architectures'):
        create_folder(os.path.join(experiments_folder, architecture))
        net = load_net(iid_dataset, architecture, device)
        ds_info = load_ds_info(iid_dataset, net)
        for splitID in tqdm(splitIDs, desc='Splits', leave=False):
            ds_info['folder'] = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
            ds_info['folder_outputs'] = os.path.join(root_folder_data, iid_dataset, architecture, iid_dataset)
            calibrators = get_calibrators(methods, net, ds_info)
            ds_info = load_ds_info(dataset_corrupted, net)
            indices_path = os.path.join(root_folder_data, experiment_name, iid_dataset, f'val_test_{splitID}.npy')
            indices = np.load(indices_path, allow_pickle=True).item()
            ds_info['indices_cal'] =  indices['val']
            ds_info['indices_test'] =  indices['test']
            for corruption in tqdm(corruptions, desc='Corruptions', leave=False):
                ds_info['corruption'] = corruption
                for intensity in tqdm(intensities, desc='Intensities', leave=False):
                    ds_info['intensity'] = intensity                    
                    folder_outputs = os.path.join(root_folder_data, iid_dataset, architecture,
                                                  dataset_corrupted, corruption, f'intensity_{intensity}')
                    create_folder(folder_outputs)
                    ds_info['folder_outputs'] = folder_outputs
                    features_test = None
                    logits_test = None
                    labels_test = None
                    folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}', 'results')
                    create_folder(os.path.join(folder_path_temp, dataset_corrupted, corruption, f'intensity_{intensity}'))
                    for method in tqdm(methods, desc="Method", leave=False):
                        saveto = os.path.join(folder_path_temp, dataset_corrupted, corruption, f'intensity_{intensity}', f'{method}.npy')
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
                            
                            
                            
                            
def missing_outputs(datasets, architectures, methods, splitIDs):
    for dataset in tqdm(datasets, desc='Datasets'):
        experiments_folder = os.path.join(experiments_folder_base, dataset)
        for architecture in tqdm(architectures[dataset], desc='Architectures', leave=False):
            create_folder(os.path.join(experiments_folder, architecture))
            net = load_net(dataset, architecture, device)
            ds_info = load_ds_info(dataset, net)
            for splitID in tqdm(splitIDs, desc='Splits', leave=False):
                path_temp = os.path.join(root_folder_data, experiment_name, dataset)
                indices = np.load(os.path.join(path_temp, f'val_test_{splitID}.npy'), allow_pickle=True).item()
                ds_info['indices_cal'] =  indices['val']
                ds_info['indices_test'] =  indices['test']
                base_folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
                create_folder(os.path.join(base_folder_path_temp, 'calibration_methods'))
                ds_info['folder'] = base_folder_path_temp
                create_folder(os.path.join(root_folder_data, dataset, architecture, dataset))
                ds_info['folder_outputs'] = os.path.join(root_folder_data, dataset, architecture, dataset)
                suffix_name = 'test.npy'
                if not(os.path.exists(os.path.join(ds_info['folder_outputs'], f'labels_{suffix_name}'))):
                    tqdm.write(ds_info['folder_outputs'])
#                         features_test, logits_test, labels_test = get_features_logits_labels(net, 'test', ds_info)



def missing_corrupted(iid_dataset, corruptions, intensities, architectures, methods, splitIDs):
    dataset_corrupted = iid_dataset+'-c'
    experiments_folder = os.path.join(experiments_folder_base, iid_dataset)
    for architecture in tqdm(architectures, desc='Architectures'):
        create_folder(os.path.join(experiments_folder, architecture))
        net = load_net(iid_dataset, architecture, device)
        ds_info = load_ds_info(iid_dataset, net)
        for splitID in tqdm(splitIDs, desc='Splits', leave=False):
            ds_info['folder'] = os.path.join(experiments_folder, architecture, f'splitID_{splitID}')
            ds_info['folder_outputs'] = os.path.join(root_folder_data, iid_dataset, architecture, iid_dataset)
            ds_info = load_ds_info(dataset_corrupted, net)
            indices_path = os.path.join(root_folder_data, experiment_name, iid_dataset, f'val_test_{splitID}.npy')
            indices = np.load(indices_path, allow_pickle=True).item()
            ds_info['indices_cal'] =  indices['val']
            ds_info['indices_test'] =  indices['test']
            for corruption in tqdm(corruptions, desc='Corruptions', leave=False):
                ds_info['corruption'] = corruption
                for intensity in tqdm(intensities, desc='Intensities', leave=False):
                    ds_info['intensity'] = intensity                    
                    folder_outputs = os.path.join(root_folder_data, iid_dataset, architecture,
                                                  dataset_corrupted, corruption, f'intensity_{intensity}')
                    create_folder(folder_outputs)
                    ds_info['folder_outputs'] = folder_outputs
                    features_test = None
                    logits_test = None
                    labels_test = None
                    folder_path_temp = os.path.join(experiments_folder, architecture, f'splitID_{splitID}', 'results')
                    create_folder(os.path.join(folder_path_temp, dataset_corrupted, corruption, f'intensity_{intensity}'))
                    suffix_name = 'test.npy'
                    if not(os.path.exists(os.path.join(ds_info['folder_outputs'], f'labels_{suffix_name}'))):
                        tqdm.write(ds_info['folder_outputs'])
                        features_test, logits_test, labels_test = get_features_logits_labels(net, 'test', ds_info)
