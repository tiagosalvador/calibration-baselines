from tqdm import tqdm
import torch
import numpy as np
import os

from calibration_methods.temperature_scaling import TemperatureScaling
from calibration_methods.vector_scaling import VectorScaling
from calibration_methods.matrix_scaling import MatrixScaling, MatrixScalingODIR
from calibration_methods.dirichlet_calibration import DirichletL2, DirichletODIR
from calibration_methods.ensemble_temperature_scaling import EnsembleTemperatureScaling
from calibration_methods.irova import IROvA
from calibration_methods.irova_ts import IROvATS
from calibration_methods.irm import IRM
from calibration_methods.irm_ts import IRMTS

from utils.utils import get_loader, create_folder
import torchvision.transforms as transforms


from calibration_methods_ood.utils.optimizer_nelder_mead import minimize_neldermead
import time

class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, img):
        return torch.clip(img + torch.normal(mean=0.0, std=self.sigma, size=img.shape), 0, 1)


def get_logits_labels_epsilon(epsilon, subset, net, ds_info):
    if (ds_info['name'] == 'cifar10') or (ds_info['name'] == 'cifar100') or (ds_info['name'] == 'svhn'):
        if epsilon > 0:
            ds_info['transform'] = transforms.Compose([
                 transforms.ToTensor(), GaussianNoise(epsilon),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            ds_info['transform'] = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
    elif ds_info['name'] == 'imagenet':
        if epsilon>0:
            ds_info['transform'] = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), GaussianNoise(epsilon),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            ds_info['transform'] = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataloader = get_loader(subset, ds_info)
    net.eval()
    logits = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            _, logits_temp = net(images.cuda())
            logits.append(logits_temp.cpu())
    logits = torch.cat(logits, dim=0)
    labels = torch.from_numpy(dataloader.dataset.targets).long()
    if (subset != 'train') and (subset != 'ood'):
        indices = ds_info['indices_'+subset]
        return logits[indices], labels[indices]
    return logits, labels
    
    
def get_acc(epsilons, subset, net, ds_info):
    accs = []
    for epsilon in epsilons:
        logits, labels = get_logits_labels_epsilon(epsilon, subset, net, ds_info)
        acc = (torch.sum(logits.argmax(dim=1) == labels)/len(labels)).item()
        accs.append(acc)
#     print(epsilons, np.round(accs,2))
    return np.array(accs)


def optmize_accuracy(epsilons, *args):
    """
    Function is used in nelder mead optimizer.
    Args:
        epsilons: float, level of gaussian perturbation
        (*args) target_accuracy: float
        (*args) modelf: object from class ModelFactory
        (*args) data: tf.data.Dataset object (x_data, y_labels), prepared with
            batch_size, shuffle etc.
        (*args) dataset_name: string
    Return:
        float, difference between target accuracy and calculated accuracy
    """
    target_accuracy, subset, net, ds_info = args
#     print(epsilons)
    return abs(target_accuracy - get_acc(epsilons, subset, net, ds_info))


def estimate_epsilons(
        subset,
        net,
        ds_info,
        number_perturbation_levels=6,
        accuracy_deviation_acceptable=0.03,
        accuracy_deviation_acceptable_last_step=0.05,
        gauss_eps_start=0.05,
        opt_delta_gauss_eps=0.5):
    """
    Optimizes levels of gaussian perturbation (=epsilons) based on a list of target accuracies.
    The target accuracies are calculated with interpolation between min & max.
    The data is perturbed with gaussian noise at a certain level (=epsilon).
    The model's accuracy for the perturbed data is calculated.
    The levels of perturbation are iteratively optimized with nelder mead.
    Args:
        number_perturbation_levels: int, number of intended levels of perturbation
        accuracy_deviation_acceptable: float, rate of acceptable deviation between
            target accuracy and calcualted accuracy based on level of perturbation
        accuracy_deviation_acceptable_last_step: float, rate of acceptable deviation
            between target accuracy and calcualted accuracy based on level of
            perturbation for the lowest accuracy
        gauss_eps_start: float, starting value for level of perturbation in optimization
        opt_delta_gauss_eps: float: nonzdelt parameter in optimization
    Return:
        optimized_epsilons: list [float], optimized levels of perturbation (=epsilons)
            according to the list of target accuracies.
    """

    start_time = time.time()
    # calculate target accuracies
    max_acc = get_acc([0.0], subset, net, ds_info)
    min_acc = 1 / ds_info['num_classes']
    perturbation_levels = range(number_perturbation_levels)
    target_accuracy_list = []
    for perturbation_level in perturbation_levels:
        target_accuracy_list.append(max_acc - (max_acc - min_acc) \
            * perturbation_level / (len(perturbation_levels) - 1))
    # estimate gauss epsilons with regard to target accuracies
    tqdm.write(f"Start estimating gaussian epsilons with regard to target accuracies: {np.round(target_accuracy_list,2)}" )
    epsilons = []
    for i, target_accuracy in enumerate(target_accuracy_list):
        tqdm.write("Step_%s - Target Accuracy: %s" % (str(i), str(target_accuracy)))
        if i == 0:
            epsilons.append(0.0)
        else:
            if i == 1:
                x0 = gauss_eps_start
            else:
                x0 = epsilons[i - 1]

            if i == len(target_accuracy_list) - 1:
                tolerance = max_acc * accuracy_deviation_acceptable_last_step
            else:
                tolerance = max_acc * accuracy_deviation_acceptable
            x = minimize_neldermead(
                optmize_accuracy,
                x0,
                args=(target_accuracy, subset, net, ds_info),
                tol_abs=tolerance,
                nonzdelt=opt_delta_gauss_eps,
                callback=None,
                maxiter=None, maxfev=None, disp=False,
                return_all=False, initial_simplex=None,
                xatol=1e-4, fatol=1e-4, adaptive=False,
            )
            epsilons.append(x[0])
    time_needed = time.time() - start_time
    tqdm.write("Finished estimating gaussian epsilons!!")
    tqdm.write(f"Time needed: {time_needed}")
    return epsilons
    

class Perturbed():    
    def __init__(self, method='TemperatureScaling'):
        self.method = method
    
    def fit(self, net, ds_info):
        base_file_path = os.path.join(ds_info['folder'], 'calibration_methods', 'perturbed_init')
        if os.path.exists(os.path.join(base_file_path, 'labels.npy')):
            logits_all = torch.from_numpy(np.load(os.path.join(base_file_path, 'logits.npy')))
            labels_all = torch.from_numpy(np.load(os.path.join(base_file_path, 'labels.npy'))).long()
            epsilons = np.load(os.path.join(base_file_path, 'epsilons.npy'))
        else:
            epsilons = estimate_epsilons('cal', net, ds_info)
            # Gather the logits and labels for the different epsilons
            logits_all = torch.zeros(0, ds_info['num_classes'])
            labels_all = torch.zeros(0).long()
            for epsilon in epsilons:
                logits, labels = get_logits_labels_epsilon(epsilon, 'cal', net, ds_info)
                logits_all = torch.vstack((logits_all, logits))
                labels_all = torch.hstack((labels_all, labels))
            create_folder(os.path.join(ds_info['folder'], 'calibration_methods', 'perturbed_init'))
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'perturbed_init', 'logits.npy'), logits_all)
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'perturbed_init', 'labels.npy'), labels_all)
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'perturbed_init', 'epsilons.npy'), epsilons)

        self.epsilons = epsilons
        
        if 'TemperatureScaling' == self.method:
            calibrator = TemperatureScaling()
            calibrator.fit(logits_all, labels_all)
        elif 'TemperatureScalingMSE' == self.method:
            calibrator = TemperatureScaling(loss='mse')
            calibrator.fit(logits_all, labels_all)
        elif 'VectorScaling' == self.method:
            calibrator = VectorScaling()
            calibrator.fit(logits_all, labels_all)
        elif 'MatrixScaling' == self.method:
            calibrator = MatrixScaling()
            calibrator.fit(logits_all, labels_all)
        elif 'MatrixScalingODIR' == self.method:
            calibrator = MatrixScalingODIR(logits_all.shape[1])
            calibrator.fit(logits_all, labels_all)
        elif 'DirichletL2' == self.method:
            calibrator = DirichletL2(logits_all.shape[1])
            calibrator.fit(logits_all, labels_all)
        elif 'DirichletODIR' == self.method:
            calibrator = DirichletODIR(logits_all.shape[1])
            calibrator.fit(logits_all, labels_all)
        elif 'EnsembleTemperatureScaling' == self.method:
            calibrator = EnsembleTemperatureScaling()
            calibrator.fit(logits_all, labels_all)
        elif 'EnsembleTemperatureScalingCE' == self.method:
            calibrator = EnsembleTemperatureScaling(loss='ce')
            calibrator.fit(logits_all, labels_all)
        elif 'IRM' == self.method:
            calibrator = IRM()
            calibrator.fit(logits_all, labels_all)
        elif 'IRMTS' == self.method:
            calibrator = IRMTS()
            calibrator.fit(logits_all, labels_all)
        elif 'IROvA' == self.method:
            calibrator = IROvA()
            calibrator.fit(logits_all, labels_all)
        elif 'IROvATS' == self.method:
            calibrator = IROvATS()
            calibrator.fit(logits_all, labels_all)
        self.calibrator = calibrator
        
        
    def predict_proba(self, logits):
        return self.calibrator.predict_proba(logits)


    def predict(self, logits):
        return self.calibrator.predict(logits)