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

from utils import get_loader, create_folder
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
    

def get_pmax(logits):
    return torch.nn.Softmax(dim=1)(logits).max(dim=1).values

def get_entropy(logits):
    softmaxes = torch.nn.Softmax(dim=1)(logits)
    entropy = -softmaxes * torch.log(softmaxes + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return -entropy

def get_energy(logits, T=1.0):
    return T*torch.logsumexp(logits/T, dim=1)
    
    
class AdaptivePerturbed():    
    def __init__(self, method='TemperatureScaling', metric='ATCPMAX'):
        self.method = method
        self.metric = metric
    
    def fit(self, net, ds_info):
        base_file_path = os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init')
        if os.path.exists(os.path.join(base_file_path, 'labels.npy')):
            logits_all = np.load(os.path.join(base_file_path, 'logits.npy'), allow_pickle=True).item()
            labels_all = np.load(os.path.join(base_file_path, 'labels.npy'), allow_pickle=True).item()
            epsilons = np.load(os.path.join(base_file_path, 'epsilons.npy'))
            acc_all = np.load(os.path.join(base_file_path, 'accs.npy'), allow_pickle=True).item()
        else:
            epsilons = estimate_epsilons('cal', net, ds_info)
            # Gather the logits and labels for the different epsilons
            logits_all = {}
            labels_all = {}
            acc_all = {}
            for epsilon in epsilons:
                logits, labels = get_logits_labels_epsilon(epsilon, 'cal', net, ds_info)
                logits_all[epsilon] = logits
                labels_all[epsilon] = labels
                acc_all[epsilon] = (torch.sum(logits.argmax(dim=1) == labels)/len(labels)).item()
            create_folder(os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init'))
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init', 'logits.npy'), logits_all)
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init', 'labels.npy'), labels_all)
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init', 'epsilons.npy'), epsilons)
            np.save(os.path.join(ds_info['folder'], 'calibration_methods', 'adaptive_perturbed_init', 'accs.npy'), acc_all)

            
        self.acc_cal = (logits_all[0.0].argmax(dim=1) == labels_all[0.0]).float().mean().item()
        self.pmax_cal = get_pmax(logits_all[0.0])
        self.entropy_cal = get_entropy(logits_all[0.0])
        self.energy_cal = get_energy(logits_all[0.0])
            
        self.epsilons = epsilons
        self.accs = np.array([acc_all[key] for key in acc_all])
        calibrator = {}
        for epsilon in epsilons:
            if 'TemperatureScaling' == self.method:
                calibrator[epsilon] = TemperatureScaling()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'TemperatureScalingMSE' == self.method:
                calibrator[epsilon] = TemperatureScaling(loss='mse')
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'VectorScaling' == self.method:
                calibrator[epsilon] = VectorScaling()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'MatrixScaling' == self.method:
                calibrator[epsilon] = MatrixScaling()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'MatrixScalingODIR' == self.method:
                calibrator[epsilon] = MatrixScalingODIR(logits_all[epsilon].shape[1])
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'DirichletL2' == self.method:
                calibrator[epsilon] = DirichletL2(logits_all[epsilon].shape[1])
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'DirichletODIR' == self.method:
                calibrator[epsilon] = DirichletODIR(logits_all[epsilon].shape[1])
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'EnsembleTemperatureScaling' == self.method:
                calibrator[epsilon] = EnsembleTemperatureScaling()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'EnsembleTemperatureScalingCE' == self.method:
                calibrator[epsilon] = EnsembleTemperatureScaling(loss='ce')
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'IRM' == self.method:
                calibrator[epsilon] = IRM()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'IRMTS' == self.method:
                calibrator[epsilon] = IRMTS()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'IROvA' == self.method:
                calibrator[epsilon] = IROvA()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
            elif 'IROvATS' == self.method:
                calibrator[epsilon] = IROvATS()
                calibrator[epsilon].fit(logits_all[epsilon], labels_all[epsilon])
        self.calibrator = calibrator


    def predict_proba(self, logits):
        if self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
        return self.calibrator[self.epsilons[np.abs(acc_test-self.accs).argmin()]].predict_proba(logits)


    def predict(self, logits):
        if self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
        return self.calibrator[self.epsilons[np.abs(acc_test-self.accs).argmin()]].predict(logits)