import torch
import numpy as np

from calibration_methods.temperature_scaling import TemperatureScaling
from calibration_methods.vector_scaling import VectorScaling
from calibration_methods.matrix_scaling import MatrixScaling, MatrixScalingODIR
from calibration_methods.dirichlet_calibration import DirichletL2, DirichletODIR
from calibration_methods.ensemble_temperature_scaling import EnsembleTemperatureScaling
from calibration_methods.irova import IROvA
from calibration_methods.irova_ts import IROvATS
from calibration_methods.irm import IRM
from calibration_methods.irm_ts import IRMTS

def get_pmax(logits):
    return torch.nn.Softmax(dim=1)(logits).max(dim=1).values

def get_entropy(logits):
    softmaxes = torch.nn.Softmax(dim=1)(logits)
    entropy = -softmaxes * torch.log(softmaxes + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return -entropy

def get_energy(logits, T=1.0):
    return T*torch.logsumexp(logits/T, dim=1)

class Forged_v1():
    def __init__(self, method, metric):
        self.method = method
        self.metric = metric

    # Find the temperature
    def fit(self, logits, labels, verbose=False):
        
        pmax = get_pmax(logits)
        order = np.argsort(pmax)
        x = torch.cumsum((logits[order].argmax(dim=1)==labels[order]).float(),dim=0)/(torch.arange(len(logits))+1)
        x[:100] = 1.0
        max_acc = torch.max(x[100:])
        min_acc = torch.min(x)
        accs = np.linspace(min_acc, max_acc, 5)
        ns = []
        for acc in accs:
            ns.append(np.argmin(np.abs(x-acc)).item())
        
        calibrators_forged = []
        pmaxs_forged = []
        entropies_forged = []
        energies_forged = []
        accs_forged = []
        for n in ns:
            top1 = (logits.argmax(dim=1) == labels)
            select = order<n
            logits_forged = logits[select]
            labels_forged = labels[select]
            pmaxs_forged.append(get_pmax(logits_forged).mean().item())
            entropies_forged.append(get_entropy(logits_forged).mean().item())
            energies_forged.append(get_energy(logits_forged).mean().item())
            if 'TemperatureScaling' == self.method:
                calibrator = TemperatureScaling()
            elif 'TemperatureScalingMSE' == self.method:
                calibrator = TemperatureScaling(loss='mse')
            elif 'VectorScaling' == self.method:
                calibrator = VectorScaling()
            elif 'MatrixScaling' == self.method:
                calibrator = MatrixScaling()
            elif 'MatrixScalingODIR' == self.method:
                calibrator = MatrixScalingODIR(logits.shape[1])
            elif 'DirichletL2' == self.method:
                calibrator = DirichletL2(logits.shape[1])
            elif 'DirichletODIR' == self.method:
                calibrator = DirichletODIR(logits.shape[1])
            elif 'EnsembleTemperatureScaling' == self.method:
                calibrator = EnsembleTemperatureScaling()
            elif 'EnsembleTemperatureScalingCE' == self.method:
                calibrator = EnsembleTemperatureScaling(loss='ce')
            elif 'IRM' == self.method:
                calibrator = IRM()
            elif 'IRMTS' == self.method:
                calibrator = IRMTS()
            elif 'IROvA' == self.method:
                calibrator = IROvA()
            elif 'IROvATS' == self.method:
                calibrator = IROvATS()
            calibrator.fit(logits_forged, labels_forged)
            calibrators_forged.append(calibrator)
            accs_forged.append((logits_forged.argmax(dim=1) == labels_forged).float().mean().item())
        self.calibrators_forged = calibrators_forged
        self.accs_forged = np.array(accs_forged)
        self.pmaxs_forged = np.array(pmaxs_forged)
        self.entropies_forged = np.array(entropies_forged)
        self.energies_forged = np.array(energies_forged)
        
        self.acc_cal = (logits.argmax(dim=1) == labels).float().mean().item()
        self.pmax_cal = get_pmax(logits)
        self.entropy_cal = get_entropy(logits)
        self.energy_cal = get_energy(logits)
        
    def predict_proba(self, logits):
        if self.metric == 'PMAX':
            pmax_mean = get_pmax(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.pmaxs_forged-pmax_mean))]
        elif self.metric == 'NE':
            entropy_mean = get_entropy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.entropies_forged-entropy_mean))]
        elif self.metric == 'NGY':
            energy_mean = get_energy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.energies_forged-energy_mean))]
        elif self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        return calibrator.predict_proba(logits)

    def predict(self, logits):
        if self.metric == 'PMAX':
            pmax_mean = get_pmax(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.pmaxs_forged-pmax_mean))]
        elif self.metric == 'NE':
            entropy_mean = get_entropy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.entropies_forged-entropy_mean))]
        elif self.metric == 'NGY':
            energy_mean = get_energy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.energies_forged-energy_mean))]
        elif self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        return calibrator.predict(logits)

    
class Forged_v1():
    def __init__(self, method, metric, nsets=5):
        self.method = method
        self.metric = metric
        self.nsets = nsets

    # Find the temperature
    def fit(self, logits, labels, verbose=False):
        
        pmax = get_pmax(logits)
        order = np.argsort(pmax)
        x = torch.cumsum((logits[order].argmax(dim=1)==labels[order]).float(),dim=0)/(torch.arange(len(logits))+1)
        x[:100] = 1.0
        max_acc = torch.max(x[100:])
        min_acc = torch.min(x)
        accs = np.linspace(min_acc, max_acc, self.nsets)
        ns = []
        for acc in accs:
            ns.append(np.argmin(np.abs(x-acc)).item())
        
        calibrators_forged = []
        pmaxs_forged = []
        entropies_forged = []
        energies_forged = []
        accs_forged = []
        for n in ns:
            top1 = (logits.argmax(dim=1) == labels)
            select = order<n
            logits_forged = logits[select]
            labels_forged = labels[select]
            pmaxs_forged.append(get_pmax(logits_forged).mean().item())
            entropies_forged.append(get_entropy(logits_forged).mean().item())
            energies_forged.append(get_energy(logits_forged).mean().item())
            if 'TemperatureScaling' == self.method:
                calibrator = TemperatureScaling()
            elif 'TemperatureScalingMSE' == self.method:
                calibrator = TemperatureScaling(loss='mse')
            elif 'VectorScaling' == self.method:
                calibrator = VectorScaling()
            elif 'MatrixScaling' == self.method:
                calibrator = MatrixScaling()
            elif 'MatrixScalingODIR' == self.method:
                calibrator = MatrixScalingODIR(logits.shape[1])
            elif 'DirichletL2' == self.method:
                calibrator = DirichletL2(logits.shape[1])
            elif 'DirichletODIR' == self.method:
                calibrator = DirichletODIR(logits.shape[1])
            elif 'EnsembleTemperatureScaling' == self.method:
                calibrator = EnsembleTemperatureScaling()
            elif 'EnsembleTemperatureScalingCE' == self.method:
                calibrator = EnsembleTemperatureScaling(loss='ce')
            elif 'IRM' == self.method:
                calibrator = IRM()
            elif 'IRMTS' == self.method:
                calibrator = IRMTS()
            elif 'IROvA' == self.method:
                calibrator = IROvA()
            elif 'IROvATS' == self.method:
                calibrator = IROvATS()
            calibrator.fit(logits_forged, labels_forged)
            calibrators_forged.append(calibrator)
            accs_forged.append((logits_forged.argmax(dim=1) == labels_forged).float().mean().item())
        self.calibrators_forged = calibrators_forged
        self.accs_forged = np.array(accs_forged)
        self.pmaxs_forged = np.array(pmaxs_forged)
        self.entropies_forged = np.array(entropies_forged)
        self.energies_forged = np.array(energies_forged)
        
        self.acc_cal = (logits.argmax(dim=1) == labels).float().mean().item()
        self.pmax_cal = get_pmax(logits)
        self.entropy_cal = get_entropy(logits)
        self.energy_cal = get_energy(logits)
        
    def predict_proba(self, logits):
        if self.metric == 'PMAX':
            pmax_mean = get_pmax(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.pmaxs_forged-pmax_mean))]
        elif self.metric == 'NE':
            entropy_mean = get_entropy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.entropies_forged-entropy_mean))]
        elif self.metric == 'NGY':
            energy_mean = get_energy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.energies_forged-energy_mean))]
        elif self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        return calibrator.predict_proba(logits)

    def predict(self, logits):
        if self.metric == 'PMAX':
            pmax_mean = get_pmax(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.pmaxs_forged-pmax_mean))]
        elif self.metric == 'NE':
            entropy_mean = get_entropy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.entropies_forged-entropy_mean))]
        elif self.metric == 'NGY':
            energy_mean = get_energy(logits).mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.energies_forged-energy_mean))]
        elif self.metric == 'ATCPMAX':
            pmax_test = get_pmax(logits)
            acc_test = (pmax_test>np.quantile(self.pmax_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNE':
            entropy_test = get_entropy(logits)
            acc_test = (entropy_test>np.quantile(self.entropy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        elif self.metric == 'ATCNGY':
            energy_test = get_energy(logits)
            acc_test = (energy_test>np.quantile(self.energy_cal,1-self.acc_cal)).float().mean().item()
            calibrator = self.calibrators_forged[np.argmin(np.abs(self.accs_forged-acc_test))]
        return calibrator.predict(logits)
    