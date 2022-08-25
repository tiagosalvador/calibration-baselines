import torch
from torch import nn
from torch.nn import functional as F
from scipy import optimize
import numpy as np

from tqdm import tqdm

def get_ks(confidences, ground_truth):
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.max(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_ks_l1(confidences, ground_truth):
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.mean(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_brier(confidences, ground_truth):
    # Compute Brier
    brier = np.zeros(confidences.shape)
    brier[ground_truth] = (1-confidences[ground_truth])**2
    brier[np.logical_not(ground_truth)] = (confidences[np.logical_not(ground_truth)])**2
    brier = np.mean(brier)
    return brier


def bin_confidences_and_accuracies(confidences, ground_truth, bin_edges, indices):
    i = np.arange(0, bin_edges.size-1)
    aux = indices == i.reshape((-1, 1))
    counts = aux.sum(axis=1)
    weights = counts / np.sum(counts)
    correct = np.logical_and(aux, ground_truth).sum(axis=1)
    a = np.repeat(confidences.reshape(1, -1), bin_edges.size-1, axis=0)
    a[np.logical_not(aux)] = 0
    bin_accuracy = correct / counts
    bin_confidence = a.sum(axis=1) / counts
    return weights, bin_accuracy, bin_confidence


def get_ece(confidences, ground_truth, nbins):
    # Repeated code from determine edges. Here it is okay if the bin edges are not unique defined
    confidences_sorted = confidences.copy()
    confidences_index = confidences.argsort()
    confidences_sorted = confidences_sorted[confidences_index]
    aux = np.linspace(0, len(confidences_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(confidences_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            confidences_sorted[bin_indices == i][confidences_sorted[bin_indices == i] == max(confidences_sorted[bin_indices == i])],
            confidences_sorted[bin_indices == (i + 1)][
                confidences_sorted[bin_indices == (i + 1)] == min(confidences_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = 0
    bin_edges[-1] = 1
    bin_indices = bin_indices[np.argsort(confidences_index)]

    weights, bin_accuracy, bin_confidence = bin_confidences_and_accuracies(confidences, ground_truth, bin_edges,
                                                                           bin_indices)
    ece = np.dot(weights, np.abs(bin_confidence - bin_accuracy))
    return ece


def compute_scores(confidences, ground_truth, nbins):

    # Compute ECE
    ece = get_ece(confidences, ground_truth, nbins)

    # Compute Brier
    brier = get_brier(confidences, ground_truth)

    # Compute KS
    ks = get_ks(confidences, ground_truth)

    # Compute KS-L1
    ks_l1 = get_ks_l1(confidences, ground_truth)

    return ece, ks, ks_l1, brier

def compute_classwise_scores(softmaxes, labels, nbins):
    total = 0
    ece = 0
    ks = 0
    ks_l1 = 0
    brier = 0
    for k in range(softmaxes.shape[1]):
        select = labels==k
        n_k = torch.sum(select).item()
        ece_temp, ks_temp, ks_l1_temp, brier_temp = compute_scores(softmaxes[:,k].numpy(), select.numpy(), nbins)
        ece += n_k*ece_temp
        ks += n_k*ks_temp
        ks_l1 += n_k*ks_l1_temp
        brier += n_k*brier_temp
        total += n_k
    ece /= total
    ks /= total
    ks_l1 /= total
    brier /= total
    return ece, ks, ks_l1, brier


def get_uncertainty_measures(softmaxes, labels):
    results = {}
    
    # Compute Accuracy
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    results['acc'] = accuracies.float().mean().item()

    # Top-1 ECE equal mass (with nbins=15), Brier Score, KS
    confidences = confidences.numpy()
    accuracies = accuracies.numpy()
    results['top1_ece_eq_mass'], results['top1_ks'], results['top1_ks_l1'], results['top1_brier'] = compute_scores(confidences, accuracies, 15)
    
    # Classwise ECE
    results['cw_ece_eq_mass'], results['cw_ks'], results['cw_ks_l1'], results['cw_brier'] = compute_classwise_scores(softmaxes, labels, 15)
    
    # NLL (Negative Log-Likelihood)
    nll_criterion = nn.NLLLoss(reduction='none')
    results['nll'] = torch.mean(nll_criterion(torch.log(softmaxes), labels)).item()
    
    ## Transform to onehot encoded labels
    labels_onehot = torch.FloatTensor(softmaxes.shape[0], softmaxes.shape[1])
    labels_onehot.zero_()
    labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)
    results['brier'] = torch.mean(torch.sum((softmaxes - labels_onehot) ** 2, dim=1,keepdim = True)).item()
    
    return results