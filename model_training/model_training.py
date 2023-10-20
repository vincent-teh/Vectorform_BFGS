import json
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as Transforms

from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Literal, get_args, Dict, List


def train_for_n_epochs(model: nn.Module,
                       optimizer: Optimizer,
                       epoch: int,
                       criteria,
                       training_dataloader: DataLoader,
                       testing_dataloader: DataLoader,
                       verbose: bool = False
                       ) -> tuple[list, list]:
    """
    Train the model for given number of epochs.

    Args:
        model (nn.Module): Model of the neural network.
        optimizer (Optimizer): Choice of optimizer.
        epoch (int): no. of epoch to be trained.
        criteria (_type_): loss function of selected.
        training_dataloader (DataLoader): training data loader.
        testing_dataloader (DataLoader): testing data loader.
        verbose (bool, optional): Shows loss value per printing frequency. Defaults to False.

    Returns:
        train_losses (list): List of training losses.
        accuracies (list): List of testing accuracy.
        train_times (list): List of training time needed per epochs.
    """


    printfreq = int(len(training_dataloader) * 0.1)
    accuracies   = []
    train_losses = []
    train_times  = []
    running_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for iteration in range(1,epoch+1):
        start_time = time.time()
        print(f'Epoch #{iteration}')
        for i, (inputs, labels) in enumerate(training_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            def closure():
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criteria(outputs, labels)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            if isinstance(loss, float):
                loss = torch.Tensor([loss,])
            running_loss += loss.item()

            # Print, append the loss value and reset the running loss
            if i % printfreq == printfreq-1:
                train_losses.append(running_loss/printfreq)
                if verbose:
                    print(running_loss/printfreq)
                running_loss = 0
        print(f'{optimizer.__class__.__name__}\'s loss == {train_losses[-1]}')
        train_times.append(time.time() - start_time) # In second
        accuracies.append(testing_evaluation(model, testing_dataloader))
        if torch.isnan(loss).any():
            print(f"===Loss value drop to NaN at epoch {iteration}, stop training===")
            break
    return train_losses, accuracies, train_times


def testing_evaluation(model: nn.Module,
                       testing_dataloader: DataLoader):
    """
    Evaluate the model's accuracy.

    Args:
        model (nn.Module): Model of the neural network.
        testing_dataloader (DataLoader): Testing dataloader.

    Returns:
        float: testing accuracy in percentage.
    """
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        print('=========Start of testing for=========.')
        for i, (image, label) in tqdm(enumerate(testing_dataloader)):
            image.reshape(-1)
            outputs = model(image)
            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += label.shape[0]
            n_correct += (predictions == label).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'==========Accuracy is {acc:.2f}%==========')
    return acc


_DATASET = Literal['MNIST', 'CIFAR10']

def get_dataloader(dataset: _DATASET, path: str, batch_size: int
                   ):

    if dataset not in get_args(_DATASET):
        raise ValueError(f'Dataset {dataset} not supported')
    if dataset == 'CIFAR10':
        transform = Transforms.Compose([Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dimension = 3
    else:
        transform = Transforms.Compose([Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,), (0.5,))])
        dimension = 1
    if dataset == "MNIST":
        dataset_fn = torchvision.datasets.MNIST
        size_after_pooling = 784
    elif dataset == 'CIFAR10':
        dataset_fn = torchvision.datasets.CIFAR10
        size_after_pooling = 1024
    train = dataset_fn(root=path, train=True, download=True, transform=transform)
    test  = dataset_fn(root=path, train=False, download=True, transform=transform)

    return {"train": torch.utils.data.DataLoader(train, batch_size=batch_size),
            "test":  torch.utils.data.DataLoader(test, batch_size=batch_size)}, \
            dimension, size_after_pooling


def save_to_json(filepath:str,
                 filename: str,
                 losses: List[float],
                 accuracies: List[float],
                 times: List[float]):
    """
    Save the training results into json file.

    Args:
        filepath (str): Folder containing the results.
        filename (str): Name of the file.
        losses (List[float]): List of loss value.
        accuracies (List[float]): List of accuracy value.
        times (List[float]): List of time taken for each epoch.
    """
    data = {'losses': losses,
            'accuracies': accuracies,
            'times': times}
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    filename = os.path.join(filepath, filename)
    i = 0
    _filename = filename + '-' + str(i) + '.json'
    while os.path.exists(_filename):
        i += 1
        _filename = filename + '-' + str(i) + '.json'

    with open(_filename, 'w') as file:
        json.dump(data, file)
