import time
import torch
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Literal, get_args, Dict
import torchvision
import torchvision.transforms as Transforms


def train_for_n_epochs(model: nn.Module,
                       optimizer: Optimizer,
                       epoch: int,
                       criteria,
                       training_dataloader: DataLoader,
                       testing_dataloader: DataLoader,
                       verbose: bool = False
                       ) -> tuple[list, list]:

    printfreq = int(len(training_dataloader) * 0.1)
    loss = []
    time = []
    acc  = []
    for _ in range(epoch):
        start_time = time.time()
        print(f'Epoch #{epoch}')
        for i, (inputs, labels) in enumerate(training_dataloader):
            def closure():
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criteria(outputs, labels)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            running_loss += loss.item()

            # Print, append the loss value and reset the running loss
            if i % printfreq == printfreq-1:
                loss.append(running_loss/printfreq)
                if verbose:
                    print(running_loss/printfreq)
                running_loss = 0
        time.append(time.time() - start_time) # In second
        acc.append(testing_evaluation(model, testing_dataloader))

        print(f'{optimizer.__class__,__name__}\'s loss == {loss[-1]}')
    return loss, acc, time


def testing_evaluation(model: nn.Module,
                       testing_dataloader: DataLoader):
    n_correct = len(testing_dataloader)
    n_samples = 0
    with torch.no_grad():
        print('=========Start of testing for=========.')
        for i, (image, label) in tqdm(enumerate(testing_dataloader)):
            image.reshape(-1)
            outputs = model(image)
            # value, index
            _, predictions = torch.max(outputs, 1)
            n_correct += (predictions == label).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'==========Accuracy is {acc:.2f}%==========')
    return acc


_DATASET = Literal['MNIST', 'CIFAR10']

def get_dataloader(dataset: _DATASET, path: str, batch_size: int
                   ) -> Dict["train": DataLoader, "test": DataLoader]:

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
    elif dataset == 'CIFAR10':
        dataset_fn = torchvision.datasets.CIFAR10
    train = dataset_fn(root=path, train=True, download=True, transform=transform)
    test  = dataset_fn(root=path, train=True, download=True, transform=transform)

    return {"train": torch.utils.data.DataLoader(train, batch_size=batch_size),
            "test":  torch.utils.data.DataLoader(test, batch_size=batch_size)}


