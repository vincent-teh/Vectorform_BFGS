import json
from dataclasses import dataclass
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as Transforms
import torch.utils.data as data
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Literal, List


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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image, label = image.to(device), label.to(device)
            image.reshape(-1)
            outputs = model(image)
            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += label.shape[0]
            n_correct += (predictions == label).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'==========Accuracy is {acc:.2f}%==========')
    return acc


@dataclass
class DataSetParam:
    dimension: int
    dataset_fn: Callable[[Any], torchvision.datasets.VisionDataset]
    size_after_pool: int
    path: str = ''

    @property
    def transform(self):
        normalize_param = (0.5,) * self.dimension
        return Transforms.Compose([Transforms.ToTensor(),
                                    Transforms.Normalize(normalize_param, normalize_param)])

    def train(self, path: str) -> torchvision.datasets.VisionDataset:
        return self.dataset_fn(
            root=path,train=True, download=True, transform=self.transform) # type: ignore

    def test(self, path: str) -> torchvision.datasets.VisionDataset:
        return self.dataset_fn(
            root=path, train=False, download=True, transform=self.transform) # type: ignore


DataLoaderMap = {
    "MNIST": DataSetParam(1, torchvision.datasets.MNIST, 784),
    "CIFAR10": DataSetParam(3, torchvision.datasets.CIFAR10, 1024),
    "FMNIST": DataSetParam(1, torchvision.datasets.FashionMNIST, 784),
}


_DATASET = Literal['MNIST', 'CIFAR10', 'FMNIST']

def get_dataloader(
    dataset: _DATASET, path: str, batch_size: int
    ) -> tuple[dict[str, data.DataLoader], dict[str, data.DataLoader], int, int]:
    loader = DataLoaderMap[dataset]
    train = loader.train(path)
    test  = loader.test(path)
    return {"train": data.DataLoader(train, batch_size=batch_size),
            "test":  data.DataLoader(test, batch_size=batch_size)}, \
            loader.dimension, \
            loader.size_after_pool


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
