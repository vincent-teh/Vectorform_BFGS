import model_training
import torch.nn as nn
import torch
import os
import sys
import yaml

from conjgrad import ConjGrad
from memory_less_bfgs import MLBFGS
from model_training import ConvNet
from ombfgs import OMBFGS
from vmbfgs import VMBFGS
from torch.optim import Adam, LBFGS, Optimizer, SGD
from typing import List, Type


def get_optimizer(model: nn.Module, name: str, param: dict) -> Optimizer:
    """
    Generates respective optimizer based on name given.

    Args:
        model (nn.Module): Neural network model.
        name (str): Name of the optimizer.
        param (dict): Dictionary of parameters of the optimizer.
    """
    optimizer_map: dict[str, Type[Optimizer]] = {
        'SGD': SGD,
        'Adam': Adam,
        'cg': ConjGrad,
        'MLBFGS': MLBFGS,
        "VMBFGS": VMBFGS,
        "OMBFGS": OMBFGS,
        "LMBFGS": LBFGS
    }
    if name not in optimizer_map:
        raise ValueError(f'{name} optimizer is not supported yet')

    optimizer_constructor = optimizer_map[name]
    return optimizer_constructor(model.parameters(), **param)


def read_yml_file(config_path: str):
    with open(os.path.join(config_path, 'training_config.yml'), 'r') as f:
        data = yaml.safe_load(f)
        datapath = data['Paths'].get('data')
        resultpath = data['Paths'].get('results')
        datasets = [key for key, value in data['Dataset'].items()
                    if value == True]
        optimizers = data['Optimizer']
    return datapath, resultpath, datasets, optimizers


def training_pipeline(data_path: str, result_path: str, datasets: List[str], optimizers):
    """
    Standard training pipeline

    Args:
        data_path (str): Paths to which data is stored.
        result_path (str): Paths to which result is stored
        datasets (List[str]): List of datasets to be tested.
        optimizers (_type_): List of optimizer dictionary to be tested.
    """
    BATCH_SIZE = 100

    criteria = nn.CrossEntropyLoss()
    for dataset in datasets:    # Train for all enabled datasets.
        trainloader, n_channel, size_after_pool = \
            model_training.get_dataloader(dataset, data_path, BATCH_SIZE)
        for optimizer_name, param_set in optimizers.items():    # Train for all optimizers.
            for set_name, params in param_set.items():          # Train for all optimizers' params.
                if not params['Train']:
                    continue
                model = ConvNet(n_channel, size_after_pool)
                with torch.device("cuda" if torch.cuda.is_available() else "cpu") as device:
                    model.to(device)
                optimizer = get_optimizer(model, optimizer_name, params['param'])
                epoch = params['epoch']
                print(
                    f'=====Start training {optimizer_name} E{epoch} {params["param"]}=====')
                losses, accs, times = \
                    model_training.train_for_n_epochs(model, optimizer, epoch, criteria,
                                                    trainloader['train'],
                                                    trainloader['test'],
                                                    verbose=True)

                # File saving after trained for n epochs.
                # {root path}/SGD/
                filepath = os.path.join(result_path, optimizer_name)
                model_training.save_to_json(filepath, str(set_name), losses, accs, times)


def main() -> None:
    if sys.argv[1]:
        config_path = sys.argv[1]
    else:
        config_path = ''
    datapath, resultpath, datasets, optimizers = read_yml_file(config_path)
    training_pipeline(datapath, resultpath, datasets, optimizers)

if __name__ == "__main__":
    main()
