import model_training
import torch.nn as nn
import torch
import os
import yaml

from conjgrad import ConjGrad
from model_training import ConvNet
from torch.optim import SGD, Adam
from memory_less_bfgs import MLBFGS


def get_optimizer(model: nn.Module, name: str, param: dict):
    """
    Generates respective optimizer based on name given.

    Args:
        model (nn.Module): Neural network model.
        name (str): Name of the optimizer.
        param (dict): Dictionary of parameters of the optimizer.
    """
    if name == 'SGD':
        return SGD(model.parameters(), **param)
    if name == 'Adam':
        return Adam(model.parameters(), **param)
    if name == 'cg':
        return ConjGrad(model.parameters(), **param)
    if name == 'MLBFGS':
        return MLBFGS(model.parameters())
    raise ValueError(f'{name} optimizer is not supported yet')


def main(config_path: str, root_path: str):
    # Hard-coded values
    BATCH_SIZE = 100

    with open(os.path.join(config_path, 'training_config.yml'), 'r') as f:
        data = yaml.safe_load(f)
        PATHS = data['Paths']
        datasets = [key for key, value in data['Dataset'].items()
                    if value == True]
        optimizers = data['Optimizer']

    criteria = nn.CrossEntropyLoss()
    for dataset in datasets:    # Train for all enabled datasets.
        trainloader, n_channel, size_after_pool = \
            model_training.get_dataloader(dataset, PATHS['data'], BATCH_SIZE)
        for optimizer_name, param_set in optimizers.items():    # Train for all optimizers.
            for set_name, params in param_set.items():          # Train for all optimizers' params.
                if not params['Train']:
                    continue
                model = ConvNet(n_channel, size_after_pool)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                filepath = os.path.join(root_path, PATHS['results'], optimizer_name)
                model_training.save_to_json(filepath, str(set_name), losses, accs, times)


if __name__ == "__main__":
    main('', '')
