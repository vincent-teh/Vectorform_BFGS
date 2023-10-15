import model_training
import torch.nn as nn
import os
import yaml

from model_training import ConvNet
from torch.optim import SGD, Adam


def get_optimizer(model, name, param):
    if name == 'SGD':
        return SGD(model.parameters(), **param)
    if name == 'Adam':
        return Adam(model.parameters(), **param)
    raise ValueError(f'{name} optimizer is not supported yet')


def main():
    # Hard-coded values
    BATCH_SIZE = 100

    with open('training_config.yml', 'r') as f:
        data = yaml.safe_load(f)
        PATHS = data['Paths']
        datasets = [key for key, value in data['Dataset'].items()
                    if value == True]
        optimizers = data['Optimizer']

    criteria = nn.CrossEntropyLoss()
    for dataset in datasets:
        trainloader, n_channel, size_after_pool = \
            model_training.get_dataloader(dataset, PATHS['data'], BATCH_SIZE)
        for optimizer_name, param_set in optimizers.items():
            for _, params in param_set.items():
                if not params['Train']:
                    continue
                model = ConvNet(n_channel, size_after_pool)
                optimizer = get_optimizer(
                    model, optimizer_name, params['param'])
                epoch = params['epoch']
                print(
                    f'=====Start training {optimizer_name} E{epoch} {params["param"]}=====')
                losses, accs, times = \
                    model_training.train_for_n_epochs(model, optimizer, epoch, criteria,
                                                      trainloader['train'],
                                                      trainloader['test'],
                                                      verbose=True)

                filepath = os.path.join(PATHS['results'], optimizer_name)
                filename = optimizer_name
                for _, value in params['param'].items():
                    filename += str(value)
                filename += '.json'
                model_training.save_to_json(filepath, filename, losses, accs, times)


if __name__ == "__main__":
    main()
