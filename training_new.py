from model_training import ConvNet, train_for_n_epochs
import model_training
from torch.optim import SGD
import torch.nn as nn


def main():
    trainloader = model_training.get_dataloader('MNIST', , 100)
    model = ConvNet(3, 700)
    optimizer = SGD(lr=0.001)
    epoch = 1
    criteria = nn.CrossEntropyLoss()
    loss, acc, time = train_for_n_epochs(model, optimizer, epoch, criteria,
                                         trainloader['train'], trainloader['test'])
    print(loss, acc, time)

if __name__ == "__main__":
    main()
