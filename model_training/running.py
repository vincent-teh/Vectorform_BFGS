import model_training
import torch
import torch.nn as nn

from conjgrad import ConjGrad
from memory_less_bfgs import MLBFGS
from vmbfgs import VMBFGS

from model_training import FileSaver, JsonFileSaver
from model_training import ConvNet
from torch.optim import Adam, SGD, Optimizer


def get_optimizer(model: nn.Module, name: str, param: dict) -> Optimizer:
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
        return MLBFGS(model.parameters(), **param)
    if name == "VMBFGS":
        return VMBFGS(model.parameters(), **param)
    raise ValueError(f'{name} optimizer is not supported yet')


class TrainingPipelineBuilder:
    """Standard training pipeline for CNN
    Args:
        file_saver (FileSaver): Operation in which result was saved, defaults to JSON format with incremental name.
    """
    def __init__(self,
                 dataset,
                 file_saver: FileSaver = JsonFileSaver()) -> None:
        self.file_saver = file_saver
        self.criteria = nn.CrossEntropyLoss()
        self.dataset = dataset

    def run(self, data_path: str, result_path: str, datasets: list[str], optimizers):
        """
        Args:
            data_path (str): Paths to which data is stored.
            result_path (str): Paths to which result is stored
            datasets (List[str]): List of datasets to be tested.
            optimizers (_type_): List of optimizer dictionary to be tested.
        """
        BATCH_SIZE = 100
        self.trainloader, self.n_channel, self.size_after_pool = \
            model_training.get_dataloader(self.dataset, data_path, BATCH_SIZE)
        for self.optimizer_name, param_set in optimizers.items():    # Train for all optimizers.
            for self.set_name, self.params in param_set.items():          # Train for all optimizers' params.
                if not self.params['Train']:
                    continue
                self._run_single()

    def _run_single(self):
        model = ConvNet(self.n_channel, self.size_after_pool)
        with torch.device("cuda" if torch.cuda.is_available() else "cpu") as device:
            model.to(device)

        optimizer = get_optimizer(model, self.optimizer_name, self.params['param'])
        epoch = self.params['epoch']
        print(f'=====Start training {self.optimizer_name} E{epoch} {self.params["param"]}=====')
        losses, accs, times = model_training.train_for_n_epochs(
            model, optimizer, epoch, self.criteria, self.trainloader['train'], self.trainloader['test'], verbose=True)

        # ================= Saving the files ==================
        data = {'losses': losses, 'accs': accs, 'times': times}
        filepath = os.path.join(self.result_path, self.optimizer_name)
        self.file_saver.save(data, filepath, str(self.set_name))
