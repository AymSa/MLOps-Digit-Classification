import argparse
import os
from typing import List
from typing import Optional
import sys
import logging

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

import mlflow
from mlflow.client import MlflowClient
from prefect import flow, task, get_run_logger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from multiprocessing import cpu_count
import yaml
from yaml.loader import SafeLoader
import random


class LightningNet(pl.LightningModule):
    def __init__(
        self,
        classes: int,
        learning_rate: float,
        dropout: float,
        output_dims: List[int],
        n_layers: int,
    ):
        super().__init__()

        assert len(output_dims) == n_layers

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.output_dims = output_dims

        layers: List[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, classes))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        logits = self.layers(data.view(-1, 28 * 28))
        return F.log_softmax(logits, dim=1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        data, target = batch
        output = self(data)

        pred = output.argmax(dim=1)

        return pred, target

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@task(retries=3, retry_delay_seconds=60)
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = cpu_count()
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        mnist_full = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


@task
def objective(
    trial: optuna.trial.Trial,
    ModelClass,
    datamodule,
    classes,
    epochs,
    percent_valid_examples,
) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("learning_rate", 0.001, 0.1)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        for i in range(n_layers)
    ]

    model = ModelClass(classes, lr, dropout, output_dims, n_layers)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=percent_valid_examples,
        enable_checkpointing=False,
        max_epochs=epochs,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    # trainer.tune(model, datamodule=datamodule)
    # lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)

    # new_lr = lr_finder.suggestion()

    hyperparameters = dict(
        n_layers=n_layers, dropout=dropout, output_dims=output_dims, lr=lr
    )
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()

@task 
def get_yaml_params(yaml_path : str) -> dict:
    with open("./config.yaml") as f:
        dict_yaml = yaml.load(f, Loader=SafeLoader)

    return dict_yaml

@task 
def get_parser_params() -> dict :
    parser = argparse.ArgumentParser(
        description="PytorchLightning MNIST Classification."
    )

    parser.add_argument(
        "--model_name",
        "-n",
        default="pytorch-mnist-simple-nn",
        help="Choose model name.",
    )
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="Train model.",
    )
    parser.add_argument(
        "--tunning",
        action="store_true",
        help="Activate the tunning feature.",
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )

    parser.add_argument(
        "--stage",
        "-s",
        default="Staging",
        help="Choose model stage.",
    )

    parser.add_argument(
        "--transition",
        action="store_true",
        help="Update the trained model to specified stage.",
    )

    parser.add_argument(
        "--use_parser",
        "-u",
        action="store_true",
        help="Use the argparser parameters",
    )

    args = parser.parse_args()

    return vars(args)


@flow
def perform_tuning(
    ClassModel,
    datamodule,
    classes,
    use_prunning,
    epochs,
    percent_valid_examples,
    n_trials,
    timeout,
):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "optuna"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    pruner = (
        optuna.pruners.MedianPruner() if use_prunning else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )

    func = lambda trial: objective(
        trial, ClassModel, datamodule, classes, epochs, percent_valid_examples
    )
    study.optimize(func, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


@task
def set_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return MlflowClient(tracking_uri)


@task
def process_tuned_params(best_params, classes):
    n_layers = best_params["n_layers"]
    output_dims = []

    for i in range(n_layers):

        output_dims.append(best_params[f"n_units_l{i}"])
        del best_params[f"n_units_l{i}"]

    new_best_params = best_params | {"output_dims": output_dims, "classes": classes}

    return new_best_params


@task
def register_model(client, model, model_name, stage, artifact_path, transition):
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path,
        registered_model_name=model_name,
    )
    if transition:
        mlflow_entity = client.get_latest_versions(name=model_name, stages=["None"])
        client.transition_model_version_stage(
            name=model_name, version=mlflow_entity[0].version, stage=stage
        )


@task(retries=3, retry_delay_seconds=60)
def load_model(model_name, stage):
    return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}")

#TODO : Prefect extension -> Support **kwargs as parameters 
@flow(version=os.getenv("GIT_COMMIT_SHA"))
def main(config_path : str, prefect_args : dict) -> None:

    """
    Description :
        Main function for loading MNIST dataset, training and evaluating with registery

    """

    ### Setup ###
    initial_params = {
        "classes": 10,
        "dropout": random.random(),
        "learning_rate": random.random(),
        "n_layers": 3,
        "output_dims": [random.randint(20, 200) for _ in range(3)],
    }
    parser_args = get_parser_params()
    yaml_args = get_yaml_params(config_path)
    
    args = argparse.Namespace(**parser_args, **yaml_args) if parser_args['use_parser'] else argparse.Namespace(**yaml_args, **prefect_args) 
    
    client = set_mlflow(args.mflow_tracking_uri, args.mlflow_experiment_name)
    dir_lightning_ckpt = os.path.join(args.dir_ckpts, "lightning")

    ### Local Dataset ###

    datamodule = MNISTDataModule(data_dir=args.root_data, batch_size=args.batch_size)

    ###HyperParameter Search and Model Creation###
    if args.tunning:
    
        best_params = perform_tuning(
            LightningNet,
            datamodule,
            args.classes,
            args.pruning,
            args.epochs,
            args.percent_valid_examples,
            args.tune_trials,
            args.tune_timeout,
        )
        best_params = process_tuned_params(best_params, args.classes)

    model_params = best_params if args.tunning else initial_params

    with mlflow.start_run():

        mlflow.log_params(model_params)

        model = LightningNet(**model_params)

        ###Training and testing###

        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            default_root_dir=dir_lightning_ckpt,
        )

        if args.train:
            trainer.fit(model, train_dataloaders=datamodule.train_dataloader())

            # Sauvegarde du modèle après entrainement complet, ajout transition
            register_model(
                client,
                model,
                args.model_name,
                args.stage,
                args.mlflow_artifact_path,
                args.transition,
            )

        # Chargement du modèle

        model = load_model(args.model_name, args.stage)

        test_results = trainer.test(model, dataloaders=datamodule.test_dataloader())

        for test_result in test_results:
            mlflow.log_metrics(test_result)


if __name__ == "__main__":
    main(config_path='./config.py', prefect_args = {'model_name' : "pytorch-mnist-simple-nn", 
                                                    'train' : False, 
                                                    'tunning' : False, 
                                                    'pruning' : False, 
                                                    'transition' : False, 
                                                    'stage': "Staging", 
                                                    'use_parser' : False})
