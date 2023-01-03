import argparse
import os
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything
import torch
import mlflow
from prefect import flow
from data import MNISTDataModule
from model import LinearNet, ConvNet, register_model, load_model
from utils import get_parser_params, get_yaml_params, set_mlflow
from tunning import perform_tuning
from prefect.task_runners import SequentialTaskRunner

# TODO : Prefect extension -> Support **kwargs as parameters
@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=SequentialTaskRunner)
def run(config_path: str, prefect_args: dict) -> None:

    """
    Description :
        Main function for loading MNIST dataset, training and evaluating with registery

    """

    ### Setup ###
    seed_everything(42)
    os.makedirs("instance/", exist_ok=True)
    parser_args = get_parser_params()
    yaml_args = get_yaml_params(config_path)

    args = (
        argparse.Namespace(**parser_args, **yaml_args)
        if parser_args["use_parser"]
        else argparse.Namespace(**yaml_args, **prefect_args)
    )

    client = set_mlflow(args.mflow_tracking_uri, args.mlflow_experiment_name)
    dir_lightning_ckpt = os.path.join(args.dir_ckpts, "lightning")

    match args.model_name:
        case "LinearNet":
            model_class = LinearNet
            initial_params = {
                "input_dim": 28,
                "classes": args.classes,
                "dropout": 0.5,
                "learning_rate": 0.01,
                "n_layers": 3,
                "output_dims": [512, 128, 64],
                "optimizer": torch.optim.Adam,
            }
        case "ConvNet":
            model_class = ConvNet
            initial_params = {
                "input_dim": 28,
                "classes": args.classes,
                "num_conv_layers": 1,
                "num_filters": [32],
                "num_neurons": 16,
                "drop_conv2": 0.2,
                "drop_fc1": 0.5,
                "optimizer": torch.optim.Adam,
                "learning_rate": 0.01,
            }

        case _:
            raise NameError(f"Model not supported : {args.model_name}")

    ### Local Dataset ###

    datamodule = MNISTDataModule(data_dir=args.root_data, batch_size=args.batch_size)

    ###HyperParameter Search and Model Creation###
    if args.tunning and args.train:
        best_params = perform_tuning(
            model_class,
            datamodule,
            args.classes,
            args.pruning,
            args.epochs,
            args.percent_valid_examples,
            args.tune_trials,
            args.tune_timeout,
            args.model_name,
            args.tune_storage,
        )

    ###Training and testing###
    with mlflow.start_run():
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            max_epochs=args.epochs,
            accelerator="gpu",  # TODO
            devices=1 if torch.cuda.is_available() else None,
            default_root_dir=dir_lightning_ckpt,
        )

        if args.train:
            model_params = best_params if args.tunning else initial_params
            mlflow.log_params(model_params)

            model = model_class(**model_params)
            trainer.fit(model, datamodule=datamodule)  # ICI QUE SE TROUVE LE PROBLEME

            # Sauvegarde du modèle après entrainement complet, ajout transition

            register_model(
                client,
                model,
                args.model_name,
                args.stage,
                args.mlflow_artifact_path,
                args.transition,
            )
        else:
            # Chargement du modèle
            try:
                model = load_model(
                    args.model_name, args.stage
                )  # TODO si erreur de chemin specifié ?
            except mlflow.exceptions.MlflowException:
                raise

        test_results = trainer.test(model, dataloaders=datamodule.test_dataloader())

        for test_result in test_results:
            mlflow.log_metrics(test_result)


if __name__ == "__main__":
    run(
        config_path="./config.yaml",
        prefect_args={
            "model_name": "ConvNet",
            "train": False,
            "tunning": True,
            "pruning": False,
            "transition": False,
            "stage": "None",
            "use_parser": True,
        },
    )
