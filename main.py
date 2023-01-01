import argparse
import os
import pytorch_lightning as pl
import torch
import mlflow
from prefect import flow
import random
import sys 

from data import MNISTDataModule
from model import LinearNet, ConvNet, register_model, load_model
from utils import get_parser_params, get_yaml_params, set_mlflow
from tunning import perform_tuning

#TODO : Prefect extension -> Support **kwargs as parameters 
@flow(version=os.getenv("GIT_COMMIT_SHA"))
def run(config_path : str, prefect_args : dict) -> None:

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

    match args.model_name:
        case "LinearNet":
            model_class = LinearNet
        case "ConvNet":
            model_class = ConvNet
        case _:
            raise NameError(f'Model not supported : {args.model_name}' )

    ### Local Dataset ###

    datamodule = MNISTDataModule(data_dir=args.root_data, batch_size=args.batch_size)

    ###HyperParameter Search and Model Creation###
    if args.tunning:
        #Set optuna 
        #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
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
            args.tune_storage
        )
        
    ###Training and testing###

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        accelerator="gpu", #TODO
        devices=1 if torch.cuda.is_available() else None,
        default_root_dir=dir_lightning_ckpt,
    )

    with mlflow.start_run():
        if args.train:
            model_params = best_params if args.tunning else initial_params
            mlflow.log_params(model_params)

            model = model_class(**model_params)
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
        else:
            # Chargement du modèle

            model = load_model(args.model_name, args.stage) #TODO si erreur de chemin specifié ?

        test_results = trainer.test(model, dataloaders=datamodule.test_dataloader())

        for test_result in test_results:
            mlflow.log_metrics(test_result)


if __name__ == "__main__":
    pl.utilities.seed.seed_everything(42)
    run(config_path='./config.yaml', prefect_args = {'model_name' : "ConvNet", 
                                                    'train' : True, 
                                                    'tunning' : True, 
                                                    'pruning' : False, 
                                                    'transition' : False, 
                                                    'stage': "Staging", 
                                                    'use_parser' : True,})
