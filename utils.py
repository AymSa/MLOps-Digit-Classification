from prefect import task 
import yaml 
from yaml import SafeLoader
import argparse
import mlflow
from mlflow.client import MlflowClient

### GET PARAMETERS ###

@task 
def get_yaml_params(yaml_path : str) -> dict:
    with open(yaml_path) as f:
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


### SET URI MLFLOW###

@task
def set_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return MlflowClient(tracking_uri)




