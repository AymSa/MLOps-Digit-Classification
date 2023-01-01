from prefect import task, flow
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl

@task
def process_tuned_params(model_name, best_params, input_dim, classes):
    best_params['optimizer'] = getattr(torch.optim, best_params['optimizer_name'])
    del best_params['optimizer_name']

    match model_name:
        case "LinearNet":
            n_layers = best_params["n_layers"]
            output_dims = []
            for i in range(n_layers):
                output_dims.append(best_params[f"n_units_l{i}"])
                del best_params[f"n_units_l{i}"]

            new_best_params = best_params | {"output_dims": output_dims, "input_dim":input_dim ,"classes": classes}
            return new_best_params
        case "ConvNet":

            num_conv_layers = best_params["num_conv_layers"]
            num_filters = []
            for i in range(num_conv_layers):
                num_filters.append(best_params[f"num_filter_{i}"])
                del best_params[f"num_filter_{i}"]
            return best_params | {"num_filters":num_filters ,"input_dim":input_dim ,"classes": classes}
        case _:
            raise NameError(f'Model not supported : {model_name}')


@task
def objective(
    trial: optuna.trial.Trial,
    model_class,
    datamodule,
    classes,
    epochs,
    percent_valid_examples,
    model_name
) -> float:

    input_dim = datamodule.input_dim
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(torch.optim, optimizer_name)

    match model_name:
        case "LinearNet":
            # We optimize the number of layers, hidden units in each layer and dropouts.
            n_layers = trial.suggest_int("n_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.2, 0.5)
            output_dims = [
                trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
                for i in range(n_layers)
            ]
            
            
            hyperparameters = dict(
                n_layers=n_layers, dropout=dropout, output_dims=output_dims, lr=lr, optimizer_name = optimizer_name
            )
            
            model = model_class(input_dim, classes, dropout, output_dims, n_layers, optimizer, lr)

        case "ConvNet":
            num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)  # Number of convolutional layers
            num_filters = [int(trial.suggest_int("num_filter_"+str(i), 16, 128, 16))
                        for i in range(num_conv_layers)]              # Number of filters for the convolutional layers
            num_neurons = trial.suggest_int("num_neurons", classes, 400, 10)  # Number of neurons of FC1 layer
            drop_conv2 = trial.suggest_float("drop_conv2", 0.2, 0.5)     # Dropout for convolutional layer 2
            drop_fc1 = trial.suggest_float("drop_fc1", 0.2, 0.5)         # Dropout for FC1 layer

            hyperparameters = dict(
                num_conv_layers=num_conv_layers, num_filters=num_filters, num_neurons=num_neurons, drop_conv2=drop_conv2, drop_fc1=drop_fc1, lr=lr, optimizer_name = optimizer_name
            )

            model = model_class(input_dim, classes, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1, optimizer, lr)
        case _:
            raise NameError(f'Model not supported : {model_name}')

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=percent_valid_examples,
        enable_checkpointing=False,
        max_epochs=epochs,
        accelerator="gpu", #TODO : add customisation for gpu 
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()


@flow
def perform_tuning(
    model_class,
    datamodule,
    classes,
    use_prunning,
    epochs,
    percent_valid_examples,
    n_trials,
    timeout,
    model_name, 
    storage_name
):
    
    pruner = (
        optuna.pruners.MedianPruner() if use_prunning else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=model_name,
        storage=storage_name,
        load_if_exists=True,
        direction='maximize',
        pruner=pruner,
    )

    func = lambda trial: objective(
        trial, model_class, datamodule, classes, epochs, percent_valid_examples, model_name
    )
    study.optimize(func, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return process_tuned_params(model_name, trial.params, datamodule.input_dim, classes)


