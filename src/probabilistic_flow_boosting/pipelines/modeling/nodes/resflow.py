import logging
import os
import uuid
import warnings
from functools import partial

import optuna
import pandas as pd
import torch
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import pathlib

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from probabilistic_flow_boosting.models.resflow import ResFlow, ResFlowDataModule
from probabilistic_flow_boosting.pipelines.modeling.pytorch_lightning import (
    PyTorchLightningPruningCallback,
)
from probabilistic_flow_boosting.pipelines.modeling.utils import JoblibStudy


optuna.logging.enable_propagation()
logging.basicConfig(level=logging.error)


class CudaOutOfMemory(optuna.exceptions.OptunaError):
    def __init__(self, message):
        super().__init__(message)


def train_resflow(x_train, y_train, n_epochs, dataset, fold, uuid_no, patience, split_size, batch_size, model_hyperparams):
    model = ResFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams)
    datamodule = ResFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)

    # if path does not exist, create path
    path= f"tmp/resflow-{dataset}-{str(fold)}-{uuid_no}"
    pathlib.Path(f'tmp/{path}').mkdir(parents=True, exist_ok=True) 

    callbacks = [
        StochasticWeightAveraging(swa_lrs=1e-2),
        EarlyStopping(monitor="val_nll", patience=patience),
        ModelCheckpoint(monitor="val_nll", dirpath=f"{path}/", filename=f"{path.split('/')[-1]}"),
    ]
    trainer = Trainer(
        max_epochs=n_epochs,
        devices=1,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("best_model_path returned: ", best_model_path)
    return best_model_path


def objective(trial, x_train, y_train, n_epochs, patience, split_size, batch_size, hparams) -> float:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.49)
    
    hidden_dim = trial.suggest_int("hidden_dim", *hparams["hidden_dim"])
    dim = trial.suggest_int("dim", *hparams["dim"])
    depth = trial.suggest_int("depth", *hparams["depth"])
    hidden_dropout= trial.suggest_float("hidden_dropout", *hparams["hidden_dropout"])
    residual_dropout= trial.suggest_float("residual_dropout", *hparams["residual_dropout"])
    flow_num_blocks = trial.suggest_int("flow_num_blocks", *hparams["flow_num_blocks"])
    flow_layers = trial.suggest_int("flow_layers", *hparams["flow_layers"])

    model_hyperparams = dict(
        hidden_dim = hidden_dim,
        depth =depth,
        hidden_dropout= hidden_dropout,
        residual_dropout= residual_dropout,
        flow_num_blocks= flow_num_blocks,
        flow_layers = flow_layers,
        dim=dim
    )
    print("\nhps:", model_hyperparams)
    model = ResFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams)
    datamodule = ResFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)

    try:
        trainer = Trainer(
            logger=True,
            log_every_n_steps=100,
            enable_checkpointing=False,
            max_epochs=n_epochs,
            accelerator="gpu",
            devices=1,
            callbacks=[
                StochasticWeightAveraging(swa_lrs=1e-2),
                EarlyStopping(monitor="val_nll", patience=patience),
                PyTorchLightningPruningCallback(trial, monitor="val_nll"),
            ],
        )

        trainer.logger.log_hyperparams(model_hyperparams)
        trainer.fit(model, datamodule=datamodule)
        trial.set_user_attr("best_epoch", trainer.early_stopping_callback.stopped_epoch)
        trial.set_user_attr("total_epochs", trainer.current_epoch)
        print("trainer.early_stopping_callback.best_score.item()", trainer.early_stopping_callback.best_score.item())
        return trainer.early_stopping_callback.best_score.item()
    except RuntimeError as exc:
        raise CudaOutOfMemory(str(exc))


def modeling_resflow_single_run(x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hyperparams,
    dataset,
    fold,
    split_size=0.8,
    n_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1000,
    random_seed: int = 42,
    n_trials: int = 400
):
    seed_everything(random_seed, workers=True)  # sets seeds for numpy, torch and python.random.
    uuid_no = uuid.uuid4()

    best_model_path = train_resflow(x_train=x_train, y_train=y_train, n_epochs=n_epochs, 
        dataset=dataset,
        fold=fold,
        uuid_no=uuid_no,
        patience=patience, split_size=split_size, batch_size=batch_size, model_hyperparams=model_hyperparams)
    
    model = ResFlow.load_from_checkpoint(
        best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams
    )
    print("best_model_path : ", best_model_path)
    # os.remove(best_model_path)


    return model

def modeling_resflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hyperparams,
    dataset,
    fold,
    split_size=0.8,
    n_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1000,
    random_seed: int = 42,
    n_trials: int = 400
):
    seed_everything(random_seed, workers=True)  # sets seeds for numpy, torch and python.random.
    uuid_no = uuid.uuid4()

    objective_func = partial(
        objective,
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        patience=patience,
        split_size=split_size,
        batch_size=batch_size,
        hparams=model_hyperparams,
    )

    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=n_epochs)
    # sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    sampler = optuna.samplers.RandomSampler()

    # create these paths.
    path= f"data/07_model_output/UCI/{dataset}/{fold}"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

    study = JoblibStudy(
        direction="minimize", pruner=pruner, sampler=sampler, 
        storage=f"sqlite:///{path}/resflow-{dataset}-{str(fold)}-{uuid_no}.sqlite3",
        study_name=f"resflow-{dataset}-{str(fold)}-{uuid_no}",
    )

    study.optimize(objective_func, n_trials=n_trials, timeout=60 * 60 * 6, catch=(CudaOutOfMemory), n_jobs=2)

    results = study.trials_dataframe()
    results = pd.DataFrame(results)
    results.to_csv(f"{path}/results_{dataset}_{str(fold)}_resflow.csv", mode='a')

    trial = study.best_trial
    print("After training best trial from study :", trial.params)

    model_params = trial.params

    best_model_path = train_resflow(
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        dataset=dataset,
        fold=fold,
        uuid_no=uuid_no,
        patience=patience,
        split_size=split_size,
        batch_size=batch_size,
        model_hyperparams=model_params,
    )
    print("Re-Training best over.")
    model = ResFlow.load_from_checkpoint(
        best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_params
    )
    # os.remove(best_model_path)
    print("Best trial: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return model, results, study, model_params



def model_best_resflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_parameters: dict,
    random_seed: int,
    split_size=0.8,
    n_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1000,
    n_trials: int = 400
):
    seed_everything(random_seed, workers=True)  # sets seeds for numpy, torch and python.random.

    best_model_path = train_resflow(
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        patience=patience,
        split_size=split_size,
        batch_size=batch_size,
        model_hyperparams=best_parameters,
    )
    model = ResFlow.load_from_checkpoint(
        best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **best_parameters
    )
    os.remove(best_model_path)

    return model