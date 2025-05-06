import uuid
from sklearn.model_selection import train_test_split
from itertools import product
from src.probabilistic_flow_boosting.extras.datasets.uci_dataset import UCIDataSet
from src.probabilistic_flow_boosting.models.nodeflow import NodeFlow, NodeFlowDataModule
# from src.probabilistic_flow_boosting.pipelines.modeling.nodes.nodeflow import modeling_nodeflow, model_best_nodeflow
from src.probabilistic_flow_boosting.pipelines.modeling.nodes.resflow import modeling_resflow, model_best_resflow, modeling_resflow_single_run
from src.probabilistic_flow_boosting.pipelines.modeling.nodes.multivariate import modeling_multivariate
from src.probabilistic_flow_boosting.pipelines.reporting.nodes import calculate_metrics_nodeflow, calculate_nll_catboost, calculate_metrics_resflow
import warnings
import pdb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import numpy as np
import pprint

parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('--dataset',type=str, help='dataset name as the corresponding folder name')
parser.add_argument('--n_epochs',type=int, default= 400, help='dataset name as the corresponding folder name')
parser.add_argument('--n_trials',type=int, default= 400, help='number of trials')
parser.add_argument('--fold',type=int, default= 0, help='number of trials')
parser.add_argument('--debug',type=bool, default=False, help='if trail run')

args = parser.parse_args()
best_results = dict(
    best_nlls =[],
    best_rmse1s=[],
    best_rmse2s=[],
    best_crpss=[],
    best_nll_times=[],
    best_rmse_times=[],
    best_crps_times=[]
)

if args.dataset =="year-prediction-msd":
    n_fold = 1
    model_hyperparams=dict(
        hidden_dim = [32, 1024],
        dim = [32, 1024],
        depth =[1, 20],
        hidden_dropout= [0.0, 0.9],
        residual_dropout= [0.0, 0.9],
        flow_num_blocks= [1, 20],
        flow_layers = [1, 20],
    )
    load_args =dict(
        data_delimiter=","
    )
else:
    if args.dataset == "protein-tertiary-structure":
        n_fold = 5
    else:
        n_fold = 20
    # ResFlow Params
    model_hyperparams = dict(
        hidden_dim = [32, 1024],
        dim = [32, 1024],
        depth =[1, 20],
        hidden_dropout= [0.0, 0.9],
        residual_dropout= [0.0, 0.9],
        flow_num_blocks= [1, 20],
        flow_layers = [1, 20],
    )
    load_args={}
    # model_hyperparams = dict(
    #     hidden_dim = 216,
    #     depth =10,
    #     hidden_dropout= 0.1,
    #     residual_dropout= 0.1,
    #     flow_num_blocks= 10,
    #     flow_layers = 10,
    # )
if args.debug: # if debug run
    args.dataset="concrete"
    n_fold=1
    args.n_trials = 1
    args.n_epochs = 400


print(args)
f= args.fold

# for f in range(n_fold):
print(f"########################## FOLD {str(f)} #############################")
## dataset
x_train = UCIDataSet(
    filepath_data=f"data/01_raw/UCI/{args.dataset}/data.txt",
    filepath_index_columns=f"data/01_raw/UCI/{args.dataset}/index_features.txt",
    filepath_index_rows=f"data/01_raw/UCI/{args.dataset}/index_train_{str(f)}.txt",
    load_args =load_args
).load()
y_train = UCIDataSet(
    filepath_data=f"data/01_raw/UCI/{args.dataset}/data.txt",
    filepath_index_columns=f"data/01_raw/UCI/{args.dataset}/index_target.txt",
    filepath_index_rows=f"data/01_raw/UCI/{args.dataset}/index_train_{str(f)}.txt",
    load_args =load_args
).load()

x_test = UCIDataSet(
    filepath_data=f"data/01_raw/UCI/{args.dataset}/data.txt",
    filepath_index_columns=f"data/01_raw/UCI/{args.dataset}/index_features.txt",
    filepath_index_rows=f"data/01_raw/UCI/{args.dataset}/index_test_{str(f)}.txt",
    load_args =load_args
).load()

y_test = UCIDataSet(
    filepath_data=f"data/01_raw/UCI/{args.dataset}/data.txt",
    filepath_index_columns=f"data/01_raw/UCI/{args.dataset}/index_target.txt",
    filepath_index_rows=f"data/01_raw/UCI/{args.dataset}/index_test_{str(f)}.txt",
    load_args =load_args
).load()




# train and hp tune
# model, results, study, model_params = modeling_resflow(x_train, y_train, model_hyperparams, split_size=0.8, n_epochs=args.n_epochs, patience=400, random_seed=42, batch_size=2048, n_trials=args.n_trials)
if args.debug:
    model = modeling_resflow_single_run(x_train, y_train, model_hyperparams, dataset=args.dataset, fold=f, split_size=0.8, n_epochs=args.n_epochs, patience=400, random_seed=42, batch_size=2048, n_trials=args.n_trials)
else:
    model, results, study, model_params = modeling_resflow(x_train, y_train, model_hyperparams, dataset=args.dataset,fold=f,split_size=0.8, n_epochs=args.n_epochs, patience=400, random_seed=42, batch_size=2048, n_trials=args.n_trials)
    # model, results, study, model_params = modeling_resflow(x_train, y_train, model_hyperparams, split_size=0.8, n_epochs=10, patience=400, random_seed=42, batch_size=2048, n_trials=3)

# pdb.set_trace()
# Test on current random seed
print("\nCalulating metrics:")
nll, rmse_1, rmse_2, crpss, nll_time, rmse_time, crps_time = calculate_metrics_resflow(model, x_train, y_train, x_test.values, y_test.values, num_samples=1000, batch_size=2048, sample_batch_size=4)

print(f"test nll : {nll}, test rmse 1:{rmse_1}, test rmse 2:{rmse_2}, crps: {crpss}, nll time :{nll_time}, rmse_time: {rmse_time}, crps_time: {crps_time}")

best_results["best_nlls"].append(nll)
best_results["best_rmse1s"].append(rmse_1)
best_results["best_rmse2s"].append(rmse_2)
best_results["best_crpss"].append(crpss)
best_results["best_nll_times"].append(nll_time)
best_results["best_rmse_times"].append(rmse_time)
best_results["best_crps_times"].append(crps_time)

print("best_results")
pprint.pprint(best_results)
# get the std and mean
stats = {key: {'mean': np.mean(value), 'std': np.std(value)} for key, value in best_results.items()}
print("final stats")
pprint.pprint(stats)



