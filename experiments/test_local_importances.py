import sys
import os
os.environ["NUMBA_DISABLE_JIT"] = "0"  # "1" to disable, "0" to enable
cwd = os.getcwd()
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.utils import *
from utils_reboot.datasets import *
from utils_reboot.plots import *


from model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Local Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: [EIF, EIF+, C_EIF+]')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]')
parser.add_argument('--scenario', type=int, default=1, help='scenario for training the model. Possible values: [1,2]')
parser.add_argument("--eta", type=float, default=1.5, help="eta hyperparameter of EIF+")
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs of Local Feature importance computation')

# Parse the arguments
args = parser.parse_args()

assert args.model in ["EIF+","C_EIF+","EIF"], "Model not recognized. Accepted values: ['EIF','EIF+','C_EIF+']"
assert args.interpretation in ["EXIFFI","EXIFFI+", "C_EXIFFI+"], "Interpretation not recognized"
if args.interpretation == "EXIFFI+":
    assert args.model=="EIF+", "EXIFFI+ can only be used with the EIF+ model"
if args.interpretation == "EXIFFI":
    assert args.model=="EIF", "EXIFFI can only be used with the EIF model"
if args.interpretation == "C_EXIFFI+":
    assert args.model=="C_EIF+", "C_EXIFFI+ can only be used with the C_EIF+ model"

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
pre_process = args.pre_process
model = args.model
interpretation = args.interpretation
scenario = args.scenario
eta = args.eta
n_runs = args.n_runs 

# Load dataset
dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../../datasets/data/')
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples
if dataset.shape[0] > 7500:
    dataset.downsample(max_samples=7500)

# If a dataset has lables (all the datasets except piade), the contamination is set to dataset.perc_outliers
if dataset.perc_outliers != 0:
    contamination = dataset.perc_outliers

# Split the dataset (scenario 2) for TEP dataset
if scenario == 2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,
                          contamination=0)

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("Preprocessing the dataset...")
    print("#"*50)
    dataset.pre_process()
else:
    print("#"*50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI" or interpretation == "RandomForest":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, eta=eta)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, eta=eta)
# For the moment EIF+ and C_EIF+ are the same model, modify here when we have the C implementation of ExtendedIsolationForest
elif model == "C_EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, eta=eta)

os.chdir('../')
cwd=os.getcwd()

print('#'*50)
print('LFI Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Estimators: {n_estimators}')
print(f'Contamination: {contamination}')
print(f'Interpretation Model: {interpretation}')
print(f'Scenario: {scenario}')
print(f'Number of runs: {n_runs}')
print('#'*50)

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
# path_plots = cwd +"/experiments/results/"+dataset.name+"/plots/lfi_imp_plots"
# if not os.path.exists(path_plots):
#     os.makedirs(path_plots)

#----------------- LOCAL IMPORTANCES -----------------#
# initialize local_importances paths
path_experiment = path_experiments + "/local_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
    

path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)

if pre_process:
    path_experiment_model_interpretation_trees = path_experiment_model + "/" + interpretation + f"/trees_{n_estimators}_pre_process"
else:
    path_experiment_model_interpretation_trees = path_experiment_model + "/" + interpretation + f"/trees_{n_estimators}"
    
if not os.path.exists(path_experiment_model_interpretation_trees):
    os.makedirs(path_experiment_model_interpretation_trees)

path_experiment_model_interpretation_cont = path_experiment_model_interpretation_trees + f"/cont_{int(contamination*100)}"
if not os.path.exists(path_experiment_model_interpretation_cont):
    os.makedirs(path_experiment_model_interpretation_cont)
    
path_experiment_model_interpretation_imp_mat = path_experiment_model_interpretation_cont + "/imp_mat"
if not os.path.exists(path_experiment_model_interpretation_imp_mat):
    os.makedirs(path_experiment_model_interpretation_imp_mat)

path_experiment_model_interpretation_imp_mat_scenario = path_experiment_model_interpretation_imp_mat + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_imp_mat_scenario):
    os.makedirs(path_experiment_model_interpretation_imp_mat_scenario)

path_experiment_model_interpretation_bars = path_experiment_model_interpretation_cont + "/bars"
if not os.path.exists(path_experiment_model_interpretation_bars):
    os.makedirs(path_experiment_model_interpretation_bars)

path_experiment_model_interpretation_bars_scenario = path_experiment_model_interpretation_bars + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_bars_scenario):
    os.makedirs(path_experiment_model_interpretation_bars_scenario)

path_experiment_model_interpretation_labels = path_experiment_model_interpretation_cont + "/labels"
if not os.path.exists(path_experiment_model_interpretation_labels):
    os.makedirs(path_experiment_model_interpretation_labels)

path_experiment_model_interpretation_labels_scenario = path_experiment_model_interpretation_labels + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_labels_scenario):
    os.makedirs(path_experiment_model_interpretation_labels_scenario)

#Compute local importances

full_importances,labels = experiment_local_importances(I, dataset, p=contamination, interpretation=interpretation, n_runs=n_runs)    
save_element(labels, path_experiment_model_interpretation_labels_scenario, filetype="npz")
save_element(full_importances, path_experiment_model_interpretation_imp_mat_scenario, filetype="csv.gz")


# Compute bars and save it in path_experiment_model_interpretation_bars_scenario
imp_path = get_most_recent_file(path_experiment_model_interpretation_imp_mat_scenario)
bars = compute_bars(dataset=dataset,importances_file=imp_path,filetype="csv.gz",model=model,interpretation=interpretation)
save_element(bars,path_experiment_model_interpretation_bars_scenario,filetype="csv.gz")

# plot global importances
# most_recent_file = get_most_recent_file(path_experiment_model_interpretation)
# bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation)
# score_plot(dataset, most_recent_file, plot_path=path_plots, show_plot=False, model=model, interpretation=interpretation)