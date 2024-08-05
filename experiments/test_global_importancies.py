import sys
import os
cwd = os.getcwd()
sys.path.append("..")
from collections import namedtuple
from append_to_path import append_dirname
append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.experiments import *
from utils_reboot.utils import *
from utils_reboot.datasets import *
from utils_reboot.plots import *


from ExIFFI_C.model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument("--scaler_type", type=int, default=1, help="Scaler to use: 1 for StandardScaler, 2 for MinMaxScaler")
parser.add_argument('--model', type=str, default="EIF", help='Model to use: [EIF+, C_EIF+]')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: [EXIFFI+, C_EXIFFI+]')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--downsample',type=bool,default=False, help='If set, downsample the dataset if it has more than 7500 samples')

# Parse the arguments
args = parser.parse_args()

assert args.model in ["EIF+","C_EIF+","IF","EIF+_centroid","EIF+_distrib_split","EIF+_centroid_split"], "Model not recognized. Accepted values: ['EIF+','C_EIF+']"
assert args.interpretation in ["EXIFFI+", "C_EXIFFI+","DIFFI"], "Interpretation not recognized"
if args.interpretation == "EXIFFI+":
    assert args.model in ["EIF+", "EIF+_centroid","EIF+_distrib_split","EIF+_centroid_split"], "EXIFFI+ can only be used with the EIF+ model"
if args.interpretation == "C_EXIFFI+":
    assert args.model=="C_EIF+", "C_EXIFFI+ can only be used with the C_EIF+ model"
"EIF+_centroid_split"

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs
pre_process = args.pre_process
model = args.model
interpretation = args.interpretation
scenario = args.scenario
downsample = args.downsample
scaler_type = args.scaler_type

dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../../datasets/data/')
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples
if dataset.shape[0] > 7500 and downsample:
    dataset.downsample(max_samples=7500)

# If a dataset has lables (all the datasets except piade), the contamination is set to dataset.perc_outliers
if dataset.perc_outliers != 0:
    contamination = dataset.perc_outliers

if scenario==2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("Preprocessing the dataset...")
    print("#"*50)
    dataset.pre_process(scaler_type=scaler_type)
else:
    print("#"*50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)

#import ipdb; ipdb.set_trace()

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI" or interpretation == "RandomForest":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
# For the moment EIF+ and C_EIF+ are the same model, modify here when we have the C implementation of ExtendedIsolationForest
elif model == "C_EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+_centroid":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, use_centroid_importance=True)
elif model == "EIF+_distrib_split":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, use_distribution_aware_splits=True)
elif model == "EIF+_centroid_split":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, use_centroid_importance=True, use_distribution_aware_splits=True)

os.chdir('../')
cwd=os.getcwd()

print('#'*50)
print('GFI Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Estimators: {n_estimators}')
print(f'Contamination: {contamination}')
print(f'Interpretation Model: {interpretation}')
print(f'Scenario: {scenario}')
print(f'Scaler: {scaler_type}')
print('#'*50)

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots/score_plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

#----------------- GLOBAL IMPORTANCES -----------------#
# initialize global_importances paths
path_experiment = path_experiments + "/global_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
    

path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)
    
path_experiment_model_interpretation_imp_mat = path_experiment_model + "/" + interpretation + "/imp_mat"
if not os.path.exists(path_experiment_model_interpretation_imp_mat):
    os.makedirs(path_experiment_model_interpretation_imp_mat)

path_experiment_model_interpretation_imp_mat_scenario = path_experiment_model_interpretation_imp_mat + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_imp_mat_scenario):
    os.makedirs(path_experiment_model_interpretation_imp_mat_scenario)

path_experiment_model_interpretation_bars = path_experiment_model + "/" + interpretation + "/bars"
if not os.path.exists(path_experiment_model_interpretation_bars):
    os.makedirs(path_experiment_model_interpretation_bars)

path_experiment_model_interpretation_bars_scenario = path_experiment_model_interpretation_bars + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_bars_scenario):
    os.makedirs(path_experiment_model_interpretation_bars_scenario)

    
##################################
##################################
##################################
# imp_path = get_most_recent_file(path_experiment_model_interpretation_imp_mat_scenario)
# path_plots = cwd +"/experiments/results/"+dataset.name+"/plots/GFI_plots"
# bar_plot(dataset, imp_path, filetype="csv.gz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
# quit()
##################################
##################################
##################################


#Compute global importances
#full_importances,_ = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation)    
full_importances = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation)
save_element(full_importances, path_experiment_model_interpretation_imp_mat_scenario, filetype="csv.gz")

# Compute bars and save it in path_experiment_model_interpretation_bars_scenario
imp_path = get_most_recent_file(path_experiment_model_interpretation_imp_mat_scenario)
bars = compute_bars(dataset=dataset,importances_file=imp_path,filetype="csv.gz",model=model,interpretation=interpretation)
save_element(bars,path_experiment_model_interpretation_bars_scenario,filetype="csv.gz")

# plot global importances
imp_path = get_most_recent_file(path_experiment_model_interpretation_imp_mat_scenario)
#bar_plot(dataset, imp_path, filetype="npz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
score_plot(dataset, imp_path, plot_path=path_plots, show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
