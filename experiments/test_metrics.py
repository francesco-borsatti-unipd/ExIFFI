import sys
import os
cwd = os.getcwd()
#os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple
from append_to_path import append_dirname
append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *

from ExIFFI_C.model_reboot.EIF_reboot import ExtendedIsolationForest
from model_reboot.EIF_reboot import IsolationForest as EIF_IsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Performance Metrics')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')
parser.add_argument('--pre_process',type=bool, default=False, help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--downsample',type=bool,default=False, help='If set, downsample the dataset if it has more than 7500 samples')
parser.add_argument('--compute_GFI', type=bool, default=False, help='If set compute the Feature Importances')
parser.add_argument('--compute_perf', action='store_true', help='If set compute the model performances')

# Parse the arguments
args = parser.parse_args()

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
compute_GFI = args.compute_GFI
compute_perf = args.compute_perf

#import ipdb; ipdb.set_trace()

# Load the dataset
dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if (dataset.shape[0]>7500) and downsample:
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
    dataset.pre_process()
elif scenario==2 and not pre_process:
    print("#"*50)
    print("Dataset not preprocessed")
    #dataset.initialize_train()
    dataset.initialize_test()
    print("#"*50)
elif scenario==1 and not pre_process:
    print("#"*50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)


assert model in ["IF","sklearn_IF","EIF","EIF+"], "Evaluation Model not recognized"

if model == "sklearn_IF":
    I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "IF":
    I=EIF_IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

os.chdir('../')
cwd=os.getcwd()

filename = cwd + "/utils_reboot/EIF+_vs_ACME-AD.pickle"

if not os.path.exists(filename):
    dict_time = {"fit":{"EIF+":{},"ACME-AD":{}}, 
            "predict":{"EIF+":{},"ACME-AD":{}},
            "importances":{"EXIFFI+":{},"ACME-AD":{}}}
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
               
with open(filename, "rb") as file:
    dict_time = pickle.load(file)

print('#'*50)
print('Performance Metrics Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Interpretation: {interpretation}')
print(f'Estimators: {n_estimators}')
print(f'Contamination: {contamination}')
print(f'Scenario: {scenario}')
print(f'Number of runs: {n_runs}')
print('#'*50)

# Fit the model
start_time = time.time()
I.fit(dataset.X_train)
fit_time = time.time() - start_time
try:
    dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
except:
    print('Model not recognized: creating a new key in the dict_time for the new model')
    dict_time["fit"].setdefault(I.name, {}).setdefault(dataset.name, []).append(fit_time)

start_time = time.time()
score=I.predict(dataset.X_test)
y_pred=I._predict(dataset.X_test,p=contamination)
predict_time = time.time() - start_time
try:
    dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
except:
    print('Model not recognized: creating a new key in the dict_time for the new model')
    dict_time["predict"].setdefault(I.name, {}).setdefault(dataset.name, []).append(predict_time)

if compute_GFI:
    print('Computing Feature Importances...')
    print('#'*50)
    anomalies=dataset.X_test[np.where(y_pred==1)[0]]
    start_time = time.time()
    importances=I.local_importances(anomalies)
    importances_time = time.time() - start_time
    try:
        dict_time["importances"][interpretation].setdefault(dataset.name, []).append(importances_time)
    except:
        print('Model not recognized: creating a new key in the dict_time for the new model')
        dict_time["importances"].setdefault(interpretation, {}).setdefault(dataset.name, []).append(importances_time)

with open(filename, "wb") as file:
    pickle.dump(dict_time, file)

if compute_perf:
    # Compute the performance metrics using the performance function from utils_reboot.utils
    print('Computing performance metrics...')
    print('#'*50)
    performance_metrics,path = performance(y_pred=y_pred, y_true=dataset.y_test, score=score, I=I, model_name=I.name, dataset=dataset,contamination=dataset.perc_outliers, path=cwd, scenario=scenario, downsample=downsample, n_runs=n_runs)
    print('Performance metrics computed and saved in:', path)
