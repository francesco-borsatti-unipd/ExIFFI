import os
import sys
sys.path.append('../')
cwd=os.getcwd()
from append_to_path import append_dirname
append_dirname("ExIFFI_Industrial_Test")
import pickle
from utils_reboot.utils import *
from utils_reboot.datasets import *
import argparse
from ExIFFI_C.model_reboot.EIF_reboot import ExtendedIsolationForest
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score


# Create the argument parser
parser = argparse.ArgumentParser(description='Test Performance Metrics')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=200, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=npt.NDArray, default=np.linspace(0.0,0.1,10), help='Global feature importances parameter: contamination')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--downsample',type=bool,default=False, help='If set, downsample the dataset if it has more than 7500 samples')
parser.add_argument('--return_perf', action='store_true', help='If set return the model performances results')

def get_precision_file(dataset,model,scenario):    
    path=os.path.join(cwd+"/results/",dataset.name,'experiments','metrics',model,f'scenario_{str(scenario)}')
    file_path=get_most_recent_file(path)
    results=open_element(file_path)
    return results

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
model = args.model
interpretation = args.interpretation
scenario = args.scenario
pre_process = args.pre_process
downsample = args.downsample
return_perf = args.return_perf

dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()


print('#'*50)
print('Performance Metrics Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Interpretation: {interpretation}')
print(f'Scenario: {scenario}')
print('#'*50)

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if (dataset.shape[0]>7500) and downsample:
    dataset.downsample(max_samples=7500)

if dataset.perc_outliers != 0:
    contamination = dataset.perc_outliers

if scenario==2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("\n\nPreprocessing the dataset...")
    print("#"*50)
    dataset.pre_process()
else:
    print("#"*50)
    print("\n\nDataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)

#import ipdb; ipdb.set_trace()

if model == "EIF+":
    I = ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF":
    I = ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif (model == "IF"):
    I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

if return_perf:
    print("#"*50)
    print(f'Performance values for {dataset.name} {model} scenario {str(scenario)}')
    print(get_precision_file(dataset,model,scenario).T)
    print("#"*50)

os.chdir('../')
cwd=os.getcwd()

filename = cwd + "/utils_reboot/EIF+_vs_ACME-AD.pickle"
with open(filename, "rb") as file:
    dict_time = pickle.load(file)

print('#'*50)
print(f'Fit time for {model} {dataset.name} scenario {str(scenario)}: {np.round(np.mean(dict_time["fit"][model][dataset.name]),3)}')
print(f'Predict time for {model} {dataset.name} scenario {str(scenario)}: {np.round(np.mean(dict_time["predict"][model][dataset.name]),3)}')
print(f'Importances time for {model} {dataset.name} scenario {str(scenario)}: {dict_time["importances"][interpretation][dataset.name]}')
print('#'*50)