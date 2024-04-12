import sys
import os
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
parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
parser.add_argument('--json_path', type=str, default=os.path.dirname(cwd)+'/datasets/data/', help='Path to the json file with the feature names')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = os.path.dirname(cwd)+args.dataset_path
json_path = args.json_path

dataset=Dataset(dataset_name,path=dataset_path,feature_names_filepath=json_path)
dataset.drop_duplicates()

#plt_data=compute_plt_data(local_imp_matrix_path)

local_imp_path=os.path.join(cwd,
                  'experiments','results',
                  dataset_name,'experiments',
                  'local_importances',
                  'EIF+','EXIFFI+')

local_imp_matrix_path=get_most_recent_file(local_imp_path)
local_imp_matrix=open_element(local_imp_matrix_path, filetype='npz')

#import ipdb; ipdb.set_trace()

bar_plot(dataset, 
         local_imp_matrix_path, 
         filetype="npz", 
         plot_path='experiments/results/piade_s2/plots/lfi_imp_plots', 
         f=min(dataset.shape[1],6),
         show_plot=True,
         save_image=False, 
         model='EIF+', 
         interpretation='EXIFFI+')
