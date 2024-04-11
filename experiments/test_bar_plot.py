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

# print('#'*50)
# print(f'I am in path → {cwd}')
# print('#'*50)
local_imp_path=os.path.join(cwd,
                  'experiments','results',
                  'piade_s2','experiments',
                  'local_importances',
                  'EIF+','EXIFFI+')
# print(f'Path of local importances files → {local_imp_path}')

local_imp_matrix_path=get_most_recent_file(local_imp_path)
local_imp_matrix=open_element(local_imp_matrix_path, filetype='npz')

datapath=os.path.join(os.path.dirname(cwd),'datasets','data','PIADE/')
dataset=Dataset('piade_s2_all_alarms',path=datapath)
dataset.drop_duplicates()

import ipdb; ipdb.set_trace()

#plt_data=compute_plt_data(local_imp_matrix_path)

bar_plot(dataset, 
         local_imp_matrix_path, 
         filetype="npz", 
         plot_path='experiments/results/piade_s2/plots/lfi_imp_plots', 
         f=min(dataset.shape[1],6),
         show_plot=True,
         save_image=False, 
         model='EIF+', 
         interpretation='EXIFFI+')
