import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from append_to_path import append_dirname
append_dirname('ExIFFI_Industrial_Test')
from ExIFFI_original.utils_reboot.datasets import Dataset
from ExIFFI_original.utils_reboot.utils import *
from ExIFFI_original.utils_reboot.experiments import *
from ExIFFI_original.utils_reboot.plots import score_plot

d=Dataset('TEP_ACME',path='../../datasets/data/TEP/',feature_names_filepath='../../datasets/data/')

cwd=os.path.dirname(os.getcwd())
dirpath_imp=os.path.join(cwd,'experiments','results',
                  'TEP_ACME','experiments','global_importances','EIF+',
                  'EXIFFI+','imp_mat','scenario_2')
path_plots=os.path.join(cwd,'ExIFFI_original','experiments','results',
                  'TEP_ACME','plots','score_plots')
path_imp=get_most_recent_file(dirpath_imp)
imp_mat=open_element(path_imp,filetype="csv.gz")

score_plot(d,
           path_imp,
           plot_path=path_plots,
           show_plot=True,
           model="EIF+",
           interpretation="EXIFFI+",
           scenario=2,
           save_image=True)
