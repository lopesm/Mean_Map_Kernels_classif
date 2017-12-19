'''
Script to perform a SVM supervised classification using Empirical Mean Kernel and following a Monte Carlo procedure.
'''

import argparse
import scipy as sp
from EMK import process_EMK_MC_procedure


##Parser initialization
parser = argparse.ArgumentParser(description="Run script for performing SVM supervised classification using GMK and following a Monte Carlo procedure.")
parser.add_argument("-X",action='store',dest='X',default='X_ms_2014_pratiques.npy', help='X of dataset pixels',type=str)
parser.add_argument("-ID",action='store',dest='ID',default='ID_2014_pratiques.npy', help='ID of pixels',type=str)
parser.add_argument("-Y",action='store',dest='Y',default='Y_2014_pratiques.npy', help='Labels of pixels',type=str)
parser.add_argument('-smin', action='store', dest='smin', default=-18, help='Lowest $\sigma$ value to test in the interval [2**smin,2**smax]',type=int)
parser.add_argument('-smax', action='store', dest='smax', default=-10, help='Highest $\sigma$ value to test in the interval [2**smin,2**smax]',type=int)
parser.add_argument('-C', action='store', dest='C', default=[10], help='C values to test (for SVM)',type=list)
parser.add_argument('-acc', action='store', dest='acc', default="F1Mean", help='Accuracy parameter for training the model',type=str)
parser.add_argument('-REP', action='store', dest='REP', default=30, help='Number of iterations of the MC procedure',type=int)
parser.add_argument('-test_prop', action='store', dest='test_prop', default=0.25, help='Proportion of the dataset to use for testing subset during the CV',type=float)
parser.add_argument('-n_folds', action='store', dest='n_folds', default=3, help='Number of folds for the stratified k-folds CV',type=int)
parser.add_argument('-n_jobs', action='store', dest='n_jobs', default=2, help='Number of jobs to run in parallel',type=int)
parser.add_argument('-VERBOSE', action='store_true', dest='VERBOSE', default=False, help='VERBOSE')
args = parser.parse_args()



##Load data
X = sp.load(args.X)
ID = sp.load(args.ID)
Y = sp.load(args.Y)
# These files were obtained from the following function:
# raster_name = 'serie_formosat_smoothed_concat_2014.tif'
# mask_grasslands_ID = 'echantillon_id_modcond_3classes.tif'
# mask_grasslands_modcond = 'modcon_3classes.tif'
# X, ID = get_samples_from_roi(raster_name,mask_grasslands_ID)
#  Y = get_samples_from_roi(raster_name,mask_grasslands_modcond)[1]


##Classification and Kernel parameters
SIGMA = 2.0**sp.arange(args.smin,args.smax+1,1)
grid_search = {'C': args.C, 'SIGMA': SIGMA}
acc = args.acc
REP = args.REP
test_prop = args.test_prop
n_jobs = args.n_jobs
VERBOSE = args.VERBOSE
n_folds = args.n_folds


###Process classifications
CM, YP, best_params = process_EMK_MC_procedure(X,ID,Y,grid_search,acc=acc,REP=REP,test_prop=test_prop,n_folds=n_folds,n_jobs=n_jobs,VERBOSE=VERBOSE)
for i in range(len(CM)):
    print CM[i].get_F1Mean()
    print best_params[i]

del X, Y

