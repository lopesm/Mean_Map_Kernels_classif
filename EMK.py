'''Script containing functions and classes to compute the Empirical Mean Kernel between two objects and to plug it in the kernel of SVM for supervised classification.
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from model_objects import model_objects
from MC_procedure import mc_procedure
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel



def EMKernel(Oi,Oj,sigma):
    """
    Evaluation of the EMK between two objects.
    
    Input:
        - Oi (ObjectProperties)
        - Oj (ObjectProperties)
        - sigma (float): sigma parameter
    
    Return:
        - Kij (float): EMK kernel evaluation between Oi and Oj using GMK
    """

    Kij = rbf_kernel(Oi.x,Oj.x,gamma=sigma).mean()
    
    return Kij


class EMK(BaseEstimator,TransformerMixin):
    """Pipeline for the computation of Empirical Mean Kernel.
    """
    def __init__(self,sigma=2**(-10)):
        super(EMK,self).__init__()
        self.sigma = sigma
        
    def transform(self,G):
        """Compute the pairwise EMK kernel function between each sample of G.
        
        Input:
            -G (list): list of ObjectProperties objects
        Ouput:
            - kernel_matrix (array): matrix of pairwise EMK kernel
        """

        n = len(self.G_train_)
        nt = len(G)
        kernel_matrix = sp.zeros((nt,n))
        
        for i in range (nt):
            for j in range(n):
                kernel_matrix[i,j] = EMKernel(G[i],self.G_train_[j],self.sigma)
          
        return kernel_matrix
    

    def fit(self, G, y=None, **fit_params):
        self.G_train_ = G
        return self    


def process_EMK_MC_procedure(X,ID,Y,grid_search,acc="F1Mean",REP=30,test_prop=0.25,n_folds=3,n_jobs=-1,RS=0,VERBOSE=False):
    """Process a SVM supervised classification with a Monte Carlo procedure using the Empirical Mean Kernel.
    
    Input:
        - X (array): matrix (N x d) of objects' pixels (N is the total number of pixels, d is the number of variables).
        - ID (array): vector (N x 1) of corresponding IDs of pixels allowing the identification of the objects
        - Y (array): vector (N x 1) of labels
        - grid_search (dict): grid search parameters of for the cross-validation {'C': [C], 'SIGMA': [SIGMA]}
        - acc (str): accuracy parameter for the cross-validation (default "F1Mean")
        - REP (int): number of repetitions of the Monte Carlo procedure (default: 30)
        - test_prop (float): proportion of the dataset to use for testing subset during the n-fold cross-validation (default: 20.25)
        -n_folds (int): number of folds for the stratified n-folds cross-validation (default : 3)
        - n_jobs (int): number of jobs to perform in parallel (default : -1)
        - RS (int) : random seed used for the stratified k-fold CV
        - VERBOSE (bool): verbose (default: False)
        - 
    Output:
        - CM (list) : list of objects of class ConfusionMatrix containing the confusion matrices resulting from the REP number of classifications
        - YP (list): list of vectors (N x 1) of predicted labels over the number of repetitions (REP)
        - best_params (list) : list of combination of parameters cross-validated over the number of repetitions
    """
    
    #Get the objects and their labels
    O, labels = model_objects(X, ID, Y)

    
    #Parameters of the classification   
    grid_search_ = dict([
        ('svm__kernel',['precomputed']),
        ('svm__C',grid_search['C']),
        ('Kernel__sigma',grid_search['SIGMA']),
        ])
    
    kernel = EMK()       
    #Process the classifications in parallel
    results = Parallel(n_jobs=n_jobs,verbose=False)(delayed(mc_procedure)(i,O,labels,kernel,grid_search_,acc,test_prop,n_folds,RS,VERBOSE) for i in range(REP))
    
    #Reshape the results
    CM = [results[i][0] for i in range(REP)]
    YP = [results[i][1] for i in range(REP)]
    best_params = [results[i][2] for i in range(REP)]

    return CM, YP, best_params
