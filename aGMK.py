'''Script containing functions and classes to compute the $alpha-Gaussian Mean Kernel between two objects modeled by a Gaussian distribution and to plug it in the kernel of SVM for supervised classification.
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from scipy import linalg
from model_objects import model_objects,GaussianObject
from MC_procedure import mc_procedure
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin



def aGMKernel(Ni,Nj,alpha,gamma):
    """
    Evaluation of the aGMK mean map kernel between Ni and Nj (GaussianObjet structure)
    
    Input:
        - Ni (GaussianObject)
        - Nj (GaussianObject)
        - gamma (float): gamma parameter
        - alpha (float): alpha parameter
    
    Return:
        - Kij (float): kernel evaluation between Ni and Nj using aGMK
        
    TODO: faster computation of Kij
    """
    
    #Dimension of data
    d = Ni.mu.size
    I = sp.eye(d)

    ##Normalisation
    deltaMean = (Ni.mu-Nj.mu).reshape(d,)
    SigmaSum = alpha * (Ni.Sigma+Nj.Sigma) + I/gamma
    Kij = (linalg.det(2*gamma*alpha * Ni.Sigma + I) * linalg.det(2*gamma*alpha * Nj.Sigma + I))**0.25
    Kij *= sp.exp(-0.5*sp.dot(deltaMean.T,linalg.solve(SigmaSum,deltaMean)))
    Kij /= sp.sqrt(linalg.det(SigmaSum*gamma))   
    
    return Kij


class aGMK(BaseEstimator,TransformerMixin):
    """Pipeline for the computation of alpha-Gaussian Mean Kernel.
    """
    def __init__(self,gamma=10**1,alpha=1):
        super(aGMK,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def transform(self,G):
        """Compute the pairwise aGMK kernel function between each sample of G.
        
        Input:
            -G (list): list of GaussianObject objects
        Ouput:
            - kernel_matrix (array): matrix of pairwise aGMK kernel
        """

        n = len(self.G_train_)
        nt = len(G)
        #Ks = sp.zeros((n,1))
        kernel_matrix = sp.zeros((nt,n))
        
#         for j in range(n):
#             Ks[j] = sp.sqrt(aGMKernel(self.G_train_[j],self.G_train_[j],self.alpha,self.gamma))
#                 
#         for i in range(nt):
#             Kts = sp.sqrt(aGMKernel(G[i],G[i],self.alpha,self.gamma))
#             for j in range(n):
#                 kernel_matrix[i,j] = aGMKernel(G[i],self.G_train_[j],self.alpha,self.gamma)/Kts/Ks[j]
        
        for i in range (nt):
            for j in range(n):
                kernel_matrix[i,j] = aGMKernel(G[i],self.G_train_[j],self.alpha, self.gamma)
        
           
        return kernel_matrix

    def fit(self, G, y=None, **fit_params):
        self.G_train_ = G
        return self    


def process_aGMK_MC_procedure(X,ID,Y,grid_search,acc="F1Mean",REP=30,test_prop=0.25,n_folds=3,n_jobs=-1,RS=0,VERBOSE=False):
    """Process a SVM supervised classification with a Monte Carlo procedure using the $\alpha$-Gaussian Mean Kernel.
    
    Input:
        - X (array): matrix (N x d) of objects' pixels (N is the total number of pixels, d is the number of variables).
        - ID (array): vector (N x 1) of corresponding IDs of pixels allowing the identification of the objects
        - Y (array): vector (N x 1) of labels
        - grid_search (dict): grid search parameters of for the cross-validation {'C': [C], 'GAMMA': [GAMMA], 'ALPHA': [ALPHA]}
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
    #Model the objects by a Gaussian distribution (mean vector and covariance matrix)
    G = [GaussianObject(o.x) for o in O]
    
    #Parameters of the classification   
    grid_search_ = dict([
        ('svm__kernel',['precomputed']),
        ('svm__C',grid_search['C']),
        ('Kernel__alpha', grid_search['ALPHA']),
        ('Kernel__gamma',grid_search['GAMMA']),
        ])
    
    kernel = aGMK()       
    #Process the classifications in parallel
    results = Parallel(n_jobs=n_jobs,verbose=False)(delayed(mc_procedure)(i,G,labels,kernel,grid_search_,acc,test_prop,n_folds,RS,VERBOSE) for i in range(REP))
    
    #Reshape the results
    CM = [results[i][0] for i in range(REP)]
    YP = [results[i][1] for i in range(REP)]
    best_params = [results[i][2] for i in range(REP)]

    return CM, YP, best_params
