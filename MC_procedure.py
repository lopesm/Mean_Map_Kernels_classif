'''Script containing functions to process a SVM supervised classification at the object scale with customized kernel function and using a Monte Carlo procedure.
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from confusion_matrix import ConfusionMatrix



class MC_SVM:
    """Monte Carlo procedure for supervised classification using SVM and customized kernel function.
    """
    def __init__(self,kernel,n_folds=5):
        """
        Input:
            - kernel (class): estimator of the kernel function        
            - n_folds (int): number of folds for the stratified k-folds CV
        """
                
        self.n_folds=n_folds
        self.pipe = Pipeline([('Kernel', kernel),('svm', SVC()),])

                        
    def fit_predict(self,Gtr,Ytr,Gt,Yt,grid_search,acc_param="F1Mean",RS=0,VERBOSE=False):
        """Learn the model on training dataset and predict on the testing dataset.
        
        Input:
            - Gtr (array): training subset
            - Ytr (array): true labels of training subset
            - Gt (array): testing subset
            - Yt (array): true labels of testing subset
            - grid_search (dict): grid search for the CV
            - acc (str): accuracy parameter for the cross-validation (default "F1Mean", otherwise it will be the OA)
            - RS (int) : random seed used for the stratified k-fold CV
            - VERBOSE (bool): verbose (default: False)
        
        Return:
            - confMatrix (object ConfusionMatrix): confusion matrix issued from the classification
            - yp (array): vector of predicted labels of the testing subset
            - grid.best_params_ (dict): combination of parameters that gave the best results during the CV
        
        TODO: implement own scoring parameter
        """
                    
        if acc_param== "F1Mean":
            score = 'f1_macro'
        else:
            score= 'accuracy'      
        
        ## Initialization of the stratifield K-CV
        cv = StratifiedKFold(Ytr,n_folds=self.n_folds,random_state=RS)

        #Implementation of a fit and a predict methods with parameters from grid search
        grid = GridSearchCV(self.pipe, param_grid=grid_search, scoring=score,verbose=VERBOSE, cv=cv,n_jobs=3)
        grid.fit(Gtr,Ytr)
        
        model = grid.best_estimator_
        if VERBOSE:
            print grid.best_score_
        
        #Learn model
        model.fit(Gtr,Ytr) #could use refit in version 0.19 of sklearn
        #Predict
        yp = model.predict(Gt)

        #Compute confusion matrix
        confMatrix = ConfusionMatrix()
        confMatrix.compute_confusion_matrix(yp, Yt) 
        
        return confMatrix, yp, grid.best_params_
    
    def split_dataset(self,i,G,Y,grid_search,acc,test_prop,RS,VERBOSE):
        """One iteration of the MC procedure (split dataset into training and testing subsets, fit and predict).
        
        Input:
            - i (int): number of the repetition for random split of the dataset
            - G (array or list): list of objects
            - Y (array): vector of true labels
            - grid_search (dict): grid search for the CV
            - acc (str): accuracy parameter for the cross-validation (default "F1Mean")
            - test_prop (float): proportion of the dataset to use for testing subset during the n-fold cross-validation (default: 20.25)
            - RS (int) : random seed used for the stratified k-fold CV
            - VERBOSE (bool): verbose (default: False)
        
        Return:
            - confMatrix (object ConfusionMatrix): confusion matrix issued from the classification
            - yp (array): vector of predicted labels of the testing subset
            - best_params (dict): combination of parameters that gave the best results during the CV
        """
        
        #Divide the dataset into training and testing subsets
        Gtr,Gt,Ytr,Yt = train_test_split(G,Y,test_size=test_prop,random_state=i,stratify=Y)
    
        #Predict the labels of the testing subset
        confMatrix, yp, best_params = self.fit_predict(Gtr,Ytr,Gt,Yt,grid_search,acc,RS=RS,VERBOSE=VERBOSE)  
    
        return confMatrix, yp, best_params
    
def mc_procedure(i,G,Y,kernel,grid_search,acc,test_prop,n_folds=3, RS=0, VERBOSE=False):
    """One iteration of the Monte Carlo procedure for supervised classification.
    (This function was designed to enable the parallel run of the iterations of the MC procedure)
    
    Input:
        - i (int): number of the repetition for random split of the dataset
        - G (array or list): list of objects
        - Y (array): vector of true labels
        - kernel (class): estimator of the kernel function        
        - grid_search (dict): grid search for the CV
        - acc (str): accuracy parameter for the cross-validation (default "F1Mean")
        - test_prop (float): proportion of the dataset to use for testing subset during the k-folds cross-validation (default: 20.25)
        - n_folds (int): number of folds for the stratified k-folds CV
        - RS (int) : random seed used for the stratified k-fold CV
        - VERBOSE (bool): verbose (default: False)
        
    Return:
        -results (tuple): results of the classification (confusion matrix, predicted labels and best parameters)
    """
    classif = MC_SVM(kernel,n_folds)

    #Process the classifications
    results = classif.split_dataset(i,G,Y,grid_search,acc,test_prop,RS, VERBOSE )
    
    return results
