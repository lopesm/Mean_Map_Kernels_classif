'''Script containing classes for object-oriented methods.
'''

import scipy as sp


class ObjectProperties:
    """Class gathering all the properties of the objects (matrix of pixels, ID, label and coordinates of pixels).
    """
    def __init__(self,x,Id,y,coord):
        self.x = x #(array) Matrix of pixels
        self.id = Id #(int) ID of the object
        self.y = y #(int or float): label of the object
        self.coord = coord #(array) : coordinates of the pixels
        

class GaussianObject:
    """Class modeling the object X by a Gaussian distribution, i.e., represented by its mean vector and covariance matrix.
    """
    def __init__(self,X):
        #self.X = X
        self.mu = sp.mean(X,0) #Mean vector
        self.Sigma = sp.cov(X,rowvar = 0) #Covariance matrix


def model_objects(X,ID,Y):
    """Get the objects from their ID.
    
    Input:
        - X (array): matrix (N x d) of objects' pixels (N is the total number of pixels, d is the number of variables).
        - ID (array): vector (N x 1) of corresponding IDs of pixels allowing the identification of the objects
        - Y (array): vector (N x 1) of labels
    
    Return:
        - G (list): list of N objects of class ObjectProperties
        - labels (list): list of associated labels
    """
    
    list_ID = sp.unique(ID)
    list_ID = sp.sort(list_ID)
    G, labels = [], []
    for id_ in list_ID:
        t = sp.where(ID==id_)[0]
        x = X[t,:]
        label = Y[t][0][0]
        labels.append(label)
        G.append(ObjectProperties(x,id_,label,t))
    return G, labels


