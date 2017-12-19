"""Function to compute the confusion matrix of a supervised classification and to extract metrics from the matrix.
"""
import scipy as sp
from sklearn.metrics import confusion_matrix as confusion_matrix_sk

## Confusion matrix class
class ConfusionMatrix(object):
    def __init__(self):
        self.confusionMatrix = None
        self.n = None

    def compute_confusion_matrix(self,yp,yr):
        """
            Compute the confusion matrix
            Inputs:
                yp: predicted labels
                yr: reference labels
        """
        # Initialization
        self.n                = yp.size
        self.confusionMatrix  = confusion_matrix_sk(yr,yp)

    def get_OA(self):
        """
            Compute overall accuracy
        """
        self.n = sp.sum(self.confusionMatrix)
        return sp.sum(sp.diag(self.confusionMatrix))/float(self.n)

    def get_kappa(self):
        """
            Compute Kappa
        """
        nl = sp.sum(self.confusionMatrix,axis=1)
        nc = sp.sum(self.confusionMatrix,axis=0)
        OA = sp.sum(sp.diag(self.confusionMatrix))/float(self.n)

        return ((self.n**2)*OA - sp.sum(nc*nl))/(self.n**2-sp.sum(nc*nl))

    def get_F1Mean(self):
        """
            Compute F1 Mean
        """
        nl = sp.sum(self.confusionMatrix,axis=1,dtype=float)
        nc = sp.sum(self.confusionMatrix,axis=0,dtype=float)
        return 2*sp.mean( sp.divide( sp.diag(self.confusionMatrix), (nl + nc)) )

