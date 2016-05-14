from numpy import *
import csv
import matplotlib.pyplot as plt
import scipy as sp

class SoftmaxRegression:
    def __init__(self):
		self.dataMat = []
		self.labelMat = []

    def loadDataSet(self,trainfile):
            for line in open(trainfile,'r'):
                     items = line.strip().split(',')
                     print items[0]
                     #self.dataMat.append([string(items[1])])
                     
          
if __name__=='__main__':
   trainfile = 'train_00.csv'
   bidsfile = 'bids.csv'
   #testfile = 'test.csv' 
   myclassification = SoftmaxRegression()
   myclassification.loadDataSet(trainfile)
