import matplotlib.pyplot as plt
import scipy as sp

from sklearn import *
from sklearn import cross_validation,linear_model
from numpy import *
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss, classification_report, accuracy_score, log_loss, roc_curve, auc, roc_auc_score

outf=open('submit_RF.csv','w')
output=open('submission_manual00.csv','w')

class SupportVectorMachine:
         def __init__(self):
                 self.dataMat = []
                 self.labelMat = []
                 self.testData = []
                 self.testid = []
                 self.totalid = []

         def loadDataSet(self,trainfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     if float(items[1])>3 and float(items[7])!=0.0:
			          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
                                  #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6])/float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          self.labelMat.append(int(float(items[12])));
			          #self.dataMat.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)

         def loadTestSet(self,testfile):
                 for line in open(testfile,'r'):
			     items = line.strip().split(',');
			     outf.write(str(items[0])+',')
			     self.totalid.append(items[0])
			     self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			     #self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          #self.testData.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])]
			    
		 self.testData = asarray(self.testData)

	 def test(self):
                 iris = datasets.load_iris()
                 X, y = self.dataMat,self.labelMat
                 X_test = self.testData
                 output.write('bidder_id'+','+'prediction'+'\n')
                 for i in range(0,len(X_test)):
                     if X_test[i,6]!=0.0:
                         if X_test[i,2]/X_test[i,1]>5 and X_test[i,3]/X_test[i,2]<35:
                              output.write(str(self.totalid[i])+','+str(0.0)+'\n')
                         elif X_test[i,5]>3000:
                              output.write(str(self.totalid[i])+','+str(0.0)+'\n')
                         elif X_test[i,5]/X_test[i,6]>0.08:
                              output.write(str(self.totalid[i])+','+str(0.0)+'\n')
                         else:
                              output.write(str(self.totalid[i])+','+str(nan)+'\n')
                     else:
                         output.write(str(self.totalid[i])+','+str(0.0)+'\n')

                 
                 
                 
if __name__=='__main__':
             trainfile = 'train_reset.csv'
             testfile = 'test_reset.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
