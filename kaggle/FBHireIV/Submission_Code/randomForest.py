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
output=open('submission_newfeat_14.csv','w')

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
			     #if float(items[1])>0 or float(items[6])>0.0:
			     if float(items[1])>1 and float(items[6])!=0.0:
                                  self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
                                  #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6])/float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          self.labelMat.append(int(float(items[25])));
			          #self.dataMat.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)

         def loadTestSet(self,testfile):
                 for line in open(testfile,'r'):
			     items = line.strip().split(',');
			     outf.write(str(items[0])+',')
			     self.totalid.append(items[0])
			     #if float(items[1])>0 or float(items[6])!=0.0:
			     if float(items[1])>1 and float(items[6])!=0.0:
                                  self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
			          #self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          #self.testData.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
			          self.testid.append(items[0])
			          outf.write('nan'+'\n')
			     else:
                                  outf.write('0.0'+'\n')
		 self.testData = asarray(self.testData)

	 def test(self):
                 X, y = self.dataMat,self.labelMat
                 X_test = self.testData
                 clf_RFC = RandomForestClassifier(max_depth=8,min_samples_split=7, min_samples_leaf=8,n_estimators=10)
                 clf_RFC.fit(X, y);
                 y_pred = clf_RFC.predict(X_test);
                 y_predprob = clf_RFC.predict_proba(X_test);
                 output.write('bidder_id'+','+'prediction'+'\n')
                 for i in range(0,len(self.totalid)):
                     if self.totalid[i] in self.testid:
                         idx = self.testid.index(self.totalid[i])
                         output.write(str(self.testid[idx])+','+str(y_predprob[idx][1])+'\n')
                     else:
                         output.write(str(self.totalid[i])+','+str(0.0)+'\n')

                 
                 
                 
if __name__=='__main__':
             trainfile = 'train_feature_v0.csv'
             testfile = 'test_feature_v0.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
