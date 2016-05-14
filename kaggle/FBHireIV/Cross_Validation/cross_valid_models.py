import matplotlib.pyplot as plt
import scipy as sp

from sklearn import *
from sklearn import cross_validation,linear_model
from numpy import *
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from nolearn.dbn import DBN

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import cross_val_score,ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss, classification_report, accuracy_score, log_loss, roc_curve, auc, roc_auc_score

class SupportVectorMachine:
         def __init__(self):
                 self.dataMat = []
                 self.labelMat = []
                 self.testData = []

         def loadDataSet(self,trainfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     if float(items[1])>1 and float(items[7])!=0.0:
			          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
                      #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
                                  #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[7])/float(items[6])])
			          #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16])])
			          self.labelMat.append(int(float(items[25])));
			          #self.dataMat.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)

         def loadTestSet(self,testfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     #if float(items[1])>2 and float(items[7])!=0.0:
                             if float(items[7])>=0.0:
			          self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          #self.testData.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.testData = asarray(self.testData)
		          

	 def test(self):
                 #iris = datasets.load_iris()
                 #X, y = iris.data, iris.target
                 X, y = self.dataMat,self.labelMat
                 X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.6, random_state=12)
                 #clf = RandomForestClassifier(max_depth=6,min_samples_split=9,min_samples_leaf=15,n_estimators=5)
                 #clf = DBN([X.shape[1], 24, 2],scales=0.5,learn_rates=0.02,learn_rate_decays = 0.95, learn_rate_minimums =0.001,epochs=500,l2_costs = 0.02*0.031, dropouts=0.2,verbose=0)
                 #cvnum = ShuffleSplit(2013,n_iter=10,test_size=0.6,train_size=0.4,random_state=0)
                 for C_val in range(1,20):
                     print "********************************************************"
                     print "SVM linear C_val=",C_val
                     clf = SVC(kernel='linear', C=C_val, probability=True)
                     clf.fit(X_train, y_train);
                     scores = cross_val_score(clf,X,y,cv=3,scoring='roc_auc')
                     y_pred = clf.predict(X_test);
                     y_predprob = clf.predict_proba(X_test);
                     prf=precision_recall_fscore_support(y_test, y_pred, average='binary')
                     print ("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
                     print  classification_report(y_test,y_pred)
                     print 'The accuracy is: ', accuracy_score(y_test,y_pred)
                     print 'The log loss is:', log_loss(y_test, y_predprob)
                     print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])

                 
                 
                 
if __name__=='__main__':
             trainfile = 'train_feature_v0.csv'
             testfile = 'test_reset.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             #classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
