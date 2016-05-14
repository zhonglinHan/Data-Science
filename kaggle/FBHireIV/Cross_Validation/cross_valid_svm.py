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
			     if float(items[1])>=0 or float(items[6])>=0.0:
			          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
			          self.labelMat.append(int(float(items[25])));
		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)

         def loadTestSet(self,testfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     if float(items[1])>1 and float(items[7])!=0.0:
                             #if float(items[7])>=0.0:
			          self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          #self.testData.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.testData = asarray(self.testData)
		          

	 def test(self):
                 #iris = datasets.load_iris()
                 #X, y = iris.data, iris.target
                 X, y = self.dataMat,self.labelMat
                 X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.6, random_state=12)
                 cvnum = ShuffleSplit(2013,n_iter=10,test_size=0.6,train_size=0.4,random_state=0)
                 for depth in range(1,20):
                     for split in range(1,20):
                        for leaf in range(1,20):
                            for estimat in range(1,20):
                                clf_RFC = RandomForestClassifier(max_depth=depth,min_samples_split=split,min_samples_leaf=leaf,n_estimators=estimat)
                                clf_RFC.fit(X_train, y_train);
                                y_pred = clf_RFC.predict(X_test);
                                y_predprob = clf_RFC.predict_proba(X_test);
                                prf=precision_recall_fscore_support(y_test, y_pred, average='binary')
                                scores = cross_val_score(clf_RFC,X,y,cv=cvnum,scoring='roc_auc')
                                if scores.mean()>0.9 and prf[0]>0.95 and prf[1]>0.1:
                                #if prf[0]>0.95 and prf[1]>0.1 and roc_auc_score(y_test,y_predprob[:,1])>0.90:
                                   print ("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
                                   print "Random Forest","depth=",depth, " split=",split," leaf=",leaf, "n_estimat=",estimat
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
