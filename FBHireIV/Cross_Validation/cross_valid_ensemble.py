import matplotlib.pyplot as plt
import scipy as sp
from math import *

from sklearn import *
from sklearn import cross_validation,linear_model
from numpy import *
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from nolearn.dbn import DBN

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import cross_val_score
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
			     if float(items[1])>2 and float(items[6])!=0.0:
			          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])
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
                 #clf = RandomForestClassifier(max_depth=3,min_samples_split=9,min_samples_leaf=15,n_estimators=5)
                 #for w1 in arange(0.342, 0.347, 0.001):
                 params = {'n_estimators': 1200, 'max_depth': 4, 'subsample': 0.5,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3};
                 clf_GBC = GradientBoostingClassifier(**params);
                 clf_GBC.fit(X_train, y_train);
                 scores_GBC = cross_val_score(clf_GBC,X,y,cv=3,scoring='roc_auc')
                 clf_RFC = RandomForestClassifier(max_depth=6,min_samples_split=7, min_samples_leaf=9,n_estimators=12);
                 clf_RFC.fit(X_train, y_train);
                 scores_RFC = cross_val_score(clf_RFC,X,y,cv=3,scoring='roc_auc')
                 clf_SVC = SVC(kernel='linear', C= 0.001, probability=True);
                 clf_SVC.fit(X_train, y_train);
                 scores_SVC = cross_val_score(clf_SVC,X,y,cv=3,scoring='roc_auc')
                 for w1 in arange(0.01, 0.99, 0.01):
                   for w2 in arange(0.01, 0.99, 0.01):
                       y_predprob = clf_GBC.predict_proba(X_test)*w1+clf_RFC.predict_proba(X_test)*(1-w2)*(1-w1)+clf_SVC.predict_proba(X_test)*w2*(1-w1);
                       scoremean = scores_GBC.mean()*w1+scores_RFC.mean()*(1-w2)*(1-w1)+scores_SVC.mean()*w2*(1-w1)
                       if scoremean>0.9:
                          print '***********************************************************'
                          print 'GBC-weight =', w1, 'RFC =',(1-w1)*(1-w2), 'SVC =',w2*(1-w1)
                          print 'The log loss is:', log_loss(y_test, y_predprob)
                          print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])
                          scorestd = math.sqrt(scores_GBC.std()**2+scores_RFC.std()**2+scores_SVC.std()**2)
                          print ("Accuracy: %0.5f (+/- %0.5f)" % (scores_GBC.mean()*w1+scores_RFC.mean()*(1-w2)*(1-w1)+scores_SVC.mean()*w2*(1-w1), scorestd*2))

                 
                 
                 
if __name__=='__main__':
             trainfile = 'train_feature_v0.csv'
             testfile = 'test_reset.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             #classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
