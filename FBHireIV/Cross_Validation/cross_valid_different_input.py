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

class SupportVectorMachine:
         def __init__(self):
                 self.dataMat = []
                 self.labelMat = []
                 self.testData = []

         def loadDataSet(self,trainfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     if float(items[1])>3 and float(items[6])!=0.0 and float(items[7])!=0.0:
			          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
                                  #self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6])/float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          self.labelMat.append(int(float(items[12])));
			          #self.dataMat.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)
		 print self.dataMat
		 print self.labelMat

         def loadTestSet(self,testfile):
                 for line in open(trainfile,'r'):
			     items = line.strip().split(',');
			     if float(items[1])>3 and float(items[6])!=0.0 and float(items[7])!=0.0:
			          #self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11])])
			          self.testData.append([float(items[1])/float(items[2]),float(items[6])/float(items[7])])
		 self.testData = asarray(self.testData)
		          

	 def test(self):
                 iris = datasets.load_iris()
                 #X, y = iris.data, iris.target
                 X, y = self.dataMat,self.labelMat
                 X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=12)
                 clf_Dtree = DecisionTreeClassifier(max_depth=3,min_samples_split=2,min_samples_leaf=3)
                 clf_Dtree.fit(X_train, y_train)
                 y_pred = clf_Dtree.predict(X_test)
                 y_predprob = clf_Dtree.predict_proba(X_test)
                 print "Decision Tree"
                 print classification_report(y_test,y_pred)
                 print 'The accuracy is: ', accuracy_score(y_test,y_pred)
                 print 'The log loss is:', log_loss(y_test, y_predprob)
                 #print y_predprob
                 print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])

                 print "Random Forest"
                 clf_RFC = RandomForestClassifier(max_depth=3,min_samples_split=2,min_samples_leaf=3,n_estimators=15)
                 clf_RFC.fit(X_train, y_train)
                 y_pred = clf_RFC.predict(X_test)
                 y_predprob = clf_RFC.predict_proba(X_test)
                 print classification_report(y_test,y_pred)
                 print 'The accuracy is: ', accuracy_score(y_test,y_pred)
                 print 'The log loss is: ', log_loss(y_test, y_predprob)
                 print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])

                 print "   SVM"
                 clf_SVC = SVC(C=1, kernel='linear', probability=True)
                 clf_SVC.fit(X_train, y_train)
                 y_pred = clf_SVC.predict(X_test)
                 y_predprob = clf_SVC.predict_proba(X_test)
                 print classification_report(y_test,y_pred)
                 print 'The accuracy is: ', accuracy_score(y_test,y_pred)
                 print 'The log loss is: ', log_loss(y_test, y_predprob)
                 print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])
                 
                 print "   Logist Regression"
                 clf_Logist = linear_model.LogisticRegression(C=1e3,random_state=1)
                 clf_Logist.fit(X_train, y_train)
                 y_pred = clf_Logist.predict(X_test)
                 y_predprob = clf_Logist.predict_proba(X_test)
                 print classification_report(y_test,y_pred)
                 print 'The accuracy is: ', accuracy_score(y_test,y_pred)
                 print 'The log loss is: ', log_loss(y_test, y_predprob)
                 print 'The ROC score is:', roc_auc_score(y_test,y_predprob[:,1])
                 
                 
                 
if __name__=='__main__':
             trainfile = 'train_reset.csv'
             testfile = 'test_reset.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
