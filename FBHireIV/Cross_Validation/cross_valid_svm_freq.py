from sklearn import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from numpy import *
#import matplotlib.pyplot as plt
import scipy as sp

class SupportVectorMachine:
         def __init__(self):
                 self.dataMat = []
                 self.labelMat = []
                 self.testData = []

         def loadDataSet(self,trainfile):
                 for line in open(trainfile,'r'):
			  items = line.strip().split(',')
                          self.dataMat.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24]),float(items[25]),float(items[26]),float(items[27]),float(items[28]),float(items[29]),float(items[30]),float(items[31]),float(items[32]),float(items[33]),float(items[34]),float(items[35]),float(items[36]),float(items[37]),float(items[38]),float(items[39]),float(items[40]),float(items[41]),float(items[42]),float(items[43]),float(items[44]),float(items[45]),float(items[46]),float(items[47]),float(items[48]),float(items[49]),float(items[50]),float(items[51]),float(items[52]),float(items[53]),float(items[54]),float(items[55]),float(items[56]),float(items[57]),float(items[58]),float(items[59]),float(items[60]),float(items[61]),float(items[62]),float(items[63]),float(items[64]),float(items[65]),float(items[66]),float(items[67]),float(items[68]),float(items[69]),float(items[70]),float(items[71]),float(items[72]),float(items[73]),float(items[74]),float(items[75]),float(items[76]),float(items[77]),float(items[78]),float(items[79]),float(items[80]),float(items[81]),float(items[82]),float(items[83]),float(items[84]),float(items[85]),float(items[86]),float(items[87]),float(items[88]),float(items[89]),float(items[90]),float(items[91]),float(items[92]),float(items[93])])
			  self.labelMat.append(float(items[94]))

		 self.dataMat = asarray(self.dataMat)
		 self.labelMat = asarray(self.labelMat)-1
                 print self.labelMat
		 
	 def loadTestSet(self,testfile):
                 for line in open(testfile,'r'):
                        items = line.strip().split(',')
                        self.testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24]),float(items[25]),float(items[26]),float(items[27]),float(items[28]),float(items[29]),float(items[30]),float(items[31]),float(items[32]),float(items[33]),float(items[34]),float(items[35]),float(items[36]),float(items[37]),float(items[38]),float(items[39]),float(items[40]),float(items[41]),float(items[42]),float(items[43]),float(items[44]),float(items[45]),float(items[46]),float(items[47]),float(items[48]),float(items[49]),float(items[50]),float(items[51]),float(items[52]),float(items[53]),float(items[54]),float(items[55]),float(items[56]),float(items[57]),float(items[58]),float(items[59]),float(items[60]),float(items[61]),float(items[62]),float(items[63]),float(items[64]),float(items[65]),float(items[66]),float(items[67]),float(items[68]),float(items[69]),float(items[70]),float(items[71]),float(items[72]),float(items[73]),float(items[74]),float(items[75]),float(items[76]),float(items[77]),float(items[78]),float(items[79]),float(items[80]),float(items[81]),float(items[82]),float(items[83]),float(items[84]),float(items[85]),float(items[86]),float(items[87]),float(items[88]),float(items[89]),float(items[90]),float(items[91]),float(items[92]),float(items[93])])

                 self.testData = asarray(self.testData)-1
		         

	 def test(self):
                 outf=open('output_v0_test.csv','w')
                 outf.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
                 #iris = datasets.load_iris()
                 #X, y = iris.data, iris.target
                 #testDM,testDN = shape(self.testData)
                 X, y = self.dataMat, self.labelMat
                 clf = SVC(C=7.5, kernel='rbf', probability=True)
                 #clf.fit(X,y)
                 X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=12)
                 clf.fit(X_train, y_train)
                 scrk = clf.score(X_test, y_test)
                 print str(scrk)
                 for i in range(testDM):
                       classid = clf.predict_proba(self.testData[0,:])
                       outf.write(str(i+1)+","+str(classid[0,0])+","+str(classid[0,1])+","+str(classid[0,2])+","+str(classid[0,3])+","+str(classid[0,4])+","+str(classid[0,5])+","+str(classid[0,6])+","+str(classid[0,7])+","+str(classid[0,8])+"\n")
                          
                 #print OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X[0,:])
                 

if __name__=='__main__':
             trainfile = 'train_freq.csv'
             testfile = 'test.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             #classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
