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
                 self.convertdata = []
                 self.labelMat = []
                 self.testData = []

         def loadDataSet(self,trainfile):
                 for line in open(trainfile,'r'):
			  items = line.strip().split(',')
                          self.dataMat.append([int(items[1]),int(items[2]),int(items[3]),int(items[4]),int(items[5]),int(items[6]),int(items[7]),int(items[8]),int(items[9]),int(items[10]),int(items[11]),int(items[12]),int(items[13]),int(items[14]),int(items[15]),int(items[16]),int(items[17]),int(items[18]),int(items[19]),int(items[20]),int(items[21]),int(items[22]),int(items[23]),int(items[24]),int(items[25]),int(items[26]),int(items[27]),int(items[28]),int(items[29]),int(items[30]),int(items[31]),int(items[32]),int(items[33]),int(items[34]),int(items[35]),int(items[36]),int(items[37]),int(items[38]),int(items[39]),int(items[40]),int(items[41]),int(items[42]),int(items[43]),int(items[44]),int(items[45]),int(items[46]),int(items[47]),int(items[48]),int(items[49]),int(items[50]),int(items[51]),int(items[52]),int(items[53]),int(items[54]),int(items[55]),int(items[56]),int(items[57]),int(items[58]),int(items[59]),int(items[60]),int(items[61]),int(items[62]),int(items[63]),int(items[64]),int(items[65]),int(items[66]),int(items[67]),int(items[68]),int(items[69]),int(items[70]),int(items[71]),int(items[72]),int(items[73]),int(items[74]),int(items[75]),int(items[76]),int(items[77]),int(items[78]),int(items[79]),int(items[80]),int(items[81]),int(items[82]),int(items[83]),int(items[84]),int(items[85]),int(items[86]),int(items[87]),int(items[88]),int(items[89]),int(items[90]),int(items[91]),int(items[92]),int(items[93])])
			  self.labelMat.append(float(items[94]))

		 self.dataMat = asarray(self.dataMat)
		 self.convertdata = zeros(shape(self.dataMat))
		 for j in range(len(self.dataMat[0,:])):
            		 binnum = bincount(self.dataMat[:,j])
	         	 weight = binnum*1.0/(sum(binnum)-binnum[0])
                         for i in range(len(self.dataMat[:,0])):
		                self.convertdata[i,j] = 1.0*self.dataMat[i,j]*weight[self.dataMat[i,j]]
		 self.labelMat = asarray(self.labelMat)-1
                 print self.labelMat
		 
	 def loadTestSet(self,testfile):
                 for line in open(testfile,'r'):
                        items = line.strip().split(',')
                        self.testData.append([int(items[1]),int(items[2]),int(items[3]),int(items[4]),int(items[5]),int(items[6]),int(items[7]),int(items[8]),int(items[9]),int(items[10]),int(items[11]),int(items[12]),int(items[13]),int(items[14]),int(items[15]),int(items[16]),int(items[17]),int(items[18]),int(items[19]),int(items[20]),int(items[21]),int(items[22]),int(items[23]),int(items[24]),int(items[25]),int(items[26]),int(items[27]),int(items[28]),int(items[29]),int(items[30]),int(items[31]),int(items[32]),int(items[33]),int(items[34]),int(items[35]),int(items[36]),int(items[37]),int(items[38]),int(items[39]),int(items[40]),int(items[41]),int(items[42]),int(items[43]),int(items[44]),int(items[45]),int(items[46]),int(items[47]),int(items[48]),int(items[49]),int(items[50]),int(items[51]),int(items[52]),int(items[53]),int(items[54]),int(items[55]),int(items[56]),int(items[57]),int(items[58]),int(items[59]),int(items[60]),int(items[61]),int(items[62]),int(items[63]),int(items[64]),int(items[65]),int(items[66]),int(items[67]),int(items[68]),int(items[69]),int(items[70]),int(items[71]),int(items[72]),int(items[73]),int(items[74]),int(items[75]),int(items[76]),int(items[77]),int(items[78]),int(items[79]),int(items[80]),int(items[81]),int(items[82]),int(items[83]),int(items[84]),int(items[85]),int(items[86]),int(items[87]),int(items[88]),int(items[89]),int(items[90]),int(items[91]),int(items[92]),int(items[93])])

                 self.testData = asarray(self.testData)-1
		         

	 def test(self):
                 outf=open('output_v0_test.csv','w')
                 outf.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
                 #iris = datasets.load_iris()
                 #X, y = iris.data, iris.target
                 #testDM,testDN = shape(self.testData)
                 X, y = self.convertdata, self.labelMat
                 clf = SVC(C=7.5, kernel='rbf', probability=True)
                 #clf.fit(X,y)
                 X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=12)
                 clf.fit(X_train, y_train)
                 scrk = clf.score(X_test, y_test)
                 print str(scrk)
                 #for i in range(testDM):
                 #classid = clf.predict_proba(self.testData[0,:])
                 #outf.write(str(i+1)+","+str(classid[0,0])+","+str(classid[0,1])+","+str(classid[0,2])+","+str(classid[0,3])+","+str(classid[0,4])+","+str(classid[0,5])+","+str(classid[0,6])+","+str(classid[0,7])+","+str(classid[0,8])+"\n")
                          
                 #print OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X[0,:])
                 

if __name__=='__main__':
             trainfile = 'train.csv'
             testfile = 'test.csv'
             classificationfuc = SupportVectorMachine()
             classificationfuc.loadDataSet(trainfile)
             #classificationfuc.loadTestSet(testfile)
             classificationfuc.test()
