from numpy import *
import csv
import matplotlib.pyplot as plt
import scipy as sp

class SoftmaxRegression:
    def __init__(self):
		self.dataMat = []
		self.labelMat = []
		self.testData = []
		self.weights = []
		self.M = 0
		self.N = 0
		self.K = 0
                self.testM = 0
                self.testN = 0
		self.alpha = 0.001
                self.bias = 1.0
            

    def loadDataSet(self,trainfile):
            for line in open(trainfile,'r'):
                     items = line.strip().split(',')
                     self.dataMat.append([self.bias,float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24]),float(items[25]),float(items[26]),float(items[27]),float(items[28]),float(items[29]),float(items[30]),float(items[31]),float(items[32]),float(items[33]),float(items[34]),float(items[35]),float(items[36]),float(items[37]),float(items[38]),float(items[39]),float(items[40]),float(items[41]),float(items[42]),float(items[43]),float(items[44]),float(items[45]),float(items[46]),float(items[47]),float(items[48]),float(items[49]),float(items[50]),float(items[51]),float(items[52]),float(items[53]),float(items[54]),float(items[55]),float(items[56]),float(items[57]),float(items[58]),float(items[59]),float(items[60]),float(items[61]),float(items[62]),float(items[63]),float(items[64]),float(items[65]),float(items[66]),float(items[67]),float(items[68]),float(items[69]),float(items[70]),float(items[71]),float(items[72]),float(items[73]),float(items[74]),float(items[75]),float(items[76]),float(items[77]),float(items[78]),float(items[79]),float(items[80]),float(items[81]),float(items[82]),float(items[83]),float(items[84]),float(items[85]),float(items[86]),float(items[87]),float(items[88]),float(items[89]),float(items[90]),float(items[91]),float(items[92]),float(items[93])])
                     self.labelMat.append(int(items[94]))
                     
            self.K = len(set(self.labelMat))
            self.dataMat = mat(self.dataMat)
            self.labelMat = mat(self.labelMat).transpose()
            self.M,self.N = shape(self.dataMat)
            self.weights = mat(ones((self.N,self.K)))
            print shape(self.dataMat)
            print self.K
            
    def loadTestSet(self,testfile):
            for line in open(testfile,'r'):
                      items = line.strip().split(',')
                      self.testData.append([self.bias,float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24]),float(items[25]),float(items[26]),float(items[27]),float(items[28]),float(items[29]),float(items[30]),float(items[31]),float(items[32]),float(items[33]),float(items[34]),float(items[35]),float(items[36]),float(items[37]),float(items[38]),float(items[39]),float(items[40]),float(items[41]),float(items[42]),float(items[43]),float(items[44]),float(items[45]),float(items[46]),float(items[47]),float(items[48]),float(items[49]),float(items[50]),float(items[51]),float(items[52]),float(items[53]),float(items[54]),float(items[55]),float(items[56]),float(items[57]),float(items[58]),float(items[59]),float(items[60]),float(items[61]),float(items[62]),float(items[63]),float(items[64]),float(items[65]),float(items[66]),float(items[67]),float(items[68]),float(items[69]),float(items[70]),float(items[71]),float(items[72]),float(items[73]),float(items[74]),float(items[75]),float(items[76]),float(items[77]),float(items[78]),float(items[79]),float(items[80]),float(items[81]),float(items[82]),float(items[83]),float(items[84]),float(items[85]),float(items[86]),float(items[87]),float(items[88]),float(items[89]),float(items[90]),float(items[91]),float(items[92]),float(items[93])])

            self.testData = mat(self.testData)
            self.testM, self.testN = shape(self.testData)


    def likelihoodfunc(self):
	    likelihood = 0.0
	    for i in range(self.M):
		    t = exp(self.dataMat[i]*self.weights)
		    likelihood += log(t[0,self.labelMat[i,0]-1]/sum(t))
	    #print likelihood


                 
    def gradientAscent(self):
		for l in range(10):
			error = exp(self.dataMat*self.weights)
			rowsum = -error.sum(axis=1)
			rowsum = rowsum.repeat(self.K, axis=1)
			error = error/rowsum
			for m in range(self.M):
				error[m,self.labelMat[m,0]-1] += 1
			self.weights = self.weights + self.alpha * self.dataMat.transpose()* error
			
			self.likelihoodfunc()
		       # print self.weights
			
    def stochasticGradientAscent_V0(self):
		for l in range(500):
                        print l
			for i in range(self.M):
				error = exp(self.dataMat[i]*self.weights)
				rowsum = -error.sum(axis=1)
				rowsum = rowsum.repeat(self.K, axis=1)
				error = error/rowsum
				error[0,self.labelMat[i,0]-1] += 1
				self.weights = self.weights + self.alpha * self.dataMat[i].transpose()* error
			        #self.likelihoodfunc()

                
    def stochasticGradientAscent_V1(self):
                for l in range(500):
			idxs = range(self.M)
			for i in range(self.M):
				alpha = 4.0/(1.0+l+i)+0.01
				rdmidx = int(random.uniform(0,len(idxs)))
				error = exp(self.dataMat[rdmidx]*self.weights)
				rowsum = -error.sum(axis=1)
				rowsum = rowsum.repeat(self.K, axis=1)
				error = error/rowsum
				error[0,self.labelMat[rdmidx,0]-1] += 1
				self.weights = self.weights + alpha * self.dataMat[rdmidx].transpose()* error
				del(idxs[rdmidx])
				
				# self.likelihoodfunc()
		print self.weights



    def classify(self,X):
		p = X * self.weights
                Prob = exp (p)/sum(exp(p))
		return Prob
                #return p.argmax(1)[0,0]

	        
    def test(self):
         outf=open('output.csv','w')
         outf.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")         
         for i in range(self.testM):
                   classid  = self.classify(self.testData[i,:])
                   outf.write(str(i+1)+","+str(classid[0,0])+","+str(classid[0,1])+","+str(classid[0,2])+","+str(classid[0,3])+","+str(classid[0,4])+","+str(classid[0,5])+","+str(classid[0,6])+","+str(classid[0,7])+","+str(classid[0,8])+"\n")

                


     
if __name__=='__main__':
   trainfile = 'train.csv'
   testfile = 'test.csv' 
   myclassification = SoftmaxRegression()
   myclassification.loadDataSet(trainfile)
   myclassification.loadTestSet(testfile)
 #  myclassification.gradientAscent()
   myclassification.stochasticGradientAscent_V0()
   #myclassification.stochasticGradientAscent_V1()
   myclassification.test()
   


