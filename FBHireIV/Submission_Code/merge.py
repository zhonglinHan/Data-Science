from scipy import *
from numpy import *

file00=open('./merge/submission_svm_00.csv','r')
file01=open('./merge/submission_GradBoost_00.csv','r')
file02=open('./merge/submission_newfeat_12.csv','r')
output=open('./merge/submission_merge.csv','w')
outdict=open('./merge/submission_dictmerge.csv','w')

bidder = []
list00 = []
list01 = []
list02 = []
testData= []

for line in file00:
     items = line.strip().split(',');
     bidder.append(items[0]);
     list00.append([float(items[1])]);

for line in file01:
     items = line.strip().split(',');
     list01.append([float(items[1])]);

for line in file02:
     items = line.strip().split(',');
     list02.append([float(items[1])]);

list00=asarray(list00);
list01=asarray(list01);
list02=asarray(list02);

for line in open('./test_feature_v0.csv','r'):
      items = line.strip().split(',');
      testData.append([float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5]),float(items[6]),float(items[7]),float(items[8]),float(items[9]),float(items[10]),float(items[11]),float(items[12]),float(items[13]),float(items[14]),float(items[15]),float(items[16]),float(items[17]),float(items[18]),float(items[19]),float(items[20]),float(items[21]),float(items[22]),float(items[23]),float(items[24])])      

testData = asarray(testData)
output.write('bidder_id'+','+'prediction'+'\n')
for i in range(0,len(list01)):
    meanval = mean([list00[i],list01[i],list02[i]])
    stdval = std([list00[i],list01[i],list02[i]])*100
    if stdval > 10:
        #oup = min(meanval+stdval/100,0.999)
        output.write(str(bidder[i])+','+str(min(meanval+stdval/100,0.999))+'\n')
        #outdict.write(str(bidder[0])+","+str(testData[0])+","+str(testData[1])+","+str(testData[2])+","+str(testData[3])+","+str(testData[4])+","+str(testData[5])+","+str(testData[6])+","+str(testData[7])+","+str(testData[8])+","+str(testData[9])+","+str(testData[10])+","+str(testData[11])+","+str(testData[12])+","+str(testData[13])+","+str(testData[14])+","+str(testData[15])+","+str(testData[16])+","+str(testData[17])+","+str(testData[18])+","+str(testData[19])+","+str(testData[20])+","+str(testData[21])+","+str(testData[22])+","+str(testData[23])+","+str(testData[24])+"\n")
    else:
        output.write(str(bidder[i])+','+str(float(list02[i]))+'\n')