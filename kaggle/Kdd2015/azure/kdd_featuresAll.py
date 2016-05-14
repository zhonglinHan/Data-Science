import csv
import pandas as pd
from pandas import *
from numpy import *
import datetime, time
from kddref_azml_feature_1 import *

class Features:
       def __init__(self):
           self.dataTrainF = []
           self.dataTestF = []
           self.addTrainF = []
           self.addTestF = []
           self.logdata = []
       
       def loadLogData(self, datadir):
           self.logdata = pd.read_csv(datadir, header = 0)
       
       def df_to_csv(self, outfile, df):
           df.to_csv(outfile, header = True, index = False, index_label = False)
       
       def loadTrainF(self, trainfile):
           self.dataTrainF = pd.read_csv(trainfile, header = 0)

       def loadTestF(self, testfile):
           self.dataTestF = pd.read_csv(testfile, header = 0)

       def loadaddTestF(self, addtestfile):
           self.addTestF = pd.read_csv(addtestfile, header = 0)

       def loadaddTrainF(self, addtrainfile):
           self.addTrainF = pd.read_csv(addtrainfile, header = 0)

       def mergeTrainF(self, outfile, onkeys):
           mergeTrainF = merge(self.dataTrainF, self.addTrainF, how = 'inner', on = onkeys)
           mergeTrainF.to_csv(outfile, header = True, index = False, index_label = False)

       def mergeTestF(self, outfile, onkeys):
           mergeTestF = merge(self.dataTestF, self.addTestF, how = 'inner', on = onkeys)
           mergeTestF.to_csv(outfile, header = True, index = False, index_label = False)
       
       def changeSubFormat(self, subfile, outfile):
           #For Mac OS X, save your CSV file in "Windows Comma Separated (.csv)" format.
           #reader = csv.reader(open(subfile, 'rU'), dialect=csv.excel_tab)
           reader = csv.reader(open(subfile, 'rU'),delimiter=',')
           writer=csv.writer(open(outfile,'wb'))
           for row in reader:
               writer.writerow(row)

       def trainToPcent(self, featurelist):
           train_to_pcent = []

       def testToPcent(self, featurelist):
           test_to_pcent = []

       def trainToFreq(self, featurelist):
           train_to_freq = []

       def testToFreq(self, featurelist):
           test_to_freq = []

if __name__ == '__main__':
       #trainfile = '../data/train/kddtrainall_v3.csv'
       #addtrainfile = '../data/train/kddtrain_hour_perday_v2.csv'
       #testfile = '../data/test/kddtestall_v3.csv'
       #addtestfile = '../data/test/kddtest_hour_perday.csv'
       
       #onkeys_train = ['enrollment_id','label']
       #onkeys_test = 'enrollment_id'
       #outfile_train = '../data/train/kddtrainall_v5.csv'
       #outfile_test = '../data/test/kddtestall_v5.csv'

       features = Features()
       logdatadir = '../data/train/log_train.csv'
       outfile = '../data/train/kddtrain_azml_feat_pt1.csv'
       features.loadLogData(logdatadir)
       logdata = features.logdata
       #logdata = logdata.loc[logdata['enrollment_id']<5,['enrollment_id','time']]
       logdata = logdata[['enrollment_id','time']]
       print features.logdata.head()
       print features.logdata.shape
       azureml_main(logdata, None, outfile)
       #features.df_to_csv(outfile, logdata)


       #features.loadTrainF(trainfile)
       #features.loadTestF(testfile)
       #features.loadaddTrainF(addtrainfile)
       #features.loadaddTestF(addtestfile)

       #features.mergeTrainF(outfile_train, onkeys_train)
       #features.mergeTestF(outfile_test, onkeys_test)

       #features.changeSubFormat('../data/pred/pred_azml_v0.csv', '../data/pred/pred_azml_v2.csv')





