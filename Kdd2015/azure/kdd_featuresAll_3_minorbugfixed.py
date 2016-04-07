import csv
import pandas as pd
from pandas import *
from numpy import *
import datetime, time
#from datetime import date, timedelta as td
#from kddref_azml_feature_1 import *
#from kddref_azml_feature_2 import *
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

       def splitdays(self, timestart, timeend):
           #fmt = '%m/%d/%y'
           fmt = '%Y-%m-%d'
           timestart = datetime.datetime.strptime(timestart, fmt)
           timeend = datetime.datetime.strptime(timeend,fmt)
           delta = timeend - timestart
           timedates = []
           #for i in range(delta.days+2):
           for i in range(delta.days+1):
               timedates = timedates + [timestart + datetime.timedelta(days = i)]
           return timedates
       
       def getcounts(self, log, weight, timedates):
           fmt = '%Y-%m-%dT%H:%M:%S'
           # timedates is the array of starting and ending time of each of the 30 days in course period
           # log is the log piece of only ONE particular enrollment_id
           # output should be several arrays of length 30 for event counts, obj counts, etc.
           # weight: weight in scoring each event
           event_dict = dict(zip(['access','problem','page_close','navigate','video','discussion','wiki'],range(0,7)))
           #obj_dict = dict(zip(['access','problem','page_close','navigate','video','discussion','wiki'],range(7,14)))
           obj_dict = {'obj': 7}
           count_features = zeros((len(event_dict) + len(obj_dict), len(timedates)))
           count_features_wsum = zeros((1, len(timedates)+1))
           #count_features_wsum[0,0] = log.iloc[0,0] # enrollment_id
           count_features_wsum[0,0] = log[0][0] # enrollment_id
           #num_obs = log.shape[0]
           num_obs = len(log)
           timedates = array(timedates)
           #obj_previous = '----'
           #event_index_previous = -1
           timeidx_previous = -1
           obj_lst = []
           #for i in range(log.shape[0]):
           for i in range(num_obs):
               #timept = datetime.datetime.strptime(log.iloc[i,1],fmt)
               timept = datetime.datetime.strptime(log[i][1],fmt)
               timept = datetime.datetime(timept.year, timept.month, timept.day)
               timeidx = where(timedates == timept)
               #print sum(timedates == timept)
               #print timeidx[0][0]
               timeidx = int(timeidx[0][0])
               #timeidx = timeidx + 1
               
               event_index = event_dict[log[i][3]]
               obj_index = obj_dict['obj']
               count_features[event_index,timeidx] += 1
               
               if timeidx != timeidx_previous:
                  if i > 0:
                     count_features[obj_index,timeidx_previous] = len(unique(obj_lst))
                  obj_lst = []
                  timeidx_previous = timeidx
               obj_lst = obj_lst + [log[i][4]]
           
           count_features[obj_index,timeidx] = len(unique(obj_lst))
               #event_index = event_dict[log.iloc[i,3]]
               
               
               #obj_index = obj_dict[log.iloc[i,3]]
               #obj_index = obj_dict[log[i][3]]
               #obj_index = 7
               #obj_current = log.iloc[i,4]
               #obj_current = log[i][4]
               
               #if event_index != event_index_previous:
               #if obj_current != obj_previous:
               #   count_features[obj_index, timeidx] += 1
               #elif obj_current != obj_previous or
               
               #obj_previous = obj_current
               #event_index_previous = event_index
               
           
           
           
           for j in range(len(timedates)):
               count_features_wsum[0, j+1] = sum(weight * count_features[:,j])
           return count_features_wsum

       def dayfeaturemain(self, enroll_data, log_data, log_course_date, weight):
           course_ids = list(set(log_course_date.iloc[:,0]))
           num_courses = len(course_ids)
           unique_enrollment = list(set(log_data.iloc[:,0]))
           num_unique_enrollment = len(unique_enrollment)
           
           #enrollment_dict = dict(zip(unique_enrollment,range(num_unique_enrollment)))
           #courseid_dict = dict(zip(course_ids), range(num_courses))
           
           numrows = log_data.shape[0]
           timedates = self.splitdays('2014-05-26', '2014-06-24')
           count_features_total = zeros((num_unique_enrollment, 1 + len(timedates)))
           
           previous_id = -1
           enrollment_index = 0
           enrollment_log = []
           
           for i in range(numrows):
               print i, "out of", numrows
               current_id = log_data.iloc[i][0]
               if current_id != previous_id:
                  
                  if i > 0:
                     course_id = enroll_data.iloc[enrollment_index, 2]
                     course_index = where(log_course_date.iloc[:,0] == course_id)
                     course_index = int(course_index[0])
                  
                     timestart =log_course_date.iloc[course_index, 1]
                     timeend = log_course_date.iloc[course_index, 2]
                     timedates = self.splitdays(timestart, timeend)
                     count_features_total[enrollment_index,:] = self.getcounts(enrollment_log, weight, timedates)
                     enrollment_index += 1
                  enrollment_log = []
                  previous_id = current_id
               enrollment_log.append(log_data.iloc[i])
               #print type(enrollment_log)
           course_id = enroll_data.iloc[enrollment_index, 2]
           course_index = where(log_course_date.iloc[:,0] == course_id)
           course_index = int(course_index[0])
           timestart =log_course_date.iloc[course_index, 1]
           timeend = log_course_date.iloc[course_index, 2]
           timedates = self.splitdays(timestart, timeend)
           count_features_total[enrollment_index,:] = self.getcounts(enrollment_log, weight, timedates)
           df =  pd.DataFrame(count_features_total)
           featurenames = []
           for i in range(len(timedates)):
               featurenames = featurenames + ['score_day_' + str(i)]
           df.columns = ['enrollment_id'] + featurenames
           return df
           # output an array of length 30 or 31 including starting and ending time point of each day of the time span defined by timestart and timeend



if __name__ == '__main__':
       features = Features()
       timedates = features.splitdays('2014-05-26', '2014-06-24')
       print timedates
       
       weight = 1
       #weight = array([1,1,1,1,1,1,1,0])
       #weight = array([0,0,0,0,0,0,0,1])
       enroll_data = read_csv('../data/train/enrollment_train.csv', header = 0)
       log_data = read_csv('../data/train/log_train.csv', header = 0)
       log_course_date = read_csv('../data/date.csv', header = 0)
       
       #enroll_data = enroll_data.loc[enroll_data['enrollment_id']<=20,:]
       #log_data = log_data.loc[log_data['enrollment_id']<=20, :]
       df = features.dayfeaturemain(enroll_data, log_data, log_course_date, weight)
       
       df_train = read_csv('../data/train/kddtrainall_v1.csv', header = 0)
       print df.head()
       df.to_csv('../data/train/kddtrain_countbyday.csv', header = True, index = False, index_label = False)
       #sumlist = ['problem','npobj', 'video', 'nvobj', 'access', 'naobj', 'wiki', 'nwobj','discussion', 'ndobj', 'navigate', 'nnobj', 'page_close', 'ncobj']
       #sumlist = ['npobj','nvobj', 'naobj', 'nwobj', 'ndobj', 'nnobj', 'ncobj']
       #sumlist = ['problem','video','access','wiki','discussion', 'navigate', 'page_close']
       #for i in range(df.shape[0]):
       #    print "enrollment_id: ", df.iloc[i,0]
       #    print "the sum of events and obj: ", sum(df.iloc[i,1:])
       #    print "indeed the sum should be: ", sum( df_train.loc[df_train['enrollment_id']== df.iloc[i,0],sumlist].values )







       #trainfile = '../data/train/kddtrainall_v7.csv'
       #addtrainfile = '../data/train/kddtrain_azml_feat_pt2.csv'
       #testfile = '../data/test/kddtestall_v3.csv'
       #addtestfile = '../data/test/kddtest_hour_perday.csv'
       
       #onkeys_train = 'enrollment_id'
       #onkeys_test = 'enrollment_id'
       #outfile_train = '../data/train/kddtrainall_v7.csv'
       #outfile_test = '../data/test/kddtestall_v5.csv'



       #logdatadir = '../data/test/log_test.csv'
       #outfile1 = '../data/test/kddtest_azml_feat_pt1.csv'
       #outfile = '../data/test/kddtest_azml_feat_pt2.csv'
       #features.loadLogData(logdatadir)
       #logdata = features.logdata
       #logdata = logdata.loc[logdata['enrollment_id']<5,['enrollment_id','time']]
       #logdata = logdata[['enrollment_id','time']]
       #logdata = logdata.loc[logdata['enrollment_id']<5, ['enrollment_id','time','source','event']]
       #logdata = logdata[['enrollment_id','time','source','event']]
       #print logdata.head()
       #print logdata.shape
       #azureml_main(logdata, None, outfile)
       #features.df_to_csv(outfile, logdata)
       #azureml_main_2(logdata, outfile)


       #features.loadTrainF(trainfile)
       #features.loadTestF(testfile)
       #features.loadaddTrainF(addtrainfile)
       #features.loadaddTestF(addtestfile)

       #features.mergeTrainF(outfile_train, onkeys_train)
       #features.mergeTestF(outfile_test, onkeys_test)

       #features.changeSubFormat('../data/pred/pred_azml_v0.csv', '../data/pred/pred_azml_v2.csv')
