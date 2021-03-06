import csv
import pandas as pd
from pandas import *
from numpy import *
import datetime, time
from scipy import stats
from math import *
import time
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
       
       def pctbycourse(self, outfile_train, outfile_test, features_to_pct):
           df_train = self.dataTrainF
           df_test = self.dataTestF
           df_enr_train = pd.read_csv('../data/train/enrollment_train.csv', header = 0)
           df_enr_test = pd.read_csv('../data/test/enrollment_test.csv', header = 0)
           df_train = merge(df_train, df_enr_train, how = 'inner', on = 'enrollment_id')
           df_train['istrain'] = 1
           df_test = merge(df_test, df_enr_test, how = 'inner', on = 'enrollment_id')
           df_test['istrain'] = 0
           
           #print df_train['istrain'].head()
           
           dfm = concat([df_train, df_test], ignore_index=True)
           dfm = dfm[df_train.columns.values]
           #dfm = dfm[0:10]
           
           
           
           course_list = unique(dfm['course_id'].values.tolist())
           dfm = dfm.sort(columns = 'course_id', ascending = True)
           dfm = dfm.set_index([range(dfm.shape[0])])
           
           #dfm = dfm[0:1000]
           dfm.to_csv('../data/train/dfm.csv', header = True, index = False, index_label = False)
           n_enr = dfm.shape[0]
           L_enr = range(n_enr)

           #print dfm.head()
           #df_train = df_train.sort(columns = 'course_id', ascending = True)
           #df_test = df_test.sort(columns = 'course_id', ascending = True)
           df_pct = zeros((n_enr, 3 + len(features_to_pct)))
           #df_pct = DataFrame(columns = ['enrollment_id'] + ['label'] + ['istrain'] + ['course_id'] + features_to_pct)
           course_id_previous = ''
           i_first = 0
           Lfeat = range(len(features_to_pct))
           for i in L_enr:
               #if i >= 0: break
               df_pct[i,0] = dfm.loc[i, 'enrollment_id']
               df_pct[i,1] = dfm.loc[i, 'label']
               df_pct[i,2] = dfm.loc[i, 'istrain']
               #df_pct.loc[i, 'enrollment_id'] = dfm.loc[i, 'enrollment_id']
               #df_pct.loc[i, 'label'] = dfm.loc[i, 'label']
               #df_pct.loc[i, 'istrain'] = dfm.loc[i, 'istrain']
               #df_pct.loc[i, 'course_id'] = dfm.loc[i, 'course_id']
               print i, ' out of ', n_enr
               #print dfm.iloc[i,:]
               course_id_current = dfm.loc[i, 'course_id']
               #print course_id_current
               #if i > 1: break
               if course_id_current != course_id_previous:
                  if i > 0:
                     i_last = i-1
                     #df_pct[i_first : i_last + 1, 0] = dfm.loc[i_first : i_last + 1, 'enrollment_id'].values
                     for j in Lfeat:
                         feat = features_to_pct[j]
                         feat_collect = dfm.loc[i_first : i_last, feat].values.tolist()
                         #print i_first, i_last, len(feat_collect)
                         #vals = reshape([stats.percentileofscore(feat_collect, a,'strict') for a in feat_collect], [len(feat_collect), ])
                         vals = [stats.percentileofscore(feat_collect, a,'strict') for a in feat_collect]
                         vals = list(array(vals)/max(0.0000001,max(vals)) * 100)
                         #df_pct.loc[i_first : i_last, feat] = vals
                         df_pct[i_first : i_last+1, j+3] = vals
                     
                     i_first = i_last + 1
                  course_id_previous = course_id_current
           #df_pct[i_first : n_enr, 0] = dfm.loc[i_first:n_enr, 'enrollment_id'].values
           for j in Lfeat:
               #if j >= 0: break
               feat = features_to_pct[j]
               feat_collect = dfm.loc[i_first : n_enr-1, feat].values.tolist()
               #vals = reshape([stats.percentileofscore(feat_collect, a,'strict') for a in feat_collect], [len(feat_collect), ])
               vals = [stats.percentileofscore(feat_collect, a,'strict') for a in feat_collect]
               vals = list(array(vals)/max(0,0000001, max(vals)) * 100)
               #df_pct.loc[i_first : n_enr-1, feat] = vals
               df_pct[i_first : n_enr, j+3] = vals
           df_pct = DataFrame(df_pct)
           df_pct.columns = ['enrollment_id'] + ['label'] + ['istrain'] + features_to_pct
           df_train_pct = df_pct[df_pct['istrain'] == 1]
           df_test_pct = df_pct[df_pct['istrain'] == 0]
           df_train_pct.to_csv(outfile_train, header = True, index = False, index_label = False)
           df_test_pct.to_csv(outfile_test, header = True, index = False, index_label = False)
           #print type(df_pct['label'])
           #if any(isnan(df_pct['label'])):
           #   df_test_pct = df_pct[isnan(dfm['label'])]
           #if any(~isnan(df_pct['label'])):
           #   df_train_pct = df_pct[~isnan(df_pct['label'])]

               
           #dfm.drop(dfm[isnan(dfm['label'])].index)
       
       
       
       def delOutliers(self, I_out, outfile):
           #dic_outliers = {}
           df = self.dataTrainF
           df_drop_ext = df.drop(df.index[I_out])
           df_drop_ext.to_csv(outfile, header = True, index = False, index_label = False)
       
       #def delOutliers(self, outlierfile, outfile):
           #dic_outliers = {}
       #    df = self.dataTrainF
       #    count = 0
       #    for line in csv.reader(open(outlierfile,'rb')):
       #        line =  map(int, filter(None,line))
       #        #if count == 0:
       #        print line
       #        df = df.drop(df.index[line])
       #        count += 1
           
           
           
           #for key, val in csv.reader(open(outlierfile)):
           #    dic_outliers[key] = val
           
           
           #df = df.drop(df.index[dic_outliers['out_1_asc']])
           
           
           
           
       def getOutliersAscFromFile(self, file, features_asc, th_asc):
           #self.dataTrainF = self.dataTrainF[0:5000]
           #print self.dataTrainF.head()
           #print self.dataTrainF.shape
           df = read_csv(file, header = 0)
           df_features_asc = df[features_asc]
           print df_features_asc.head()
           
           I_out_0_asc = [] # indicies of outlier samples(not enrollment_id!)
           I_in_1_asc = [] # indicies of inlier samples
           I_out_1_asc = []
           I_in_0_asc = []
           for i in range(df.shape[0]):
               
               if df['label'][i] == 0:
                  if all(df_features_asc.iloc[i, :].values <= th_asc * 100):
                     print i, 'out of', df.shape[0]
                     I_out_0_asc = I_out_0_asc + [i]
                  elif all(df_features_asc.iloc[i, :].values >= (1 - th_asc) * 100):
                     I_in_0_asc = I_in_0_asc + [i]
               else:
                  if all(df_features_asc.iloc[i, :].values >= (1 - th_asc) * 100):
                     print i, 'out of', df.shape[0]
                     I_out_1_asc = I_out_1_asc + [i]
                  elif all(df_features_asc.iloc[i, :].values <= th_asc * 100):
                     I_in_1_asc = I_in_1_asc + [i]
           dic_outliers = {'out_0_asc': I_out_0_asc,
                           'in_1_asc': I_in_1_asc,
                           'out_1_asc': I_out_1_asc,
                           'in_0_asc': I_in_0_asc}
           #df = self.dataTrainF.loc[I_out_0_asc + I_in_1_asc + I_out_1_asc + I_in_0_asc]
           #df['outlabel'] = list(zeros((len(I_out_0_asc),1))) + list(ones((len(I_in_1_asc),1))) + list(2 * ones((len(I_out_1_asc),1))) + list(3 * ones((len(I_in_0_asc),1)))
           
           #df.to_csv('../data/train/kddtrain_outlier_v2.csv', header = True, index = False, index_label = False)
                           
           return dic_outliers
    
           
       
       def getOutliersAsc(self, features_asc, th_asc):
           #self.dataTrainF = self.dataTrainF[0:5000]
           #print self.dataTrainF.head()
           print self.dataTrainF.shape
           df_features_asc = DataFrame(columns = features_asc)
           
           for feat_a in features_asc:
                print feat_a, 'out of', features_asc
                feat_a_data = self.dataTrainF[feat_a].values.tolist()
                df_features_asc[feat_a] = [stats.percentileofscore(feat_a_data, a,'strict') for a in feat_a_data]
                df_features_asc[feat_a] = df_features_asc[feat_a] / max(df_features_asc[feat_a].values.tolist())
           print df_features_asc.head()
           
           I_out_0_asc = [] # indicies of outlier samples(not enrollment_id!)
           I_in_1_asc = [] # indicies of inlier samples
           I_out_1_asc = []
           I_in_0_asc = []
           for i in range(self.dataTrainF.shape[0]):
               print i, 'out of', self.dataTrainF.shape[0]
               if self.dataTrainF['label'][i] == 0:
                  if all(df_features_asc.iloc[i, :].values <= th_asc):
                     I_out_0_asc = I_out_0_asc + [i]
                  elif all(df_features_asc.iloc[i, :].values >= 1 - th_asc):
                     I_in_0_asc = I_in_0_asc + [i]
               else:
                  if all(df_features_asc.iloc[i, :].values >= 1 - th_asc):
                     I_out_1_asc = I_out_1_asc + [i]
                  elif all(df_features_asc.iloc[i, :].values <= th_asc):
                     I_in_1_asc = I_in_1_asc + [i]
           dic_outliers = {'out_0_asc': I_out_0_asc,
                           'in_1_asc': I_in_1_asc,
                           'out_1_asc': I_out_1_asc,
                           'in_0_asc': I_in_0_asc}
           df = self.dataTrainF.loc[I_out_0_asc + I_in_1_asc + I_out_1_asc + I_in_0_asc]
           df.to_csv('../data/train/kddtrain_outlier_v1.csv', header = True, index = False, index_label = False)
                           
           return dic_outliers
       
       
       def getOutliersBothSides(self, features_asc, th_asc, features_dsc, th_dsc):
           # asc means the bigger the feature value, the more possible the client will remain in the course
           # dsc means the bigger the feature value, the more possible the client will drop the course
           df_features_asc = DataFrame(columns = features_asc)
           df_features_dsc = DataFrame(columns = features_dsc)
           for feat_a, feat_d in features_asc, features_dsc:
                feat_a_data = self.dataTrainF[feat_a].values.tolist()
                feat_d_data = self.dataTrainF[feat_d].values.tolist()
                df_features_asc[feat_a] = [stats.percentileofscore(feat_a_data, a) for a in feat_a_data]
                df_features_asc[feat_d] = [stats.percentileofscore(feat_d_data, a) for a in feat_d_data]
           
           I_out_0_asc = [] # indicies of outlier samples(not enrollment_id!)
           I_out_0_dsc = []
           I_out_1_asc = []
           I_out_1_dsc = []
           for i in range(self.dataTrainF.shape[0]):
               if self.dataTrainF['label'][i] == 0:
                  if all(df_features_asc.iloc[i, :].values <= th_asc):
                     I_out_0_asc = I_out_0_asc + [i]
                  elif all(df_features_dsc.iloc[i, :].values >= 1 - th_dsc):
                     I_out_0_dsc = I_out_0_dsc + [i]
               else:
                  if all(df_features_asc.iloc[i, :].values >= 1 - th_asc):
                     I_out_1_asc = I_out_1_asc + [i]
                  elif all(df_features_dsc.iloc[i, :].values <= th_dsc):
                     I_out_1_dsc = I_out_1_dsc + [i]
           dic_outliers = {'out_0_asc': I_out_0_asc,
                           'out_0_dsc': I_out_0_dsc,
                           'out_1_asc': I_out_1_asc,
                           'out_1_dsc': I_out_1_dsc}
           return dic_outliers

       
       
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
               featurenames = featurenames + ['score_day_' + str(i+1)]
           df.columns = ['enrollment_id'] + featurenames
           return df
           # output an array of length 30 or 31 including starting and ending time point of each day of the time span defined by timestart and timeend



if __name__ == '__main__':
       features = Features()
       #timedates = features.splitdays('2014-05-26', '2014-06-24')
       #print timedates
       
       
       #event_dict = dict(zip(['access','problem','page_close','navigate','video','discussion','wiki'],range(0,7)))
       
       #weight = 1
       #weight = array([1,1,1,1,1,1,1,0])
       #weight = array([70,500,500,0,500,50,50,200])
       #weight = weight / double(sum(weight)) * 100
       #enroll_data = read_csv('../data/train/enrollment_train.csv', header = 0)
       #log_data = read_csv('../data/train/log_train.csv', header = 0)
       #log_course_date = read_csv('../data/date.csv', header = 0)
       
       #enroll_data = enroll_data.loc[enroll_data['enrollment_id']<=20,:]
       #log_data = log_data.loc[log_data['enrollment_id']<=20, :]
       #df = features.dayfeaturemain(enroll_data, log_data, log_course_date, weight)
       
       #df_train = read_csv('../data/test/kddtestall_v1.csv', header = 0)
       #print df.head()
       #df.to_csv('../data/train/kddtrain_countbyday_v10.csv', header = True, index = False, index_label = False)
       #sumlist = ['problem','npobj', 'video', 'nvobj', 'access', 'naobj', 'wiki', 'nwobj','discussion', 'ndobj', 'navigate', 'nnobj', 'page_close', 'ncobj']
       #sumlist = ['npobj','nvobj', 'naobj', 'nwobj', 'ndobj', 'nnobj', 'ncobj']
       #sumlist = ['problem','video','access','wiki','discussion', 'navigate', 'page_close']
       #for i in range(df.shape[0]):
       #    print "enrollment_id: ", df.iloc[i,0]
       #    print "the sum of events and obj: ", sum(df.iloc[i,1:])
       #    print "indeed the sum should be: ", sum( df_train.loc[df_train['enrollment_id']== df.iloc[i,0],sumlist].values )



       trainfile = '../data/train/kddtrainall_v9.csv'
       #addtrainfile = '../data/train/kddtrain_countbyday_v10.csv'
       #testfile = '../data/test/kddtestall_v9.csv'
       #addtestfile = '../data/test/kddtest_countbyday.csv'
       
       #onkeys_train = 'enrollment_id'
       #onkeys_test = 'enrollment_id'
       #outfile_train = '../data/train/kddtrainall_v10.csv'
       #outfile_test = '../data/test/kddtestall_v9.csv'



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


       features.loadTrainF(trainfile)
       #features.loadTestF(testfile)

       #start_time = time.time()
      
       #features.pctbycourse('../data/train/kddtrainpct_tmp_v3.csv', '../data/test/kddtestpct_tmp_v3.csv', features_to_pct)
       #print("--- %s seconds ---" % (time.time() - start_time))

       #outlierfile = '../data/outlier.csv'
       #outfile = '../data/train/kddtrainall_v9_drop_v1.csv'
       #features.delOutliers(outlierfile, outfile)
       
       
       
       dic_outliers = features.getOutliersAscFromFile('../data/train/kddtrainall_v9_pct.csv', ['naobj','ncobj','ndate','ndobj','nnobj','npobj','nvobj','nwobj'],0.001)
       #dic_outliers = {'out_0_asc': I_out_0_asc,
       #                    'in_1_asc': I_in_1_asc,
       #                    'out_1_asc': I_out_1_asc,
       #                    'in_0_asc': I_in_0_asc}
       print ":::::::::::::::::::::::"
       print len(dic_outliers['out_0_asc'])
       print ":::::::::::::::::::::::"

       features.delOutliers(dic_outliers['out_0_asc'], '../data/train/kddtrainall_v9_drop_v2.csv')
       #print dic_outliers['out_1_asc']
       #dic_outliers = features.getOutliersAsc(['naobj'],0.05)
       #print dic_outliers
       #w = csv.writer(open('../data/outlier.csv', 'wb'))
       #for key, val in dic_outliers.items():
       #    w.writerow([key, val])

       #features.loadTestF(testfile)
       #features.loadaddTrainF(addtrainfile)
       #features.loadaddTestF(addtestfile)

       #features.mergeTrainF(outfile_train, onkeys_train)
       #features.mergeTestF(outfile_test, onkeys_test)

       #features.changeSubFormat('../data/pred/pred_azml_v0.csv', '../data/pred/pred_azml_v2.csv')
