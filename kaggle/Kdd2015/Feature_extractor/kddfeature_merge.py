from numpy import *
from csv import *
from pandas import *

### merge ###
df_train_ori = read_csv('../data/train/enrollment_train.csv',header = 0)
#df_train_0 = read_csv('../data/train/kddtrain_v0.csv',header = 0)
df_train_1 = read_csv('../data/train/kddtrain_v1.csv',header = 0)
df_train_2 = read_csv('../data/train/kddtrain_v2.csv',header = 0)
df_train_3 = read_csv('../data/train/kddtrain_v3.csv',header = 0)
df_train_mms = read_csv('../data/train/kddmomstrain_v1.csv',header = 0)
df_train_3 = df_train_3.drop(['ndate'],axis = 1)
df_train = merge(df_train_1,df_train_2, how = 'inner', on = ['enrollment_id','label'])
df_train = merge(df_train,df_train_3, how = 'inner', on = ['enrollment_id','label'])
df_train = merge(df_train,df_train_mms, how = 'inner', on = ['enrollment_id','label'])

featureList = ['ndate', 'date_duration', 'study_days',
               'problem', 'npobj', 'video', 'nvobj',
               'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj',
               'navigate', 'nnobj', 'page_close', 'ncobj',
               'std_events', 'skew_events',
               'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
               'skew_ratio', 'kurt_ratio']

df_train = df_train[['enrollment_id'] + featureList + ['label']]
df_train.to_csv('../data/train/kddtrainall_v1.csv',header = True, index = False, index_label = False)


df_test_ori = read_csv('../data/test/enrollment_test.csv',header = 0)
df_test_0 = read_csv('../data/test/kddtest_v0.csv',header = 0)
df_test_1 = read_csv('../data/test/kddtest_v1.csv',header = 0)
df_test_2 = read_csv('../data/test/kddtest_v2.csv',header = 0)

df_test = merge(df_test_1,df_test_2, how = 'inner', on = ['enrollment_id'])
df_test = df_test[['enrollment_id','ndate','date_duration','study_days'] + list(df_test_1.columns.values[3:])]
df_test.to_csv('../data/test/kddtestall_v0.csv',header = True, index = False, index_label = False)



#print df_train_0.columns.values
#print df_train_2.columns.values
#print df_train_3.columns.values
#print df_train.columns.values
#print df_train_0.shape
#print df_train.shape
#print df_train.head()
#print df_train_3.head()
#print df_train.head()
# df = read_csv('../data/train/kddtrainall_v0.csv', header = 0)
# df1 = read_csv('../data/train/kddtrain_v1.csv', header = 0)
# df.shape()

