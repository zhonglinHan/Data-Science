import csv
import pandas as pd
from pandas import *
from numpy import *
import datetime, time
from scipy import stats
from math import *
# not a script, but a collection of interactive queries to check the code is working.
df = read_csv('../data/train/kddtrainall_v9.csv', header = 0)
df_drop = read_csv('../data/train/kddtrainall_v9_drop_v1.csv', header = 0)

df.loc[110097,['naobj','ncobj','ndate','ndobj','nnobj','npobj','nvobj','nwobj','label']]
outlierfile = '../data/outlier.csv'

id_drop = list(set(df['enrollment_id']) - set(df_drop['enrollment_id']))
df.loc[df['enrollment_id']==id_drop[100],['naobj','ncobj','ndate','ndobj','nnobj','npobj','nvobj','nwobj','label']]


pred1 = read_csv('../data/pred_old/pred_dbn_v0.csv',header = None)
pred0 = read_csv('../data/pred/kddpred_xgb_v14.csv',header = None)
pred = DataFrame(columns = ['A','B'])
pred['A'] = pred0[0]
pred['B'] = 0.875 * pred0[1] + 0.125 * pred1[1]
pred.to_csv('../data/pred/pred_dbn_e_xgb_v1.csv', header = False, index = False, index_label = False)


pred0 = read_csv('../data/pred/kddpred_xgb_v14.csv',header = None)
pred1 = pred0
pred1[1][pred0[1]>0.9] = 0.99
pred1[1][pred0[1]<0.1] = 0.00001
pred1.to_csv('../data/pred/kddpred_xgb_v14_sharp_v0.csv', header = False, index = False, index_label = False)


dfm = read_csv('../data/train/dfm.csv',header = 0)
feat = ['ndate']
feat_val = dfm.loc[dfm['course_id']=='3VkHkmOtom3jM2wCu94xgzzu1d6Dn7or','ndate'].values.tolist()
vals = reshape([stats.percentileofscore(feat_val, a,'strict') for a in feat_val], [len(feat_val), ])
vals = vals/max(0.0000001,max(vals)) * 100
ind = where(dfm['course_id'] == '3VkHkmOtom3jM2wCu94xgzzu1d6Dn7or')
ind = ind[0]
ind_train = [i for i in ind if dfm['istrain'][i] == 1] - min(ind)
ind_test = [i for i in ind if dfm['istrain'][i] == 0] - min(ind)

vals_train = vals[ind_train]
vals_test = vals[ind_test]
print vals

df0 = read_csv('../data/train/kddtrainall_v9.csv',header = 0)
df = read_csv('../data/train/kddtrainpct_tmp_v3.csv',header = 0)
df1 = df.drop(['istrain'], axis = 1)
df1 = df1.sort(columns = 'enrollment_id', ascending = True)
df1 = df1.set_index([range(df1.shape[0])])
df1 = df1[df0.columns.values]
df1.to_csv('../data/train/kddtrainall_v9_pct.csv', header = True, index = False, index_label = False)


#outlier:
df = read_csv('../data/train/kddtrainall_v9.csv',header = 0)
I_out = []
for line in csv.reader(open('../data/outlier.csv','rb')):
    line =  map(int, filter(None,line))
    I_out = I_out + line
    print line

df_out = df.loc[I_out]
df_out.to_csv('../data/train/kddtrain_outlier_v0.csv', header = True, index = False, index_label = False)

# extreme cases:
df = read_csv('../data/train/kddtrainall_v9.csv',header = 0)
df = read_csv('../data/test/kddtestall_v9.csv',header = 0)
n = df.shape[0]
F_0 = ['AccCount', 'ProCount', 'PagCount','VidCount', 'DisCount', 'WikCount']
F_1 = ['NagCount']
I_ext = []
for i in range(n):
    if all(df.loc[i,F_0] == 0):
       print i, 'out of', n
       I_ext = I_ext + [i]
df_ext = df.loc[I_ext]

df_ext.to_csv('../data/train/kddtrain_ext_low.csv', header = True, index = False, index_label = False)
df_ext.to_csv('../data/test/kddtest_ext_low.csv', header = True, index = False, index_label = False)
df_drop_ext = df.drop(df.index[I_ext])
df_drop_ext.to_csv('../data/train/kddtrainall_v9_drop_ext_v0.csv', header = True, index = False, index_label = False)
df_drop_ext.to_csv('../data/test/kddtestall_v9_drop_ext_v0.csv', header = True, index = False, index_label = False)

########
df_ext_train = read_csv('../data/train/kddtrain_ext_low.csv',header = 0)
eid_ext = df_ext_train['enrollment_id']
df_log_train = read_csv('../data/train/log_train.csv', header = 0)
df_log_train_ext = df_log_train.loc[df_log_train['enrollment_id'].isin(eid_ext)]
#df_log_train_ext.set_index([range(df_log_train_ext.shape[0])])
df_log_train_ext = merge(df_log_train_ext, df_ext_train[['enrollment_id','label']], how = 'inner', on = 'enrollment_id')
df_log_train_ext.to_csv('../data/train/log_train_ext_v0.csv', header = True, index = False, index_label = False)
#df_log_train_ext = df_log_train_ext.sort(columns = 'enrollment_id', ascending = True)

# define new features: average events attempted per event objects
df = read_csv('../data/test/kddtestall_v9.csv',header = 0)
new_feat = ['acc_ob','pro_ob','pag_ob','nav_ob','vid_ob','dis_ob','wik_ob']
obj_feat = ['naobj', 'npobj','ncobj', 'nnobj', 'nvobj', 'ndobj', 'nwobj']
evt_feat = ['AccCount', 'ProCount', 'PagCount', 'NagCount', 'VidCount', 'DisCount', 'WikCount']
for i in range(len(new_feat)):
    print i
    x = df[evt_feat[i]]
    y = df[obj_feat[i]]
    y.loc[y<=0] = 1
    df[new_feat[i]] = x/y

df.to_csv('../data/test/kddtestall_v11.csv', header = True, index = False, index_label = False)

###### define entropy


def log_entropy(x):
	e = np.sum(np.log(np.array(range(1,np.sum(x)))))
	for i in x:
		e -= np.sum(np.log(np.array(range(1,i))))
	return e

df = read_csv('../data/test/kddtestall_v11.csv',header = 0)
entro_feat_1 = ['MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount']
entro_feat_2 = ['Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count']
entro_feat_3 = ['score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
df['etp_1'] = NaN
df['etp_2'] = NaN
df['etp_3'] = NaN
for i in range(df.shape[0]):
    print i, 'out of', df.shape[0]
    x1 = df.loc[i, entro_feat_1].values
    x2 = df.loc[i, entro_feat_2].values
    x3 = df.loc[i, entro_feat_3].values
    x1 = map(int, x1)
    x2 = map(int, x2)
    x3 = map(int, x3)
    df.loc[i, 'etp_1'] = log_entropy(x1)
    df.loc[i, 'etp_2'] = log_entropy(x2)
    df.loc[i, 'etp_3'] = log_entropy(x3)

df.to_csv('../data/test/kddtestall_v12.csv', header = True, index = False, index_label = False)


### merge predictions
pred1 = read_csv('../data/pred/kddpred_xgb_v14.csv', header = None)
pred2 = read_csv('../data/pred_old/pred_dbn_v0.csv', header = None)
pred_merge = zeros((pred1.shape[0],2))
for i in range(pred_merge.shape[0]):
    pred_merge[i,0] = pred1.iloc[i,0]
    if pred1.iloc[i,1] > 0.9 and pred2.iloc[i,1] > 0.5:
       print "::::::::", i
       pred_merge[i,1] = min(pred1.iloc[i,1], pred2.iloc[i,1])
    else:
       pred_merge[i,1] = pred1.iloc[i,1]
    #elif pred1.iloc[i,1] < 0.1 and pred2.iloc[i,1] < 0.1:
    #   print "////////", i
    #   pred_merge.iloc[i,1] = max(pred1.iloc[i,1], pred2.iloc[i,1])
pred_merge = DataFrame(pred_merge)
all(pred_merge.iloc[:,1] == pred1.iloc[:,1])
pred_merge.to_csv('../data/pred/kddpred_xgb14_e_dbn0.csv', header = False, index = False, index_label = False)

#the sum of day 25 to 30 score
#the sum of day 25 to 30 score divided by total score from day 1 to day 30
#min score day 25 to 30
#max score day 25 to 30
#max score day 25 to 30 / max score day 1 to day 30
#date duration divided by ndate

df = read_csv('../data/test/kddtestall_v12.csv',header = 0)
F_1 = ['score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
F_2 = ['score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
F_3 = ['ndate']
F_4 = ['date_duration']

df['sum_last_6_sc'] = NaN
df['psum_last_6_sc'] = NaN
df['min_last_6_sc'] = NaN
df['max_last_6_sc'] = NaN
df['pmax_last_6_sc'] = NaN
df['duration_per_date'] = NaN
for i in range(df.shape[0]):
    print i, 'out of', df.shape[0]
    x1 = df.loc[i, F_1].values
    x2 = df.loc[i, F_2].values
    df.loc[i,'sum_last_6_sc'] = sum(x2)
    df.loc[i,'min_last_6_sc'] = min(x2)
    df.loc[i,'max_last_6_sc'] = max(x2)
    df.loc[i,'psum_last_6_sc'] = double(sum(x2))/double(max(1,sum(x1)))
    df.loc[i,'pmax_last_6_sc'] = double(max(x2))/double(max(1,max(x1)))
    
    x3 = df.loc[i,F_3].values
    x4 = df.loc[i,F_4].values
    df.loc[i,'duration_per_date'] = double(x4)/double(max(1,x3))

df0 = read_csv('../data/train/kddtrainall_v15.csv', header = 0)
df.to_csv('../data/test/kddtestall_v15.csv', header = True, index = False, index_label = False)

df = read_csv('../data/train/kddtrainall_v16.csv', header = 0)
df =df.drop('label', axis = 1)
df_azml = read_csv('../data/test/kdd_azml_test.csv', header = 0)
feat_azml = ['event_trend', 'events_last_week', 'events_first_week',
       'events_second_last_week', 'weekly_avg', 'weekly_std',
       'max_weekly_count', 'min_weekly_count', 'first_event_month',
       'last_event_month', 'session_count_3hr', 'session_avg_3hr',
       'session_std_3hr', 'session_max_3hr', 'session_min_3hr',
       'quadratic_b', 'quadratic_c', 'session_count_1hr',
       'session_avg_1hr', 'sessioin_std_1hr', 'sessioin_max_1hr',
       'session_min_1hr', 'session_dur_avg_3hr', 'session_dur_std_3hr',
       'sessioin_dur_max_3hr', 'session_dur_min_3hr',
       'sessioin_dur_avg_1hr', 'session_dur_std_1hr',
       'session_dur_max_1hr', 'session_dur_min_1hr', 'MonCount',
       'TueCount', 'WedCount', 'ThuCount', 'FriCount', 'SatCount',
       'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count', 'Hr3Count',
       'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count', 'Hr8Count',
       'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count', 'Hr13Count',
       'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count', 'Hr18Count',
       'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count', 'Hr23Count',
       'AccCount', 'ProCount', 'PagCount', 'NagCount', 'VidCount',
       'DisCount', 'WikCount', 'BroCount', 'SerCount', 'prob_ct', 'vid_ct',
       'seq_ct', 'cha_ct', 'com_ct']
df = df.drop(feat_azml, axis = 1)

df.to_csv('../data/train/kddtrainall_v16_azml.csv', header = True, index = False, index_label = False)


# define features indicating lagging patterns:
df = read_csv('../data/train/kddtrainall_v15.csv',header = 0)
F_days = ['score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
#th = 6
m = df.shape[0]
#n = len(F_days)
df['min_lag'] = NaN
df['max_lag'] = NaN
df['mean_lag'] = NaN
df['std_lag'] = NaN
df['min_lag_6'] = NaN
df['max_lag_6'] = NaN
df['mean_lag_6'] = NaN
df['std_lag_6'] = NaN
df_30day = df[F_days]
for i in range(m):
    print i,'out of',m
    lags = []
    lags_6 = []
    #lag_count = 0
    x = df_30day.iloc[i,:].values
    I_pos = where(x>0)[0]
    j_prev = I_pos[0]
    for j in I_pos:
        if j - j_prev > 1:
           lag_count = j - j_prev - 1
           lags = lags + [lag_count]
           if lag_count > 6:
              lags_6 = lags_6 + [lag_count]
        j_prev = j

    if not lags:
        df.loc[i,'min_lag'] = 0
        df.loc[i,'max_lag'] = 0
        df.loc[i,'mean_lag'] = 0
        df.loc[i,'std_lag'] = 0
    else:
        df.loc[i,'min_lag'] = min(lags)
        df.loc[i,'max_lag'] = max(lags)
        df.loc[i,'mean_lag'] = mean(lags)
        df.loc[i,'std_lag'] = std(lags)

    
    if not lags_6:
        df.loc[i,'min_lag_6'] = 0
        df.loc[i,'max_lag_6'] = 0
        df.loc[i,'mean_lag_6'] = 0
        df.loc[i,'std_lag_6'] = 0
    else:
        df.loc[i,'min_lag_6'] = min(lags_6)
        df.loc[i,'max_lag_6'] = max(lags_6)
        df.loc[i,'mean_lag_6'] = mean(lags_6)
        df.loc[i,'std_lag_6'] = std(lags_6)
df.to_csv('../data/train/kddtrainall_v16.csv', header = True, index = False, index_label = False)














