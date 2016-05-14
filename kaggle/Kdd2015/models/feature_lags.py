
import csv
import pandas as pd
from pandas import *
from numpy import *
import datetime, time
from scipy import stats
from math import *


#df = read_csv('../data/train/kddtrainall_v15.csv',header = 0)
df = read_csv('../data/test/kddtestall_v15.csv',header = 0)
F_days = ['score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
#df = df[0:5000]
#th = 6
m = df.shape[0]
#n = len(F_days)
df['min_lag'] = NaN
df['max_lag'] = NaN
df['mean_lag'] = NaN
df['std_lag'] = NaN
#df['min_lag_14'] = NaN
#df['max_lag_14'] = NaN
#df['mean_lag_14'] = NaN
#df['std_lag_14'] = NaN
df_30day = df[F_days]
for i in range(m):
    print i,'out of',m
    lags = []
    #lags_6 = []
    #lag_count = 0
    x = df_30day.iloc[i,:].values
    I_pos = where(x>0)[0]
    j_prev = I_pos[0]
    for j in I_pos:
        if j - j_prev > 1:
           lag_count = j - j_prev - 1
           lags = lags + [lag_count]
           #if lag_count > 6:
              #lags_6 = lags_6 + [lag_count]
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

    #if not lags_6:
    #    df.loc[i,'min_lag_14'] = 0
    #    df.loc[i,'max_lag_14'] = 0
    #    df.loc[i,'mean_lag_14'] = 0
    #    df.loc[i,'std_lag_14'] = 0
    #else:
    #    df.loc[i,'min_lag_14'] = min(lags_6)
    #    df.loc[i,'max_lag_14'] = max(lags_6)
    #    df.loc[i,'mean_lag_14'] = mean(lags_6)
    #    df.loc[i,'std_lag_14'] = std(lags_6)
#df.to_csv('../data/train/kddtrainall_v16.csv', header = True, index = False, index_label = False)
df.to_csv('../data/test/kddtestall_v16.csv', header = True, index = False, index_label = False)
#df.to_csv('../data/train/kddtrainall_v16_toy.csv', header = True, index = False, index_label = False)




