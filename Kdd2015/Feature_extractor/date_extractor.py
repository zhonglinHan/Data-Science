from numpy import *
from csv import *
from pandas import *


################## pandas #############################################################

trainset = open('kddtrain_v5.csv', 'w');
trainset.write("enrollment_id,start,end\n");

df = DataFrame.from_csv('new_log_train.csv',index_col=False)
idx_dict = df['enrollment_id'].value_counts().to_dict()

for key in idx_dict:
    selt = df[df['enrollment_id']==key]
    datedict = selt.date.value_counts().to_dict(); 
    datelist = datedict.keys();
    end = max(to_datetime(selt.date))
    sta = min(to_datetime(selt.date));
        
    trainset.write(str(key)+","+str(sta.date())+","+str(end.date())+"\n");
    print key   