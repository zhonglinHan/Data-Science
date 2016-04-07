from numpy import *
from csv import *
from pandas import *

########store id and label into label[id,label]#######################################
    
with open('truth_train.csv') as trf:
      label = dict(reader(trf));

################## pandas #############################################################
################# find out number of days during online course#########################

trainset = open('kddtrain_v2.csv', 'w');
trainset.write("enrollment_id,date_duration,label\n");

df = DataFrame.from_csv('log_train.csv',index_col=False)
idx_dict = df['enrollment_id'].value_counts().to_dict()

for key in idx_dict:
    selt = df[df['enrollment_id']==key]        
    max_datetime = max(to_datetime(selt.time));
    min_datetime = min(to_datetime(selt.time));
    delta_date = round((max_datetime - min_datetime)/np.timedelta64(1,'D'),3)
    lab = label.get(str(key));    
    trainset.write(str(key)+","+str(delta_date)+","+str(lab)+"\n")  
    print key  
