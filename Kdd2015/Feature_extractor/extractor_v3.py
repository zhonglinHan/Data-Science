from numpy import *
from csv import *
from pandas import *

########store id and label into label[id,label]#######################################
    
with open('truth_train.csv') as trf:
      label = dict(reader(trf));

################## pandas #############################################################

trainset = open('kddtrain_v3.csv', 'w');
trainset.write("enrollment_id,ndate,study_days,label\n");

df = DataFrame.from_csv('new_log_train.csv',index_col=False)
idx_dict = df['enrollment_id'].value_counts().to_dict()

for key in idx_dict:
    selt = df[df['enrollment_id']==key]
    event_stat =  selt['event'].value_counts().to_dict();
    datedict = selt.date.value_counts().to_dict(); 
    datelist = datedict.keys();
    dhour = 0;
    for datekey in datelist:
        perday = df[(df.enrollment_id==key) & (df.date==datekey)];
        delta = max(to_datetime(perday.time)) - min(to_datetime(perday.time));
        dhour = dhour + round(delta/np.timedelta64(1,'D'),3)
        
    ndate =  selt['date'].nunique();
    lab = label.get(str(key));    
    trainset.write(str(key)+","+str(ndate)+","+str(dhour)+","+str(lab)+"\n")    
