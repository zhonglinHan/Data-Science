from numpy import *
from csv import *
from pandas import *
from scipy import stats
from math import *

########store id and label into label[id,label]#######################################
    
with open('truth_train.csv') as trf:
      label = dict(reader(trf));

################## pandas #############################################################

trainset = open('kddtrain_hour_perday.csv', 'w');
trainset.write("enrollment_id,mean_hour,std_hour,skew_hour,label\n");

df = DataFrame.from_csv('sep_log_train.csv',index_col=False)
idx_dict = df['enrollment_id'].value_counts().to_dict()

for key in idx_dict:
    selt = df[df['enrollment_id']==key]
    datedict = selt.date.value_counts().to_dict(); 
    datelist = datedict.keys();
    dhour = [];
    for datekey in datelist:
        perday = df[(df.enrollment_id==key) & (df.date==datekey)];
        delta = max(to_datetime(perday.time)) - min(to_datetime(perday.time));
        dhour.append(round(delta/np.timedelta64(1,'h'),3));
        
    lab = label.get(str(key));
    dhour = asarray(dhour);
    mean_h = round(dhour.mean(),3);
    std_h = round(dhour.std(),3);
    skew_h = round(stats.moment(dhour,3)/(pow(std_h,3)+pow(10,-10)),3);
    kurt_h = round(stats.moment(dhour,4)/(pow(std_h,4)+pow(10,-10))-3,3);
    trainset.write(str(key)+","+str(mean_h)+","+str(std_h)+","+str(skew_h)+","+str(kurt_h)+","+str(lab)+"\n");
    print key, mean_h,std_h, skew_h, kurt_h;
