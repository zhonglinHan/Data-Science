from numpy import *
from csv import *
from pandas import *

########store id and label into label[id,label]#######################################
    
with open('truth_train.csv') as trf:
      label = dict(reader(trf));

################## pandas #############################################################

trainset = open('kddtrain_v1.csv', 'w');
trainset.write("enrollment_id,ndate,problem,npobj,video,nvobj,access,naobj,wiki,nwobj,discussion,ndobj,navigate,nnobj,page_close,ncobj,label\n");

df = DataFrame.from_csv('sep_log_train.csv',index_col=False)
idx_dict = df['enrollment_id'].value_counts().to_dict()
eventlist = ["problem", "video","access", "wiki","discussion", "navigate", "page_close"]

for key in idx_dict:
    selt = df[df['enrollment_id']==key]
    event_stat =  selt['event'].value_counts().to_dict()
    strevent ="";
    for i in range(7):
        evname = eventlist[i];
        val = str(event_stat.get(evname));
        selv = df[(df.event==evname) & (df.enrollment_id==key)];
        obj = str(selv['object'].nunique())
        if val == "None": val = str(0);
        strevent = strevent+val+","+obj+",";

    ndate =  selt['date'].nunique();
    lab = label.get(str(key));    
    trainset.write(str(key)+","+str(ndate)+","+strevent+str(lab)+"\n")    
