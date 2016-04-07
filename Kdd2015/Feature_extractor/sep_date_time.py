from numpy import *
from csv import *
from pandas import *



newlog = open('sep_log_train.csv', 'w');

newlog.write("enrollment_id,date,time,source,event,object\n");

############################logtrain->newlog with "date" and "time"###################

logtrain = open('log_train.csv', 'r');
next(logtrain); ##skip first line##
for line in logtrain:
    item = line.strip().split(',');
    strdate, strtime = item[1].split('T')[0], item[1].split('T')[1];
    newlog.write(item[0]+","+strdate+","+strtime+","+item[2]+","+item[3]+","+item[4]+"\n")


