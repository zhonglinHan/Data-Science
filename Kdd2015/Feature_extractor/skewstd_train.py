#########fine cosine value of each one to (0,0,0,0,0.....,absolute dropout vector) 

from numpy import *
from csv import *
from pandas import *
from scipy import stats
from math import *

momtrain = open('kddmomstrain_v1.csv','w');
trainset = open('../../Data/trainSet/kddtrain_v1.csv','r');
momtrain.write("enrollment_id,lable,std_events,skew_events,kurt_events,std_obj,skew_obj,kurt_obj,std_ratio,skew_ration,kurt_ratio\n")
next(trainset);

for line in trainset:
    items = line.strip().split(','); #items[0] id
    ent = [];
    obj = [];
    rio =[];
    
    for i in range(2,15,2):
        ent.append([float(items[i])]);
        obj.append([float(items[i+1])]);        
        rio.append((float(items[i+1])+pow(10,-10))/(float(items[i])+pow(10,-10)));##obj/event 

    std_ent = round(std(ent),3);
    skew_ent = round(stats.moment(ent,3)/pow(std_ent,3),3);
    kurt_ent = round(stats.moment(ent,4)/pow(std_ent,4)-3,3);
    std_obj = round(std(obj),3);
    skew_obj = round(stats.moment(obj,3)/pow(std_obj,3),3);
    kurt_obj = round(stats.moment(obj,4)/pow(std_obj,4)-3,3);
    std_rio = round(std(rio),3);
    skew_rio = round(stats.moment(rio,3)/(pow(std_rio,3)+pow(10,-10)),3);
    kurt_rio = round(stats.moment(rio,4)/(pow(std_rio,4)+pow(10,-10))-3,3);
    
    momtrain.write(str(items[0])+","+str(items[16])+","+str(std_ent)+","+str(skew_ent)+","+str(kurt_ent)+","+str(std_obj)+","+str(skew_obj)+","+str(kurt_obj)+","+str(std_rio)+","+str(skew_rio)+","+str(kurt_rio)+"\n")
    


