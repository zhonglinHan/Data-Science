import re
import csv
import collections
import numpy
import scipy, nltk
from numpy import *
from collections import Counter

trainset = open('../train.csv')
grades = Counter()
diction = []
outf=open('train_feature_v0.csv','w')
outf.write("bidder,total_auction,auction_types,max_auction, mean_auction, std_auction, Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")

class ReturnValue:
      def __init__(self, bidnum, auctype, auctmax, auctmean,auctstd,aucfqmean,aucfqstd, mechtype, devicenum, devmax, devmean, devstd, devfqmean, devfqstd,tmean, tstd, countrynum, ipnum, ipmax, ipmean, ipstd, ipfqmean,ipfqstd,urlnumber):
            self.bid = bidnum
            self.auty = auctype
            self.aumax = auctmax 
            self.aumean = auctmean
            self.austd = auctstd
            self.aucfqmean = aucfqmean
            self.aucfqstd = aucfqstd
            self.mechty = mechtype
            self.devnum = devicenum
            self.devmax = devmax
            self.devmean = devmean
            self.devstd = devstd
            self.devfqmean = devfqmean
            self.devfqstd = devfqstd
            self.country = countrynum
            self.time_mean = tmean
            self.time_std = tstd
            self.ipty = ipnum
            self.ipmax = ipmax
            self.ipmean = ipmean
            self.ipstd = ipstd
            self.ipfqmean = ipfqmean
            self.ipfqstd = ipfqstd
            self.urlnum =urlnumber

def findcount(strline):
    bidcount = 0
    aucstat = []
    merchstat = []
    devicstat = [] 
    timestat = []
    countrystat = []
    ipstat = []
    urlstat = []
    dictionary = open('../bids.csv')
    for text in dictionary: 
        if text.split(',')[1] == strline:
            bidcount +=1
            aucstat.append(text.split(',')[2]) 
            merchstat.append(text.split(',')[3])
            devicstat.append(text.split(',')[4])
            timestat.append(text.split(',')[5])
            countrystat.append(text.split(',')[6])
            ipstat.append(text.split(',')[7])
            urlstat.append(text.split(',')[8]) 
    if bidcount!=0:                           
        tim_mea, tim_std = freq(timestat)
        auc_mea, auc_std = listCount(aucstat)
        dev_mea, dev_std = listCount(devicstat)
        ip_mea, ip_std = listCount(ipstat)
        ipfq_mea, ipfq_std = freqCount(timestat, ipstat)
        devfq_mea, devfq_std = freqCount(timestat, devicstat)
        aucfq_mea, aucfq_std = freqCount(timestat, aucstat)
        return ReturnValue(bidcount,len(set(aucstat)),Counter(aucstat).most_common(1)[0][1], auc_mea, auc_std, aucfq_mea, aucfq_std , len(set(merchstat)), len(set(devicstat)), Counter(devicstat).most_common(1)[0][1], dev_mea, dev_std, devfq_mea, devfq_std, tim_mea, tim_std, len(set(countrystat)),len(set(ipstat)), Counter(ipstat).most_common(1)[0][1],ip_mea, ip_std,ipfq_mea, ipfq_std,len(set(urlstat)))
    else:
        return ReturnValue(bidcount, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0)
    dictionary.close()

def freq(thelist):
    tseries = map(int, thelist)
    delta = []
    if len(tseries)>1 and len(tseries):
        for i in range(1,len(tseries)):
              delta.append((tseries[i]-tseries[i-1])/526315700)
    else:
       delta.append(0.0)   
    return round(mean(delta),2), round(std(delta),2)

def listCount(thelist):
     Count=[]
     for item in set(thelist):
          Count.append(thelist.count(item))
     return round(mean(Count),2), round(std(Count),2)
     
def freqCount(timelist, featlist):
    idx = 0
    time = []
    time.append(timelist[0])
    if len(featlist)!=1:
       for idx in range(1, len(featlist)):
           if featlist[idx-1]!=featlist[idx]:
               time.append(timelist[idx])
    mean, std = freq(time)
    return mean, std
          
         
     
      
bidder = []
classtype = []

for line in trainset:
    bidder.append(line.split(',')[0]) 
    classtype.append(line.split(',')[3]) 

for i in range(0,len(bidder)): 
#for i in range(113,118):
       Result = findcount(bidder[i])
       print i, bidder[i]," ", Result.auty, Result.aumean, Result.austd, Result.devnum, Result.devmax, Result.devmean, Result.devstd, Result.ipty, Result.ipmax, Result.ipmean, Result.ipstd, classtype[i]
       outf.write(str(bidder[i])+","+str(Result.bid)+","+str(Result.auty)+","+str(Result.aumax)+","+str(Result.aumean)+","+str(Result.austd)+","+str(Result.aucfqmean)+","+str(Result.aucfqstd)+","+str(Result.mechty)+","+str(Result.devnum)+","+str(Result.devmax)+","+str(Result.devmean)+","+str(Result.devstd)+","+str(Result.devfqmean)+","+str(Result.devfqstd)+","+str(Result.time_mean)+","+str(Result.time_std)+","+str(Result.country)+","+str(Result.ipty)+","+str(Result.ipmax)+","+str(Result.ipmean)+","+str(Result.ipstd)+","+str(Result.ipfqmean)+","+str(Result.ipfqstd)+","+str(Result.urlnum)+","+str(classtype[i]))