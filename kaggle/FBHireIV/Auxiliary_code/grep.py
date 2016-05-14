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
outf=open('train_reset.csv','w')
outf.write("bidder,total_auction,auction_types,total,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")

class ReturnValue:
      def __init__(self, bidnum, auctype, auctmax, mechtype, devicenum, tmean, tstd, countrynum, ipnum, ipmaxium,urlnumber):
            self.bid = bidnum
            self.auty = auctype
            self.aumax = auctmax 
            self.mechty = mechtype
            self.devnum = devicenum
            self.country = countrynum
            self.time_mean = tmean
            self.time_std = tstd
            self.ipty = ipnum
            self.ipmax = ipmaxium
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
        return ReturnValue(bidcount, len(set(aucstat)), Counter(aucstat).most_common(1)[0][1], len(set(merchstat)), len(set(devicstat)), tim_mea, tim_std, len(set(countrystat)),len(set(ipstat)), Counter(ipstat).most_common(1)[0][1],len(set(urlstat)))
    else:
        return ReturnValue(bidcount, 0, 0, 0, 0, 0.0, 0.0, 0,0, 0,0)
    dictionary.close()

def freq(thelist):
    tseries = map(int, thelist)
    delta = []
    if len(tseries)>1 and len(tseries):
        for i in range(1,len(tseries)):
              delta.append((tseries[i]-tseries[i-1])/526315700)
    else:
       delta.append(1.0)   
    return round(mean(delta),2), round(std(delta),2)
        
    
     
      
bidder = []
classtype = []

for line in trainset:
    bidder.append(line.split(',')[0]) 
    classtype.append(line.split(',')[3]) 

for i in range(0,len(bidder)): 
#for i in range(48,50):
       Result = findcount(bidder[i])
       print i, bidder[i], Result.bid, Result.auty, Result.aumax, Result.ipty, Result.ipmax
       outf.write(str(bidder[i])+","+str(Result.bid)+","+str(Result.auty)+","+str(Result.aumax)+","+str(Result.mechty)+","+str(Result.devnum)+","+str(Result.time_mean)+","+str(Result.time_std)+","+str(Result.country)+","+str(Result.ipty)+","+str(Result.ipmax)+","+str(Result.urlnum)+","+str(classtype[i]))