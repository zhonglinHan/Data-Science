import re
import csv
import collections

dictionary = open('../bids.csv')
trainset = open('../train.csv')

count =0
for text in dictionary:
    if text.split(',')[1] == '91a3c57b13234af24875c56fb7e2b2f4rb56a':
        count +=1

print "91a3c57b13234af24875c56fb7e2b2f4rb56a","\t",count
 
