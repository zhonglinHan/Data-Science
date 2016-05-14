# example of program that calculates the average degree of hashtags
# This code calculates the distinct hashtags for each tweet
#!/usr/bin/python
import sys
import json
import string
import datetime

class Degree_cal:
    """Calculation of Average Degree """
    def __init__(self):
        """define a array to store input stream within a minute"""
        self.stream = []
        self.mintime = None
        self.deg = 0.0

    def addItem(self, time, tags):
        """ Add current time and hashtags into the stream, keep one minute window"""
        if self.mintime : 
            self.mintime = min(self.mintime, time)
        else :
            self.mintime = time
        self.stream.append([time, tags])
        for j in range(len(self.stream)):
            timeInt = self.interval(self.stream[j][0])
            if timeInt >= 60.0:
                self.stream.remove(self.stream[j])
                       

    def interval(self, time):
        """ use datetime to process the unicode timestamp and find interval
            between current time to the earliest stream element """
        ta = datetime.datetime.strptime(self.mintime, "%a %b %d %H:%M:%S +0000 %Y")
        tb = datetime.datetime.strptime(time, "%a %b %d %H:%M:%S +0000 %Y")
        tdelta = tb - ta
        return tdelta.total_seconds()

    def degree_cal(self):
        import numpy as np
        """ statistic unique tags in the stream as vertex
            find total paris of tags as neigbors for number of edges
            If total vertex is zero, return none """
        paris, vex = 0.0, 0.0
        tags_column = []
        degs = [0.0]
            
        for i in range(len(self.stream)):
            tags_column = tags_column + [tag.lower() for tag in self.stream[i][1]]
            if len(set(tags_column)) == len(tags_column):
                """
                no shared vex
                """
                paris = len(self.stream[i][1])
                vex = len(set(self.stream[i][1]))
                if vex <= 1.0 :
                    degs.append(0.0)
                else : 
                    if paris == 2.0:
                        degs.append(paris/(vex* 1.0))
                    else :
                        degs.append(paris/(vex*1.0 - 1.0))
            else :
                if degs :
                    degs.remove(degs[len(degs) - 1])
                else :
                    degs = [0.0]
                paris = len(tags_column)
                vex = len(set(tags_column))
                if vex <= 1.0 :
                    degs.append(0.0)
                else : 
                    if paris == 2.0:
                        degs.append(paris/(vex* 1.0))
                    else :
                        degs.append(paris/(vex*1.0 - 1.0))
        
        self.deg = sum(degs)    
        return '{0:.2f}'.format(self.deg)


def tweet_parse(line):
    tweet = json.loads(line)
    time = tweet['created_at']
    """Avoid retweeted status multi texts"""
    if 'retweeted_status' in tweet:
        text = tweet['retweeted_status']['text']
    else:
        text = tweet['text']
    hashtags = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    uniquetags = list(set(hashtags))
    return [time, uniquetags]    

    
if __name__ == "__main__":
    fin  = open(sys.argv[1], 'r')
    fout = open(sys.argv[2], 'w')
    curdeg = Degree_cal();
    for line in fin:
        try:
            [time, hashtags] = tweet_parse(line)
            curdeg.addItem(time, hashtags)
            fout.write(curdeg.degree_cal()+'\n')
        except KeyError:
            """ if current tweet is not well-formated with created_at
                keep it as current average of degree """
            fout.write(curdeg.degree_cal()+'\n')
