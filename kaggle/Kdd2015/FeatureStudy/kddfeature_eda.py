import csv
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
#from matplotlib.pyplot import *
#from pylab import *

class FeatureEda:

    def __init__(self):
          self.feature = []
    
    def loadData(self, trainfile):
          self.data = df_train = read_csv(trainfile, header = 0)

    def loadFeature(self, feature):
          df_train = self.data

          df0 = df_train[df_train['label']==0]
          df1 = df_train[df_train['label']==1]
          feat0 = np.array(df0[feature])
          feat1 = np.array(df1[feature])
          d = dict(class_0 = np.array(feat0), class_1 = np.array(feat1))
          self.feature = DataFrame(dict([(k,Series(v)) for k,v in d.iteritems()]))
          #print self.feature['class_0'].dropna().shape
          #print self.feature['class_1'].dropna().shape
          #print self.feature.shape

    def boxplotFeature(self,savefig,feature):
          fig = plt.figure(1)
          fig.suptitle(feature)
          _=self.feature.boxplot(return_type='axes')
          fig.savefig(savefig)
          plt.close(fig)

    def histFeature(self, savefig,feature):
          fig = plt.figure(2)
          fig.suptitle(feature)
          ax = plt.subplot(211)
          ax.set_title("Class_0")
          plt.hist(self.feature['class_0'].dropna(),40,normed = 0)
          ax=plt.subplot(212)
          ax.set_title("Class_1")
          plt.hist(self.feature['class_1'].dropna(),40,normed = 0)
          #_=np.histogram(np.array(self.feature['class_1'].dropna()), bins = 10, density = False)
          fig.savefig(savefig)
          plt.close(fig)


if __name__ == '__main__':
          featureList = ['ndate', 'date_duration', 'study_days',
                   'problem', 'npobj', 'video', 'nvobj',
                   'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj',
                   'navigate', 'nnobj', 'page_close', 'ncobj',
                   'std_events', 'skew_events',
                   'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
                   'skew_ratio', 'kurt_ratio']
          trainfile = '../data/train/kddtrainall_v1.csv'
          eda = FeatureEda()
          eda.loadData(trainfile)
          
          count = 0
          for feature in featureList:
              print count
              #feature = 'page_close'
              savefig = '../data/eda/box_' + feature + '.png'
              eda.loadFeature(feature)
              eda.boxplotFeature(savefig,feature)
              count = count + 1
                  
          count = 0
          for feature in featureList:
              print count
              savefig = '../data/eda/hist_' + feature + '.png'
              eda.loadFeature(feature)
              eda.histFeature(savefig,feature)
              count = count + 1








