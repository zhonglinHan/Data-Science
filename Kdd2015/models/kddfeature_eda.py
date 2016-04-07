import csv
import numpy as np
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
#from matplotlib.pyplot import *
#from pylab import *

class FeatureEda:

    def __init__(self):
          self.feature = []
    
    def loadData(self, trainfile):
          df_train = read_csv(trainfile, header = 0)
          self.data = df_train

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
          
    def transFeature(self, featuretrans, transfile):
          df = self.data
          for feature in featuretrans:
              df_feat = df[feature]
              plt = hist(df_feat, bins = 20)
              n = plt[0]
              n = n / sum(n)
              bins = plt[1]
              print feature
              count = 0
              for i in range(len(n)):
                  print count
                  if i<len(n)-1:
                     df.loc[(df_feat>=bins[i]) & (df_feat<bins[i+1]),feature] = n[i]
                  else:
                     df.loc[(df_feat>=bins[i]) & (df_feat<=bins[i+1]),feature] = n[i]
                  count = count + 1
          df.to_csv(transfile, header = True, index = False, index_label = False)

          

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
          featureList = ['min_lag', 'max_lag', 'mean_lag',
          'std_lag']
          trainfile = '../data/train/kddtrainall_v16.csv'
          #transfile = '../data/train/kddtrainall_v1_trans.csv'
          eda = FeatureEda()
          eda.loadData(trainfile)
#eda.transFeature(featureList,transfile)
          #count = 0
          #for feature in featureList:
          #    print count
          #    #feature = 'page_close'
          #    savefig = '../data/eda_train/box_' + feature + '.png'
          #    eda.loadFeature(feature)
          #    eda.boxplotFeature(savefig,feature)
          #    count = count + 1
                  
          count = 0
          for feature in featureList:
              print count
              savefig = '../data/eda_train/hist_' + feature + '.png'
              eda.loadFeature(feature)
              eda.histFeature(savefig,feature)
              count = count + 1







