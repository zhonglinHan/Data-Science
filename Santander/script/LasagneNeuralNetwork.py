# -*- coding: utf-8 -*-
"""
Created on Tue Mar 7 

@author: Ouranos
"""


import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid



class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()       
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


        
def pdFillNAN(df, strategy = "mean"):
    #Fills empty values with either the mean value of each feature, or an indicated number
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]

#Drop columns with 0 variation
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

print(len(remove))
train.drop(labels = remove, axis=1, inplace=True)
test.drop(labels = remove, axis=1, inplace=True)

print(train.columns)
#print(train.describe())

#Drop target, ID and some other columns...
labels = train["TARGET"]
trainId = train["ID"]
testId = test["ID"]

train.drop(labels = ["ID","TARGET","ind_var29_0","ind_var13_medio","ind_var18","ind_var26","ind_var25","ind_var32","ind_var34","ind_var37","ind_var39","num_var29_0","saldo_var29","saldo_medio_var13_medio_ult1","delta_num_reemb_var13_1y3",
                     "delta_num_reemb_var17_1y3","delta_num_reemb_var33_1y3","delta_num_trasp_var17_in_1y3","delta_num_trasp_var17_out_1y3","delta_num_trasp_var33_in_1y3","delta_num_trasp_var33_out_1y3"], axis = 1, inplace = True)
test.drop(labels = ["ID","ind_var29_0","ind_var13_medio","ind_var18","ind_var26","ind_var25","ind_var32","ind_var34","ind_var37","ind_var39","num_var29_0","saldo_var29","saldo_medio_var13_medio_ult1","delta_num_reemb_var13_1y3",
                     "delta_num_reemb_var17_1y3","delta_num_reemb_var33_1y3","delta_num_trasp_var17_in_1y3","delta_num_trasp_var17_out_1y3","delta_num_trasp_var33_in_1y3","delta_num_trasp_var33_out_1y3"], axis = 1, inplace = True)


print ("Filling in missing values...")
fillNANStrategy = "mean"
train = pdFillNAN(train, fillNANStrategy)
test = pdFillNAN(test, fillNANStrategy)


print ("Scaling...")
train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)


train = np.asarray(train, dtype=np.float64)        
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)

net = NeuralNet(
    layers=[  
        ('input', InputLayer),
        ('dropout0', DropoutLayer),
        ('hidden1', DenseLayer),
        ('hidden2', DenseLayer),
        ('output', DenseLayer),
        ],

    input_shape=(None, len(train[1])),
    dropout0_p=0.05,
    hidden1_num_units=100,
    hidden1_W=Uniform(), 
    hidden2_num_units=100,
    hidden2_W=Uniform(),

    output_nonlinearity=sigmoid,
    output_num_units=1, 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(np.float32(0.001)),
    update_momentum=theano.shared(np.float32(0.9)),    
    # Decay the learning rate
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
                       AdjustVariable('update_momentum', start=0.9, stop=0.99),
                       ],
    regression=True,
    y_tensor_type = T.imatrix,                   
    objective_loss_function = binary_crossentropy,
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=40, 
    eval_size=0.0,
    verbose=2,
    )


seednumber=1235
np.random.seed(seednumber)
net.fit(train, labels)


preds = net.predict_proba(test)[:,0] 


submission = pd.read_csv('../input/sample_submission.csv')
submission["TARGET"] = preds
submission.to_csv('Lasagne_bench.csv', index=False)
