import time, datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nolearn.dbn import DBN

def calculate_distance(distances):
    return distances ** -2

def prepare_data(df):
    """
    Feature engineering and computation of the grid.
    """
    minute = df.time%60
    df['hour'] = df['time']//60
    df['weekday'] = df['hour']//24
    df['month'] = df['weekday']//30
    df['year'] = (df['weekday']//365+1)*10.0
    df['hour'] = (df.hour%24+1+(df.time%60)/60.0)*4.0
    df['weekday'] = (df['weekday']%7+1)*3.1
    df['month'] = (df['month']%12+1)*2.1
    df['accuracy'] = np.log10(df['accuracy'])*10.0
    df.drop(['time'], axis=1, inplace=True)
    return df

def augmented_train(df_train):
    pd.options.mode.chained_assignment = None
    add_data = df_train[df_train.hour<6]
    add_data.hour = add_data.hour+96
    df_train = df_train.append(add_data)
    add_data = df_train[df_train.hour>98]
    add_data.hour = add_data.hour-96
    df_train = df_train.append(add_data)
    return df_train

def process_grid(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.5
    y_step = 0.25
    
    x_border_augment = 0.03  
    y_border_augment = 0.015
    
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4) 
        if x_max == size:
            x_max = x_max + 0.001
            
        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)   
            if y_max == size:
                y_max = y_max + 0.001
                
            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]
            
            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test)

            #Updating predictions
            preds[row_ids] = pred_labels
    
    return preds
def process_one_cell_dbn(df_cell_train, df_cell_test):
    
    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 8).values
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    row_ids = df_cell_test.index
    
    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= 500.0
    df_cell_train.loc[:,'y'] *= 1000.0
    df_cell_test.loc[:,'x'] *= 500.0
    df_cell_test.loc[:,'y'] *= 1000.0
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    #best_k=np.floor(np.sqrt(y.size)/5.1282)
    X = df_cell_train.drop(['place_id'], axis=1).values
    X_test = df_cell_test.values

    #Applying the classifier
    clf = DBN([-1, 256, 128, 64, 32, -1], learn_rates=0.01, epochs=50)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    return pred_labels, row_ids

def map_at_three (df, df_predictions):
    # DataFrame to calculate the MAP@3 score
    df_score = pd.DataFrame(np.zeros(len(df)))

    # Calculate the AP@3
    df_score.loc[df['place_id'] == df_predictions['l3']] = 1/3
    df_score.loc[df['place_id'] == df_predictions['l2']] = 1/2
    df_score.loc[df['place_id'] == df_predictions['l1']] = 1

    # Calculate the MAP@3
    score = df_score[0].mean()

    return score

def process_grid_cv(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.5
    y_step = 0.25
    
    x_border_augment = 0.03  
    y_border_augment = 0.015
    
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4) 
        if x_max == size:
            x_max = x_max + 0.001
            
        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)   
            if y_max == size:
                y_max = y_max + 0.001
                
            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]
            
            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell_dbn(df_cell_train, df_cell_test)
            #pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test)

            #Updating predictions
            preds[row_ids] = pred_labels
    
    #print('Generating submission file')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])  
    
    #Concatenating the 3 predictions for each sample
    #Writting to csv
    #df_aux.name = 'place_id'
    return df_aux

def scoreValid(dtrain, dtest):
    #dtrain, dtest = train_test_split(dat_train, test_size=0.3)	
    dtrue = dtest
    dtest = dtest.drop("place_id", 1, )	
    train_rowIDs = np.arange(dtrain.shape[0])
    test_rowIDs = np.arange(dtest.shape[0])	
    dtrain['row_id'] = train_rowIDs
    dtest['row_id'] = test_rowIDs	

    dtrain = dtrain.set_index('row_id')
    dtest = dtest.set_index('row_id')	

    dtrue['row_id'] = test_rowIDs	
    df_outputs = process_grid_cv(dtrain, dtest)
    df_outputs['row_id'] = df_outputs.index
    df_outputs['place_id'] = dtrue['place_id'].tolist()
    def score_cal (true, l1, l2, l3):
        score = 1.0*float(int(true) == int(l1))\
                          + 0.5 * float(int(true) == int(l2)) \
                          + 1/3.0* float(int(true) == int(l3))
        return score
    
    df_outputs['score'] = df_outputs.apply(lambda x : score_cal(x['place_id'], x['l1'], x['l2'], x['l3']), axis=1)
    return df_outputs['score'].mean(), df_outputs

print('Loading data')
print('Reading train.csv')
df_train = pd.read_csv('../input/train.csv', usecols=['row_id','x','y','accuracy','time','place_id'], index_col = 0)
df_train = prepare_data(df_train)

dat_train =df_train.reset_index().drop('row_id', 1,)

from sklearn.cross_validation import train_test_split
dtrain, dtest = train_test_split(dat_train, test_size = 0.35, random_state = 1219)
sco, withoutDupTmp = scoreValid(dtrain, dtest)
print sco
