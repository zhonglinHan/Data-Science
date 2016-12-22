import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter

# per raddar, all date features except for stations 24+25 are identical

def get_date_features():
    directory = '../input/'
    trainfile = 'train_date.csv'
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          chunksize=1,
                                          low_memory=False)):
        features = list(chunk.columns)
        break

    seen = np.zeros(52)
    rv = []
    for f in features:
        if f == 'Id' or 'S24' in f or 'S25' in f:
            rv.append(f)
            continue
            
        station = int(f.split('_')[1][1:])
#        print(station)
        
        if seen[station]:
            continue
        
        seen[station] = 1
        rv.append(f)
        
    return rv
        
usefuldatefeatures = get_date_features()

def get_mindate():
    directory = '../input/'
    trainfile = 'train_date.csv'
    testfile = 'test_date.csv'
    
    features = None
    subset = None
    
    for i, chunk in enumerate(pd.read_csv(directory + trainfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        print(i)
        
        if features is None:
            features = list(chunk.columns)
            features.remove('Id')
        
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        
        if subset is None:
            subset = df_mindate_chunk.copy()
        else:
            subset = pd.concat([subset, df_mindate_chunk])
            
        del chunk
        gc.collect()

    for i, chunk in enumerate(pd.read_csv(directory + testfile,
                                          usecols=usefuldatefeatures,
                                          chunksize=50000,
                                          low_memory=False)):
        print(i)
        
        df_mindate_chunk = chunk[['Id']].copy()
        df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
        subset = pd.concat([subset, df_mindate_chunk])
        
        del chunk
        gc.collect()      
        
    return subset


df_mindate = get_mindate()

df_mindate.sort_values(by=['mindate', 'Id'], inplace=True)

df_mindate['mindate_id_diff'] = df_mindate.Id.diff()

midr = np.full_like(df_mindate.mindate_id_diff.values, np.nan)
midr[0:-1] = -df_mindate.mindate_id_diff.values[1:]

df_mindate['mindate_id_diff_reverse'] = midr

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['Response'].mean().reset_index()
    grpCount = data1.groupby(columnName)['Response'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.Response
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 1]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['Response'].values
    x = pd.merge(data2[[columnName, 'Response']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['Response']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        #  x = x + np.random.normal(0, .01, x.shape[0])
    return x.fillna(x.mean())

"""
["Id", "L0_S0_D1","L0_S1_D26","L0_S10_D216","L0_S10_D221","L0_S10_D231","L0_S11_D280","L0_S12_D331","L0_S13_D355","L0_S14_D360","L0_S15_D395","L0_S16_D423","L0_S17_D432","L0_S18_D437","L0_S19_D454","L0_S2_D34","L0_S20_D462","L0_S21_D469","L0_S21_D474","L0_S21_D484","L0_S22_D543","L0_S22_D548","L0_S22_D553","L0_S22_D558","L0_S23_D617","L0_S3_D70","L0_S4_D106","L0_S4_D111","L0_S5_D115","L0_S6_D120","L0_S7_D137","L0_S8_D145","L0_S9_D152","L0_S9_D157","L0_S9_D167","L0_S10_D216","L0_S10_D221","L0_S10_D231","L0_S11_D280","L0_S12_D331","L0_S13_D355","L0_S14_D360","L0_S15_D395","L0_S16_D423","L0_S17_D432","L0_S18_D437","L0_S19_D454","L0_S20_D462","L0_S21_D469","L0_S21_D474","L0_S21_D484","L0_S22_D543","L0_S22_D548","L0_S22_D553","L0_S22_D558","L0_S23_D617","L1_S24_D677","L1_S24_D697","L1_S24_D702","L1_S24_D772","L1_S24_D801","L1_S24_D804","L1_S24_D807","L1_S24_D813","L1_S24_D818","L1_S24_D909","L1_S24_D999","L1_S24_D1018","L1_S24_D1062","L1_S24_D1116","L1_S24_D1135","L1_S24_D1155","L1_S24_D1158","L1_S24_D1163","L1_S24_D1168","L1_S24_D1171","L1_S24_D1178","L1_S24_D1186","L1_S24_D1277","L1_S24_D1368","L1_S24_D1413","L1_S24_D1457","L1_S24_D1511","L1_S24_D1522","L1_S24_D1536","L1_S24_D1558","L1_S24_D1562","L1_S24_D1566","L1_S24_D1568","L1_S24_D1570","L1_S24_D1576","L1_S24_D1583","L1_S24_D1674","L1_S24_D1765","L1_S24_D1770","L1_S24_D1809","L1_S24_D1826","L1_S25_D1854","L1_S25_D1867","L1_S25_D1883","L1_S25_D1887","L1_S25_D1891","L1_S25_D1898","L1_S25_D1902","L1_S25_D1980","L1_S25_D2058","L1_S25_D2098","L1_S25_D2138","L1_S25_D2180","L1_S25_D2206","L1_S25_D2230","L1_S25_D2238","L1_S25_D2240","L1_S25_D2242","L1_S25_D2248","L1_S25_D2251","L1_S25_D2329","L1_S25_D2406","L1_S25_D2430","L1_S25_D2445","L1_S25_D2471","L1_S25_D2497","L1_S25_D2505","L1_S25_D2507","L1_S25_D2509","L1_S25_D2515","L1_S25_D2518","L1_S25_D2596","L1_S25_D2674","L1_S25_D2713","L1_S25_D2728","L1_S25_D2754","L1_S25_D2780","L1_S25_D2788","L1_S25_D2790","L1_S25_D2792","L1_S25_D2798","L1_S25_D2801","L1_S25_D2879","L1_S25_D2957","L1_S25_D2996","L1_S25_D3011","L2_S26_D3037","L2_S26_D3081","L2_S27_D3130","L2_S28_D3223","L3_S29_D3316","L3_S29_D3474","L3_S30_D3496","L3_S30_D3506","L3_S30_D3566","L3_S30_D3726","L3_S31_D3836","L3_S32_D3852","L3_S33_D3856","L3_S34_D3875","L3_S35_D3886","L3_S35_D3895","L3_S35_D3900","L3_S36_D3919","L3_S36_D3928","L3_S37_D3942","L3_S38_D3953","L3_S39_D3966","L3_S40_D3981","L3_S40_D3985","L3_S41_D3997","L3_S42_D4029","L3_S42_D4045","L3_S43_D4062","L3_S43_D4082","L3_S44_D4101","L3_S45_D4125","L3_S46_D4135","L3_S47_D4140","L3_S48_D4194","L3_S49_D4208","L3_S49_D4218","L3_S50_D4242","L3_S51_D4255"]
"""
def GrabData():
    directory = '../input/'
    trainfiles = ['train_categorical.csv',
                  'train_date.csv',
                  'train_numeric.csv']
    testfiles = ['test_categorical.csv',
                 'test_date.csv',
                 'test_numeric.csv']
    cols_numeric_L1_S24 = open('./numeric_L1_S24.csv') 
    cols_string = cols_numeric_L1_S24.readline()
    L1_S24_NUM  = cols_string.replace('\n', '').split(',')
    cols_numeric_L1_S24.close() 
    L1_S24_DAT  = ["L1_S24_D677","L1_S24_D697","L1_S24_D702","L1_S24_D772","L1_S24_D801","L1_S24_D804","L1_S24_D807","L1_S24_D813","L1_S24_D818","L1_S24_D909","L1_S24_D999","L1_S24_D1018","L1_S24_D1062","L1_S24_D1116","L1_S24_D1135","L1_S24_D1155","L1_S24_D1158","L1_S24_D1163","L1_S24_D1168","L1_S24_D1171","L1_S24_D1178","L1_S24_D1186","L1_S24_D1277","L1_S24_D1368","L1_S24_D1413","L1_S24_D1457","L1_S24_D1511","L1_S24_D1522","L1_S24_D1536","L1_S24_D1558","L1_S24_D1562","L1_S24_D1566","L1_S24_D1568","L1_S24_D1570","L1_S24_D1576","L1_S24_D1583","L1_S24_D1674","L1_S24_D1765","L1_S24_D1770","L1_S24_D1809","L1_S24_D1826"]
    L3_S33_NUM = ['L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859', 'L3_S33_F3861', 'L3_S33_F3863', 'L3_S33_F3865', 'L3_S38_F3952']
                # ['L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859','L3_S33_F3863','L3_S33_F3865','L3_S33_F3867','L3_S33_F3869']
    cols = [['Id',
             'L1_S24_F1559', 'L3_S32_F3851',
             'L1_S24_F1827', 'L1_S24_F1582',
             'L3_S32_F3854', 'L1_S24_F1510', 
             'L1_S24_F1525'],
            ['Id',
             'L0_S21_D514'] +L1_S24_DAT + ['L1_S25_D1887', 'L2_S27_D3156',
             'L2_S28_D3227', 'L3_S29_D3325','L3_S30_D3496', 'L3_S30_D3506',
             'L3_S30_D3501', 'L3_S30_D3516','L3_S30_D3511', 'L3_S32_D3852',
             'L3_S33_D3856', 'L3_S36_D3940', 'L3_S37_D3942','L3_S46_D4135',
             'L3_S47_D4140', 'L3_S51_D4255'],
            ['Id',
             'L3_S32_F3850','L3_S29_F3407','L3_S38_F3952'] + L1_S24_NUM + L3_S33_NUM +
             ['Response']]
    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()
    del cols[2][-1]  # Test doesn't have response!
    for i, f in enumerate(testfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=50000,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()
        
    traindata = traindata.merge(df_mindate, on='Id')
    testdata = testdata.merge(df_mindate, on='Id')
        
    testdata['Response'] = 0  # Add Dummy Value
    visibletraindata = traindata[::2]
    blindtraindata = traindata[1::2]
    print(blindtraindata.columns)
    for i in range(2):
        for col in cols[i][1:]:
            print(col)
            blindtraindata.loc[:, col] = LeaveOneOut(visibletraindata,
                                                     blindtraindata,
                                                     col, False).values
            testdata.loc[:, col] = LeaveOneOut(visibletraindata,
                                               testdata, col, False).values
    del visibletraindata
    gc.collect()
    testdata.drop('Response', inplace=True, axis=1)
    return blindtraindata, testdata


def Train():
    train, test = GrabData()
    print('Train:', train.shape)
    print('Test', test.shape)
    features = list(train.columns)
    features.remove('Response')
    features.remove('Id')
    print(features)
    num_rounds = 50
    params = {}
    params['objective'] = "binary:logistic"
    params['eta'] = 0.021
    params['max_depth'] = 7
    params['colsample_bytree'] = 0.82
    params['min_child_weight'] = 3
    params['base_score'] = 0.005
    params['silent'] = True

    print('Fitting')
    trainpredictions = None
    testpredictions = None

    dvisibletrain = \
        xgb.DMatrix(train[features],
                    train.Response,
                    silent=True)
    dtest = \
        xgb.DMatrix(test[features],
                    silent=True)

    folds = 1
    for i in range(folds):
        print('Fold:', i)
        params['seed'] = i
        watchlist = [(dvisibletrain, 'train'), (dvisibletrain, 'val')]
        clf = xgb.train(params, dvisibletrain,
                        num_boost_round=num_rounds,
                        evals=watchlist,
                        early_stopping_rounds=20,
                        feval=mcc_eval,
                        maximize=True
                        )
        limit = clf.best_iteration+1
        # limit = clf.best_ntree_limit
        predictions = \
            clf.predict(dvisibletrain, ntree_limit=limit)

        best_proba, best_mcc, y_pred = eval_mcc(train.Response,
                                                predictions,
                                                True)
        print('tree limit:', limit)
        print('mcc:', best_mcc)
        print(matthews_corrcoef(train.Response,
                                y_pred))
        if(trainpredictions is None):
            trainpredictions = predictions
        else:
            trainpredictions += predictions
        predictions = clf.predict(dtest, ntree_limit=limit)
        if(testpredictions is None):
            testpredictions = predictions
        else:
            testpredictions += predictions
        imp = get_importance(clf, features)
        print('Importance array: ', imp)

    best_proba, best_mcc, y_pred = eval_mcc(train.Response,
                                            trainpredictions/folds,
                                            True)
    print(matthews_corrcoef(train.Response,
                            y_pred))

    submission = pd.DataFrame({"Id": train.Id,
                               "Prediction": trainpredictions/folds,
                               "Response": train.Response})
    submission[['Id',
                'Prediction',
                'Response']].to_csv('rawtrainxgbsubmission'+str(folds)+'.csv',
                                    index=False)

    submission = pd.DataFrame({"Id": test.Id.values,
                               "Response": testpredictions/folds})
    submission[['Id', 'Response']].to_csv('rawxgbsubmission'+str(folds)+'.csv',
                                          index=False)
    y_pred = (testpredictions/folds > .08).astype(int)
    submission = pd.DataFrame({"Id": test.Id.values,
                               "Response": y_pred})
    submission[['Id', 'Response']].to_csv('xgbsubmission'+str(folds)+'.csv',
                                          index=False)

if __name__ == "__main__":
    print('Started')
    Train()
    print('Finished')
