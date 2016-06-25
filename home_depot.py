import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error, make_scorer
import random

################################################################################
## Data Exploration 
################################################################################
## read data (encoding?)
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")[:1000]
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")[:1000]
attributes = pd.read_csv('attributes.csv')
attributes.dropna(inplace=True)
description = pd.read_csv('product_descriptions.csv')

## check content
df_train.head()

## check summary
print(str(df_train.describe()))
df_train.product_uid.value_counts()
attributes.name.value_counts()

## make some plots
df_train.relevance.hist()
df_train.relevance.value_counts()

description.product_description.str.len().hist(bins=30)
df_train.product_title.str.len().hist(bins=30)


## benchmark? 
## http://blog.revolutionanalytics.com/2016/03/classification-models.html?utm_content=bufferac8c0&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

################################################################################
## Data Preparation
################################################################################
df_train['label'] = 'train'
df_test['label'] = 'test'
## add a full set
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
## add descriptions to the full set
df_all = pd.merge(df_all, description, how='left', on='product_uid')

## data cleaning
stemmer = SnowballStemmer('english')
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

## add number of words in the search term
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

## add number of words in the description
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)

## add brand
df_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_all = pd.merge(df_all, df_brand, how = 'left', on = 'product_uid')

## add number of words in brand
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(str(x).split())).astype(np.int64)


## add material
#material = dict()
#attributes['about_material'] = attributes['name'].str.lower().str.contains('material')
#for row in attributes[attributes['about_material']].iterrows():
#    r = row[1]
#    product = r['product_uid']
#    value = r['value']
#    material.setdefault(product, '')
#    material[product] = material[product] + ' ' + str(value)
#df_material = pd.DataFrame.from_dict(material, orient='index')
#df_material = df_material.reset_index()
#df_material.columns = ['product_uid', 'material']

df_material = attributes[attributes.name.str.lower().str.contains('material')][["product_uid", "value"]].rename(columns={"value": "material"})
df_all = pd.merge(df_all, df_material, how='left', on='product_uid')

## add color
df_color = attributes[attributes.name.str.lower().str.contains('color')][["product_uid", "value"]].rename(columns={"value": "color"})
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')

## process strings
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stemmer(str(x)))
df_all['material'] = df_all['material'].map(lambda x: str(x).lower())
df_all['color'] = df_all['color'].map(lambda x: str(x).lower())


df_all['product_info'] = df_all['search_term']+ "\t" + df_all['product_title']+ "\t" + df_all['product_description'] + "\t" + df_all['brand']
## if words in search_term are in title or description
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['query_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[3]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info','material','color','brand','product_info', 'product_description'], axis = 1)
df_all.fillna(0)
df_train = df_all[df_all.label == 'train']
df_test = df_all[df_all.label == 'test']
id_test = df_test['id']

################################################################################
## Model Building
################################################################################
random.seed(2016)
n_train = df_train.shape[0]

## shuffle index to split training and validation set
random.shuffle(df_train.index.values)

## training set
y_train = df_train.iloc[: int(n_train * 0.7)]['relevance'].values
X_train = df_train.drop(['id','relevance', 'label'],axis=1).iloc[: int(n_train * 0.7)]

## validation set
y_valid = df_train.iloc[int(n_train * 0.7) +1 : ]['relevance'].values
X_valid = df_train.drop(['id','relevance', 'label'],axis=1).iloc[int(n_train * 0.7) + 1 : ].values

## testing set
X_test = df_test.drop(['id','relevance', 'label'],axis=1).values

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
#clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)

rf.feature_importances_




## calculate rmse 
def fmean_squared_error(ground_truth, predictions):
    rmse = mean_squared_error(ground_truth, predictions)**0.5
    return rmse

fmean_squared_error(y_train, y_pred)

rf.fit(X_valid, y_valid)
y_pred = rf.predict(X_valid)
fmean_squared_error(y_valid, y_pred)

################################################################################
## Results
################################################################################
## build final model
## training set
y_full_train = df_train['relevance'].values
X_full_train = df_train.drop(['id','relevance', 'label'],axis=1)
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)