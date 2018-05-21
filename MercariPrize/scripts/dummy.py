####   SET THIS FLAG   ###
issub = "Y"
develop = False
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
from pylab import plot, show, hist, bar
import gc
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix, hstack
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
import lightgbm as lgb

from nltk.corpus import stopwords
import re
from time import gmtime, strftime

def fill_missing(data):
    data.Cat1 = data.Cat1.fillna('NotPresent')
    data.Cat2 = data.Cat2.fillna('NotPresent')
    data.Cat3 = data.Cat3.fillna('NotPresent')
    data.item_description = data.item_description.fillna('No Description yet')
    data.brand_name = data.brand_name.fillna('NotPresent')


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


start_time = time.time()

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))


train_data = pd.read_csv("../input/mercari-price-suggestion-challenge/train.tsv", sep="\t")
train_data['log_price'] = np.log1p(train_data.price)
train_data = train_data[train_data.price > 2]
train_data = train_data[train_data.price < 500]
test_data = pd.read_csv("../input/mercari-price-suggestion-challenge/test.tsv", sep="\t")

n_trains = train_data.shape[0]
n_tests = test_data.shape[0]

data = pd.concat([train_data, test_data])

data['Cat1'] = data.category_name.str.split('/').str.get(0)
data['Cat2'] = data.category_name.str.split('/').str.get(1)
data['Cat3'] = data.category_name.str.split('/').str.get(2)
fill_missing(data)

'''
icidf=pd.get_dummies(data.item_condition_id)
data['ici_1']=icidf[icidf.columns[0]]
data['ici_2']=icidf[icidf.columns[1]]
data['ici_3']=icidf[icidf.columns[2]]
data['ici_4']=icidf[icidf.columns[3]]
data['ici_5']=icidf[icidf.columns[4]]

dummydf=pd.get_dummies(data.shipping)
data['shipping_0']=dummydf[dummydf.columns[0]]
data['shipping_1']=dummydf[dummydf.columns[1]]
'''
'''
data['splitname']=data.name.str.split(" ")
data['splitdescription']=data.item_description.str.split(" ")
data['splitbrandname']=data.brand_name.str.split(" ")

data['name_brand']=list(map(lambda i:len(set(data.splitname.iloc[i])&set(data.splitbrandname.iloc[i])),range(len(data))))
data['brand_desc']=list(map(lambda i:len(set(data.splitname.iloc[i])&set(data.splitdescription.iloc[i])),range(len(data))))
data['name_brand_desc']=list(map(lambda i:len(set(data.splitname.iloc[i])&set(data.splitbrandname.iloc[i])),range(len(data))))

data=data.drop(['splitname','splitdescription','splitbrandname'],axis=1)
gc.collect()
'''

# data=pd.concat([data,pd.get_dummies(data.Cat1,prefix="Cat1_")],axis=1)

cat2_rv = data[['log_price', 'Cat2']].groupby(['Cat2']).describe()
cat2_rv_dict = dict(zip(cat2_rv['log_price']['mean'].index, cat2_rv['log_price']['mean']))
data['mean_Cat2'] = list(map(lambda x: cat2_rv_dict[x], data.Cat2))
del cat2_rv
del cat2_rv_dict

cat3_rv = data[['log_price', 'Cat3']].groupby(['Cat3']).describe()
cat3_rv_dict = dict(zip(cat3_rv['log_price']['mean'].index, cat3_rv['log_price']['mean']))
data['mean_Cat3'] = list(map(lambda x: cat3_rv_dict[x], data.Cat3))
del cat3_rv
del cat3_rv_dict

data = data.drop(['Cat2', 'Cat3', 'category_name'], axis=1)
gc.collect()

brand_name_rv = data[['brand_name', 'log_price']].groupby(['brand_name']).describe()
brand_name_rv_dict = dict(zip(brand_name_rv['log_price']['mean'].index, brand_name_rv['log_price']['mean']))
data['mean_brand_name'] = list(map(lambda x: brand_name_rv_dict[x], data.brand_name))
del brand_name_rv
del brand_name_rv_dict

data['len_name'] = list(map(lambda x: len(x), data.name))

data['len_item_description'] = list(map(lambda x: len(x), data.item_description))

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 29, "norm": "l2", "tf": 1.0,
                                                              "idf": None})
                         , procs=8)
wb.dictionary_freeze = True
X_description = wb.fit_transform(data['item_description'])
del (wb)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

X_dummies = csr_matrix(pd.get_dummies(data[['item_condition_id', 'shipping', 'Cat1']],
                                            sparse = True).values)
X_more=csr_matrix(data[['mean_Cat2','mean_Cat3','mean_brand_name','len_name']].values)
sparse_merge = hstack((X_dummies, X_description,X_more)).tocsr()

mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]

X = sparse_merge[:n_trains]
X_test = sparse_merge[n_trains:]
print(sparse_merge.shape)

gc.collect()
y = train_data['log_price']
train_X, train_y = X, y

if develop:
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=47, inv_link="identity",
             threads=1)

model.fit(train_X, train_y)

print('[{}] Train FTRL completed'.format(time.time() - start_time))
if develop:
    preds = model.predict(X=valid_X)
    print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

predsF = model.predict(X_test)
print('[{}] Predict FTRL completed'.format(time.time() - start_time))

model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0,
                init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=18, inv_link="identity", threads=4)

model.fit(train_X, train_y)
print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
if develop:
    preds = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

params = {
    'learning_rate': 0.57,
    'application': 'regression',
    'max_depth': 5,
    'num_leaves': 32,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'feature_fraction': 0.65,
    'nthread': 4,
    'min_data_in_leaf': 100,
    'max_bin': 31
}

# Remove features with document frequency <=100
print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:n_trains]
X_test = sparse_merge[n_trains:]
print(sparse_merge.shape)

y = train_data['log_price']
train_X, train_y = X, y
if develop:
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

d_train = lgb.Dataset(train_X, label=train_y)
watchlist = [d_train]
if develop:
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=5500, valid_sets=watchlist, \
                  early_stopping_rounds=1000, verbose_eval=1000)

if develop:
    preds = model.predict(valid_X)
    print("LGB dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

predsL = model.predict(X_test)

print('[{}] Predict LGB completed.'.format(time.time() - start_time))

test_preds = (predsF * 0.1 + predsL * 0.22 + predsFM * 0.68)

submission: pd.DataFrame = test_data[['test_id']]

submission['price'] = np.expm1(test_preds)
submission.to_csv("submission_wordbatch_ftrl_fm_lgb.csv", index=False)

'''
print(data.columns)
data1 = data.drop(['item_condition_id', 'shipping', 'price','train_id','name','brand_name','item_description'],axis=1)


trn_columns =[  'ici_1', 'ici_2',
       'ici_3', 'ici_4', 'ici_5', 'shipping_0', 'shipping_1', 'Cat1__Beauty',
       'Cat1__Electronics', 'Cat1__Handmade', 'Cat1__Home', 'Cat1__Kids',
       'Cat1__Men', 'Cat1__NotPresent', 'Cat1__Other',
       'Cat1__Sports & Outdoors', 'Cat1__Vintage & Collectibles',
       'Cat1__Women', 'mean_Cat2', 'mean_Cat3', 'mean_brand_name', 'len_name',
       'len_item_description']

targ = ['log_price']

train_data=data1[:n_trains]
test_data=data1[n_trains:]

if issub=="N":
    train_data,test_data = train_test_split(train_data,test_size=0.05)
train_data,val_data = train_test_split(train_data,test_size=0.05)
'''
'''
lr=LinearRegression()
lr.fit(train_data[trn_columns],train_data[targ])
preds = lr.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
'''
'''
el=ElasticNet()
el.fit(train_data[trn_columns],train_data[targ])
preds = el.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
'''
'''
gbr=GradientBoostingRegressor()
gbr.fit(train_data[trn_columns],train_data[targ])
preds = gbr.predict(test_data[trn_columns])
'''

'''
mlp = MLPRegressor()
mlp.fit(train_data[trn_columns],train_data[targ])
preds = mlp.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
'''

'''
abr=AdaBoostRegressor()
abr.fit(train_data[trn_columns],train_data[targ])
preds = abr.predict(test_data[trn_columns])
print(preds)
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
'''

'''
from xgboost import XGBClassifier
xgb = XGBClassifier()
print("training")
xgb.fit(train_data[trn_columns],train_data[targ])
print("predicting")
preds = xgb.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
'''

'''
print("Training...")
import lightgbm as lgb
print('Start training...')
# train

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=50,
                        learning_rate=0.07,
                        n_estimators=80)


gbm.fit(train_data[trn_columns],train_data.log_price,
        eval_set=[(val_data[trn_columns],val_data.log_price)],
        eval_metric='rmsle',
        early_stopping_rounds=10)

print("predicting...")  
del train_data
del data
gc.collect()
test_preds= gbm.predict(test_data[trn_columns])

if issub=="N":
    df=pd.DataFrame()
    df['preds']=test_preds
    df['price']=list(test_data[targ]['log_price'])
    print(np.sqrt(np.mean(np.square(df.preds-df.price))))
    print("Completed!!")
else:
    sub=pd.DataFrame()
    sub['test_id']=test_data.test_id.astype('int64')
    sub['price']=np.exp(test_preds)-1
    #print(sub.test_id)
    sub.to_csv("sample_submission.csv",index=False)
    print("Completed!!")
'''