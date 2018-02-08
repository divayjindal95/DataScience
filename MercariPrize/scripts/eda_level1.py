import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
from pylab import plot, show, hist, bar
import gc

issub = "N"

train_data = pd.read_csv("../input/train.tsv", sep="\t")
train_data['log_price'] = np.log(train_data.price + 1)

test_data = pd.read_csv("../input/test.tsv", sep="\t")

n_trains = train_data.shape[0]
n_tests = test_data.shape[0]

data = pd.concat([train_data, test_data])

data['Cat1'] = data.category_name.str.split('/').str.get(0)
data['Cat2'] = data.category_name.str.split('/').str.get(1)
data['Cat3'] = data.category_name.str.split('/').str.get(2)

dummydf = pd.get_dummies(data.shipping)
data['shipping_0'] = dummydf[dummydf.columns[0]]
data['shipping_1'] = dummydf[dummydf.columns[1]]

icidf = pd.get_dummies(data.item_condition_id)
data['ici_1'] = icidf[icidf.columns[0]]
data['ici_2'] = icidf[icidf.columns[1]]
data['ici_3'] = icidf[icidf.columns[2]]
data['ici_4'] = icidf[icidf.columns[3]]
data['ici_5'] = icidf[icidf.columns[4]]

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

data.Cat1 = data.Cat1.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat1)
data['Cat1Tf'] = le.transform(data.Cat1)

data.Cat2 = data.Cat2.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat2)
data['Cat2Tf'] = le.transform(data.Cat2)

data.Cat3 = data.Cat3.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat3)
data['Cat3Tf'] = le.transform(data.Cat3)

data.brand_name = data.brand_name.fillna('OTHERS')
le = LabelEncoder()
le.fit(data.brand_name)
data['brand_nameTf'] = le.transform(data.brand_name)

data.item_description = data.item_description.fillna('No Description yet')

data['len_name'] = list(map(lambda x: len(x), data.name))
data['len_item_description'] = list(map(lambda x: len(x), data.item_description))

data1 = data.drop(
    ['price', u'Cat1', u'Cat2', u'Cat3', 'train_id', 'name', 'category_name', 'brand_name', 'item_description'], axis=1)

trn_columns = ['item_condition_id', 'shipping', 'len_name', 'brand_nameTf', u'Cat1Tf',
               u'Cat2Tf', u'Cat3Tf', 'len_item_description', 'ici_1',
               'ici_2', 'ici_3', 'ici_4', 'ici_5', 'shipping_0', 'shipping_1']
targ = ['log_price']

train_data = data1[:n_trains]
test_data = data1[n_trains:]

if issub == "N":
    train_data, test_data = train_test_split(train_data, test_size=0.05)

train_data, val_data = train_test_split(train_data, test_size=0.05)

print("training")

lr = LinearRegression()
lr.fit(train_data[trn_columns], train_data[targ])
preds = lr.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds - test_data[targ]))))

'''
gbr=GradientBoostingRegressor()
gbr.fit(train_data[trn_columns],train_data[targ])
preds = gbr.predict(test_data[trn_columns])
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))
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

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=400,
                        learning_rate=0.2,
                        n_estimators=40)
gbm.fit(train_data[trn_columns],train_data.log_price,
        eval_set=[(val_data[trn_columns],val_data.log_price)],
        eval_metric='l2',
        early_stopping_rounds=5)

print("predicting...")  
del train_data
del data
gc.collect()
preds= gbm.predict(test_data[trn_columns])
'''

'''
df=pd.DataFrame()
df['preds']=preds
df['price']=list(val_data[targ]['log_price'])
print(np.sqrt(np.mean(np.square(df.preds-df.price))))
'''

sub = pd.DataFrame()
sub['test_id'] = test_data.test_id.astype('int64')
sub['price'] = np.exp(preds) - 1
# print(sub.test_id)
sub.to_csv("sample_submission.csv", index=False)
print("Completed!!")