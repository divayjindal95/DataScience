import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np
import pandas as pd
from pylab import plot,show,hist,bar

train_data = pd.read_csv("../input/train.tsv",sep="\t")
train_data['log_price'] = np.log(train_data.price+1)

test_data = pd.read_csv("../input/test.tsv",sep="\t")

n_trains = train_data.shape[0]
n_tests = test_data.shape[0]

data = pd.concat([train_data,test_data])

data['Cat1']=data.category_name.str.split('/').str.get(0)
data['Cat2']=data.category_name.str.split('/').str.get(1)
data['Cat3']=data.category_name.str.split('/').str.get(2)

data['len_name']=list(map(lambda x : len(x),data.name))


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

data.Cat1= data.Cat1.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat1)
data['Cat1Tf'] = le.transform(data.Cat1)


data.Cat2= data.Cat2.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat2)
data['Cat2Tf'] = le.transform(data.Cat2)


data.Cat3= data.Cat3.fillna('NotPresent')
le = LabelEncoder()
le.fit(data.Cat3)
data['Cat3Tf'] = le.transform(data.Cat3)

data.brand_name=data.brand_name.fillna('OTHERS')
le=LabelEncoder()
le.fit(data.brand_name)
data['brand_nameTf']= le.transform(data.brand_name)


data1 = data.drop([ 'price',u'Cat1',u'Cat2', u'Cat3','train_id','name','category_name','brand_name','item_description'],axis=1)

trn_columns = ['item_condition_id','shipping','len_name','brand_nameTf', u'Cat1Tf',
       u'Cat2Tf', u'Cat3Tf']
targ = ['log_price']


train_data,test_data = train_test_split(data1[:n_trains],test_size=0.3)

#train_data=data1[:n_trains]
#test_data=data1[n_trains:]

'''
lr=LinearRegression()
lr.fit(train_data[trn_columns],train_data[targ])
preds = lr.predict(test_data[trn_columns])
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

abr=AdaBoostRegressor()
abr.fit(train_data[trn_columns],train_data[targ])
preds = abr.predict(test_data[trn_columns])
print(preds)
print(np.sqrt(np.mean(np.square(preds-test_data[targ]))))



'''
sub=pd.DataFrame()
sub['test_id']=test_data.test_id.astype('int64')
sub['price']=np.exp(preds)-1
#print(sub.test_id)
sub.to_csv("sample_submission.csv",index=False)
'''