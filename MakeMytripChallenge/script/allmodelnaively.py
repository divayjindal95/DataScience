import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from  sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

train_data = pd.read_csv("../data/train.csv")
train_data_len=len(train_data)
test_data=pd.read_csv("../data/test.csv")
test_data_len=len(test_data)



def getint(data):
    nicedata=data.dropna()
    cls=dict()
    for i in xrange(len(nicedata.columns)):
        if data.dtypes[i]==object and data.columns[i]!='P':
            le = LabelEncoder()
            nicedata[nicedata.columns[i]] = le.fit_transform(nicedata[nicedata.columns[i]])
            cls[nicedata.columns[i]]=le.classes_
    return nicedata,cls

print train_data.dtypes
#print len(train_data)
#print len(test_data)
train_data=train_data.dropna()
test_data=test_data.dropna()
data=pd.concat([train_data,test_data])
le=LabelEncoder()
data,cls=getint(data)
#print data.head()



data.O=np.log(data.O+1)
data.H=np.log(data.H+1)
data.K=np.log(data.K+1)

sc = StandardScaler()
data.O=sc.fit_transform(np.reshape(data.O,(len(data.O),1)))
sc = StandardScaler()
data.H=sc.fit_transform(np.reshape(data.H,(len(data.H),1)))
sc = StandardScaler()
data.K=sc.fit_transform(np.reshape(data.K,(len(data.K),1)))



fin_train_data=data.iloc[:len(train_data)]
fin_test_data=data.iloc[len(train_data)+1:]


#print len(train_data)
#print len(test_data)
#print train_data.columns

trncols=[u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K',
       u'L', u'M', u'N', u'O']
testcols=['P']

X,x,Y,y=train_test_split(fin_train_data[trncols],fin_train_data[testcols])

rfc=RandomForestClassifier()
rfc.fit(X,Y)
rfc.predict(x)
print rfc.score(x,y)
print rfc.feature_importances_

gb=GradientBoostingClassifier()
gb.fit(X,Y)
print "gb",gb.score(x,y)
print gb.feature_importances_


# mnb = MultinomialNB()
# mnb.fit(X,Y)
# print mnb.score(x,y)

# svc=SVC()
# svc.fit(X,Y)
# print svc.score(x,y)


lr=LogisticRegression()
lr.fit(X,Y)
print "lr",lr.score(x,y)

few_cols = ['C','N','I','O','B']
gb=GradientBoostingClassifier()
gb.fit(X[few_cols],Y)
print "few",gb.score(x[few_cols],y)
#print X[few_cols].head()
#print fin_train_data.corr()[fin_train_data.corr()>0.5]

lr=LogisticRegression()
lr.fit(X[few_cols],Y)
print "lr few",lr.score(x[few_cols],y)