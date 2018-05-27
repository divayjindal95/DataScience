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
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
train_data = pd.read_csv("../data/train.csv")
train_data_len=len(train_data)
test_data=pd.read_csv("../data/test.csv")
test_data_len=len(test_data)



def getint(data):
    nicedata=data
    cls=dict()
    for i in xrange(len(nicedata.columns)):
        if data.dtypes[i]==object and data.columns[i]!='P':
            le = LabelEncoder()
            nicedata[nicedata.columns[i]] = le.fit_transform(nicedata[nicedata.columns[i]])
            cls[nicedata.columns[i]]=le.classes_
    return nicedata,cls

#train_data=train_data.dropna()
#test_data=test_data.dropna()
data=pd.concat([train_data,test_data])

data.A=data.A.fillna(data['A'].mode()[0])
data.D=data.D.fillna(data['D'].mode()[0])
data.E=data.E.fillna(data['E'].mode()[0])
data.G=data.G.fillna(data['G'].mode()[0])
data.F=data.F.fillna(data['F'].mode()[0])
data.B=data.A.fillna(data['B'].median())
data.N=data.N.fillna(data['N'].median())

data,cls=getint(data)

data.O=np.log(data.O+1)
data.H=np.log(data.H+1)
data.K=np.log(data.K+1)
data.N=np.log(data.N+1)
data.C=np.log(data.C+1)

data=pd.concat([data,pd.get_dummies(data.I,'I')],axis=1)
data=pd.concat([data,pd.get_dummies(data.F,'F')],axis=1)
#print data.describe().to_string()

sc = StandardScaler()
data.O=sc.fit_transform(np.reshape(data.O,(len(data.O),1)))
sc = StandardScaler()
data.H=sc.fit_transform(np.reshape(data.H,(len(data.H),1)))
sc = StandardScaler()
data.K=sc.fit_transform(np.reshape(data.K,(len(data.K),1)))
sc = StandardScaler()
data.N=sc.fit_transform(np.reshape(data.N,(len(data.N),1)))
sc = StandardScaler()
data.C=sc.fit_transform(np.reshape(data.C,(len(data.C),1)))
sc = StandardScaler()
data.B=sc.fit_transform(np.reshape(data.B,(len(data.B),1)))


fin_train_data=data.iloc[:len(train_data)]
fin_test_data=data.iloc[len(train_data)+1:]



trncols=[u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L',
       u'M', u'N', u'O', u'I_0', u'I_1']
# u'F_0', u'F_1', u'F_2',
#        u'F_3', u'F_4', u'F_5', u'F_6', u'F_7', u'F_8', u'F_9', u'F_10',
#        u'F_11', u'F_12', u'F_13'
testcols=['P']

X,x,Y,y=train_test_split(fin_train_data[trncols],fin_train_data[testcols])


# gb=GradientBoostingClassifier()
# gb.fit(X,Y)
# print "gb",gb.score(x,y)
# print gb.feature_importances_

lr=LogisticRegression()
lr.fit(X,Y)
print "lr",lr.score(x,y)