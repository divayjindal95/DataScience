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
from sklearn.model_selection import KFold,cross_val_score

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



data=pd.concat([train_data,test_data])

data.A=data.A.fillna(data['A'].mode()[0])
data.D=data.D.fillna(data['D'].mode()[0])
data.E=data.E.fillna(data['E'].mode()[0])
data.G=data.G.fillna(data['G'].mode()[0])
data.F=data.F.fillna(data['F'].mode()[0])
data.B=data.A.fillna(data['B'].median())
data.N=data.N.fillna(data['N'].median())

#print len(data.dropna())
#print data.describe()




le=LabelEncoder()
data,cls=getint(data)


data.O=np.log(data.O+1)
data.H=np.log(data.H+1)
data.K=np.log(data.K+1)
data.N=np.log(data.N+1)
data.C=np.log(data.C+1)

# sc = StandardScaler()
# data.O=sc.fit_transform(np.reshape(data.O,(len(data.O),1)))
# sc = StandardScaler()
# data.H=sc.fit_transform(np.reshape(data.H,(len(data.H),1)))
# sc = StandardScaler()
# data.K=sc.fit_transform(np.reshape(data.K,(len(data.K),1)))
# sc = StandardScaler()
# data.N=sc.fit_transform(np.reshape(data.N,(len(data.N),1)))
# sc = StandardScaler()
# data.C=sc.fit_transform(np.reshape(data.C,(len(data.C),1)))
# sc = StandardScaler()
# data.B=sc.fit_transform(np.reshape(data.B,(len(data.B),1)))


trncols=[u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K',
       u'L', u'M', u'N', u'O']
testcols=['P']

data_bin  = ['A','I','J','L','F']
#trncols=data_bin

fin_train_data=data.iloc[:len(train_data)]
fin_test_data=data.iloc[len(train_data):]

X,x,Y,y=train_test_split(fin_train_data[trncols],fin_train_data[testcols])


rfc=GradientBoostingClassifier()
rfc.fit(X[trncols],Y)
print rfc.score(x[trncols],y)

kf=KFold(n_splits=5)
print cross_val_score(rfc,X[trncols],Y,cv=kf)

final = pd.DataFrame()
final['id']=fin_test_data.id
final['P']=pd.to_numeric(rfc.predict(fin_test_data[trncols]),downcast='signed')
final.to_csv('../final.csv',index=False)
