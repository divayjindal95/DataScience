import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from  sklearn.naive_bayes import MultinomialNB

train_data = pd.read_csv("../data/train.csv")
train_data_len=len(train_data)
test_data=pd.read_csv("../data/test.csv")
test_data_len=len(test_data)
data=pd.concat([train_data,test_data])
train_data.drop(train_data[train_data.isnull().sum(axis=1)>2].index)
# print train_data.head(2)
# train_data.ix[train_data[train_data.A.isnull()].index,'A']=0
# print train_data[train_data.A.isnull()]
# print train_data[train_data.A==-1]
# le=LabelEncoder()
# train_data.A=le.fit_transform(train_data.A)
# print le.classes_
#
# print train_data.head(2)
#
# print train_data.A.value_counts()
#
# #print len(train_data[(train_data.A==0) & (train_data.P==1)])
#
# train_data_A_1 =train_data[train_data.A==1]
# #print train_data_A_1
# print train_data_A_1.describe()
# print train_data_A_1.describe(include=['O'])
#
#
# train_data_A_2=train_data[train_data.A==2]
# #print train_data_A_1
# print train_data_A_2.describe()
# print train_data_A_2.describe(include=['O'])



#mnb.fit(train_data[train_data.columns[1:-1]],train_data[train_data.A.notnull()].A)

data_1=data.dropna()
#le=LabelEncoder()
#data_1.apply(le.fit)
#print data_1.head()
#print le.classes_
def getint(data):
    nicedata=data.dropna()
    cls=dict()
    for i in xrange(len(nicedata.columns)):
        if data.dtypes[i]==object:
            le = LabelEncoder()
            nicedata[nicedata.columns[i]] = le.fit_transform(nicedata[nicedata.columns[i]])
            cls[nicedata.columns[i]]=le.classes_
    return nicedata,cls

data_1,dictcls=getint(data_1)
print data_1.head()
print dictcls
mnb = MultinomialNB()
mnb.fit(data_1[data_1.columns[1:-1]],data_1[data_1.A.notnull()].A)
#mnb.predict(data[data.D.isnull()])


print data[data.isnull().sum(axis=1)>2]
print data[(data.I=='f') & (data.J=='f')& (data.K==0)& (data.L=='f')& (data.M=='p')]