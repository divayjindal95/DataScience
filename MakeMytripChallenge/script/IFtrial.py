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
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,cross_val_score,LeaveOneOut
#from sklearn.cross_validation import KFold,train_test_split,cross_val_score

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


data,cls=getint(data)


# data.O=np.log(data.O+1)
# data.H=np.log(data.H+1)
# data.K=np.log(data.K+1)
# data.N=np.log(data.N+1)
# data.C=np.log(data.C+1)

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


data['H_frac']=data.H-data.H.map(lambda x:int(x))
data['H_int'] = data.H.map(lambda x:int(x))

data['C_frac']=data.C-data.C.map(lambda x:int(x))
data['C_int'] = data.C.map(lambda x:int(x))

data['N_frac']=data.N-data.N.map(lambda x:int(x))
data['N_int'] = data.N.map(lambda x:int(x))

data=pd.concat([data,pd.get_dummies(data.A,'A')],axis=1)
data=pd.concat([data,pd.get_dummies(data.F,'F')],axis=1)


print data.head()
print data.columns
trncols=[u'A', u'B','C_frac','C_int', u'D', u'E', u'F', u'G', u'H_int','H_frac', u'I', u'J', u'K',
       u'L', u'M','N_frac','N_int', u'O']

trncols=[u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'id', u'H_frac', u'H_int', u'C_frac', u'C_int', u'N_frac', u'N_int', u'A_0', u'A_1', u'F_0', u'F_1', u'F_2', u'F_3', u'F_4', u'F_5', u'F_6', u'F_7', u'F_8', u'F_9', u'F_10', u'F_11', u'F_12', u'F_13']
testcols=['P']

data_bin  = ['A','I','J','L','F']
#trncols=data_bin

fin_train_data=data.iloc[:len(train_data)]
fin_test_data=data.iloc[len(train_data):]

#print fin_train_data[(fin_train_data.I==1) & (fin_train_data.J==0)].tostring()
print len(fin_train_data)
print len(fin_train_data[(fin_train_data.I==1) & (fin_train_data.J==1)]),len(fin_train_data[(fin_train_data.I==1) & (fin_train_data.J==1) & (fin_train_data.P==1)]),
print len(fin_train_data[(fin_train_data.I==0) & (fin_train_data.J==0)]),len(fin_train_data[(fin_train_data.I==0) & (fin_train_data.J==0) & (fin_train_data.P==0)])
print len(fin_train_data[(fin_train_data.I==0) & (fin_train_data.J==1)]),len(fin_train_data[(fin_train_data.I==0) & (fin_train_data.J==1) & (fin_train_data.P==0)])


print len(fin_test_data[(fin_test_data.I==1) & (fin_test_data.J==0)]),len(fin_test_data)

fin_train_data = fin_train_data[(fin_train_data.I==1) & (fin_train_data.J==0)]

from sklearn.utils import shuffle
fin_train_data= shuffle(fin_train_data)

X=fin_train_data[trncols]
Y=fin_train_data[testcols]
rfc=GradientBoostingClassifier(n_estimators=30)
#rfc=LogisticRegression()
rfc=LinearRegression()
#rfc=MultinomialNB()
kf=KFold(n_splits=5)
lo = LeaveOneOut()
accs=cross_val_score(rfc,X,Y,cv=kf)
accslo=cross_val_score(rfc,X,Y,cv=lo)
#print np.mean(accs),np.mean(accslo)
rfc.fit(X,Y)
#print rfc.score(X,Y)
#print rfc.predict(X)<0.5
rsss = pd.DataFrame((Y==0)==(rfc.predict(X)<0.5))
#print rsss[rsss.P==True]

# asnls=[]
#
# orans=y.P.tolist()
# x=x.reset_index(xrange(len(y)))
#
# for i in xrange(len(x)):
#     if x.I.iloc[i]==0 and x.J.iloc[i]==0:
#         asnls.append(1)
#     if x.I.iloc[i]==1 and x.J.iloc[i]==1:
#         asnls.append(1)
#     if x.I.iloc[i]==0 and x.J.iloc[i]==1:
#         asnls.append(1)
#     if x.I.iloc[i]==1 and x.J.iloc[i]==0:
#         asnls.append(orans[i])
#     i+=1
#
# res=0
# for a,b in zip(asnls,orans):
#     res+=np.abs(a-b)
# print res/len(orans)
fintestindex=fin_test_data.index

for e in fintestindex:
    if (fin_test_data['I'][e]==1) and (fin_test_data['J'][e]==1):
        fin_test_data['P'][e]=0
    if (fin_test_data['I'][e]==0) and (fin_test_data['J'][e]==0):
        fin_test_data['P'][e]=1
    if (fin_test_data['I'][e]==0) and (fin_test_data['J'][e]==1):
        fin_test_data['P'][e]=1
    # if (fin_test_data['I'][e]==1) and (fin_test_data['J'][e]==0):
    #     fin_test_data['P']=0
print fin_test_data.P

remaining=fin_test_data[fin_test_data.P.isnull()]
remainingans =rfc.predict(remaining[trncols])>0.5
fin_test_data[fin_test_data.P.isnull()]['P'][:]=np.reshape(remainingans.astype(int),(len(remainingans)))
fin_test_data[fin_test_data.P.isnull()]['P'][:]=1
print fin_test_data[fin_test_data.P.isnull()]['P'][:]
#print fin_test_data.P

final = pd.DataFrame()
final['id']=fin_test_data.id
# #final['P']=pd.to_numeric(rfc.predict(fin_test_data[trncols]),downcast='signed')
# final['P']=rfc.predict(fin_test_data[trncols]).astype(int)
# final.to_csv('../data/final.csv',index=False)