import numpy as np
import pandas as pd
import matplotlib.pylab as plt


train_data = pd.read_csv("../data/train.csv")
test_data=pd.read_csv("../data/test.csv")
## pd.read_csv("../data/test.csv")


#print train_data.info()
'''
15 cols : A to O for training 
Many vars have null value
Non null : C , H , I , J , K , L , M  , O , P
with null : A , B , D , E , F, G , N 
6 cols with float or int 
9 are with string ( seems like binary are there in them) 

'''

#print train_data.describe()

#print train_data.head(5)
'''
   id  A      B       C  D  E   F  G      H  I  J   K  L  M    N    O  P
0   1  b  18.42  10.415  y  p  aa  v  0.125  t  f   0  f  g  120  375  1
1   2  a  21.75  11.750  u  g   c  v  0.250  f  f   0  t  g  180    0  1
2   3  b  30.17   1.085  y  p   c  v  0.040  f  f   0  f  g  170  179  1
3   4  b  22.67   2.540  y  p   c  h  2.585  t  f   0  f  g    0    0  0
4   5  a  36.00   1.000  u  g   c  v  2.000  t  t  11  f  g    0  456  0

'''


#print train_data.corr()
'''
the kind of corr seen states the vars are not linearly correlated
'''


print train_data[(train_data.A!='a') & (train_data.A!='b')]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_data.A=le.fit_transform(train_data.A)
print le.classes_
#print train_data.D.describe()
#print train_data.D.value_counts()

le=LabelEncoder()
train_data.D = le.fit_transform(train_data.D)

print le.classes_

#print train_data.corr()


#print train_data.E.value_counts()
#print train_data.corr()

le=LabelEncoder()
train_data.E = le.fit_transform(train_data.E)


print train_data.F.value_counts()
print test_data.F.value_counts()


print train_data.G.value_counts()
print train_data.I.value_counts()
print train_data.J.value_counts()
print train_data.L.value_counts()
print train_data.M.value_counts()





print test_data.G.value_counts()
print test_data.I.value_counts()
print test_data.J.value_counts()
print test_data.L.value_counts()
print test_data.M.value_counts()