import pandas as pd
import numpy as np
from pylab import  hist,show,plot
import pylab
import gc
import math
#from scikits.statsmodels.tsa.api import ARMA
data = pd.read_csv("../data/train.csv",iterator=True,chunksize=10000000)
sample_data1 = data.__next__()
del data
mydate = pd.to_datetime(sample_data1.date)
sample_data1.index = mydate


month1 = sample_data1[sample_data1.index.month==1]
del sample_data1

removed = month1[month1.unit_sales>0]
del month1

'''
print removed.head()
            id        date  store_nbr  item_nbr  unit_sales  onpromotion
date                                                                    
2013-01-01   0  2013-01-01         25    103665         7.0          NaN
2013-01-01   1  2013-01-01         25    105574         1.0          NaN
2013-01-01   2  2013-01-01         25    105575         2.0          NaN
2013-01-01   3  2013-01-01         25    108079         1.0          NaN
2013-01-01   4  2013-01-01         25    108701         1.0          NaN

print removed.info()
DatetimeIndex: 1211884 entries, 2013-01-01 to 2013-01-31
Data columns (total 6 columns):
id             1211884 non-null int64
date           1211884 non-null object
store_nbr      1211884 non-null int64
item_nbr       1211884 non-null int64
unit_sales     1211884 non-null float64
onpromotion    0 non-null float64
'''

#plot(removed.index,removed.unit_sales)
#show()

'''
from pandas.plotting import lag_plot
lag_plot(np.log(removed.unit_sales))
show()
'''
'''
removed=removed[removed.store_nbr==25]
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(removed.unit_sales[:10000])
show()
'''

'''
#finding the correlation between variables
unit_sales = removed.unit_sales
dataframe = pd.concat([unit_sales.shift(3),unit_sales.shift(10),np.log(unit_sales.shift(1)), unit_sales], axis=1)
dataframe.columns = ['t-3','t-2','t-1', 't']
result = dataframe.corr()
print(result)
'''
