import pandas as pd
import numpy as np
from pylab import  hist,show,plot
import pylab
import gc
import math

data = pd.read_csv("../data/train.csv",iterator=True,chunksize=10000000)
sample_data1 = data.__next__()
del data
mydate = pd.to_datetime(sample_data1.date)
sample_data1.index = mydate

#print sample_data.describe().to_string()
'''
                   id       store_nbr      item_nbr      unit_sales  onpromotion
count  1000000.000000  1000000.000000  1.000000e+06  1000000.000000          0.0
mean    499999.500000       26.342937  6.175491e+05        8.636756          NaN
std     288675.278932       16.468453  3.034943e+05       19.737254          NaN
min          0.000000        1.000000  9.699500e+04     -168.000000          NaN
25%     249999.750000       10.000000  3.580960e+05        2.000000          NaN
50%     499999.500000       27.000000  5.871860e+05        4.000000          NaN
75%     749999.250000       41.000000  8.723170e+05        9.000000          NaN
max     999999.000000       54.000000  1.118683e+06     5219.785000          NaN 

data contains error in unit sales ( negative values )
'''

month1 = sample_data1[sample_data1.index.month==1]
del sample_data1

#print month1.info()
'''
DatetimeIndex: 1211964 entries, 2013-01-01 to 2013-01-31
Data columns (total 6 columns):
id             1211964 non-null int64
date           1211964 non-null object
store_nbr      1211964 non-null int64
item_nbr       1211964 non-null int64
unit_sales     1211964 non-null float64
onpromotion    0 non-null float64
dtypes: float64(2), int64(3), object(1)
memory usage: 60.1+ MB

onpromotion is always null
'''

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



#log_unit_sales = np.log(removed['unit_sales'])
#hist(bins=1000,x=log_unit_sales,normed=1)

#item_nbr = np.log(removed['item_nbr'])
#hist(bins=1000,x=item_nbr,normed=1)

#store_nbr = np.log(removed['store_nbr'])
#hist(bins=1000,x=store_nbr,normed=1)

'''
# item vs units
tot_units=removed.groupby(["item_nbr","date"]).describe()#['unit_sales'].transform('sum')
units = tot_units['unit_sales','mean']
units = np.sort(units)
pylab.ylabel("mean units")
pylab.xlabel("item no")
plot(np.log(units))
'''

'''
#store vs units
tot_units=removed.groupby(["store_nbr"]).describe()#['unit_sales'].transform('sum')
units = tot_units['unit_sales','mean']
units = np.sort(units)
pylab.ylabel("mean units")
pylab.xlabel("shop no")
plot(units)
'''

'''
tot_units=removed.groupby(["date"]).describe()
units = tot_units['unit_sales','mean']
print units
pylab.ylabel("mean units")
pylab.xlabel("date")
#plot(units)
pylab.plot(range(1,len(units)+1),units)
'''

'''
removed=removed[removed.store_nbr==25]
removed=removed[ removed.item_nbr==103665]
print removed
plot(removed.index,removed.unit_sales)
pylab.ylabel("units")
pylab.xlabel("date")
pylab.title("store 25,item=103665")
show()
'''


#plot(removed.index,removed.unit_sales,'-o')
#show()

