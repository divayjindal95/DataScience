import pandas as pd
import numpy as np
from pylab import show,scatter,xlim,xticks,plot,hist

#item_cat = pd.read_csv("../Data/item_categories.csv")
# 84 categories

items=pd.read_csv("../Data/items.csv")
#22170items

#shops=pd.read_csv("../Data/shops.csv")
# 60shops

sales=pd.read_csv("../Data/sales_train_v2.csv")
sales=pd.merge(left=sales,right=items[['item_id','item_category_id']],on='item_id',how='inner')
# 2935848 count

#print sales.head()

allcols =[u'date',
          u'date_block_num',
          u'shop_id',
          u'item_id',
          u'item_price',
          u'item_cnt_day',
          u'item_category_id']

print sales.describe()
#print sales.info()
'''
item_price
item price is highly skewed , with median as 399, max : 307980 , mean : 890 , var : 1729
some negative values are even present , check for that

item_cnt_day
this is also highly skewed , also negative values are present
'''
samplesales=sales[(sales.item_price<5000) & (sales.item_price>0) & (sales.item_cnt_day<=100)]
samplesales_inv = sales[(sales.item_price>5000) & (sales.item_price>0)]
print len(samplesales)
#print samplesales[samplesales.item_price<0]
#hist(samplesales['item_price'],bins=10)
#show()

#print samplesales[samplesales.item_cnt_day<0]

''' 7000 data points have this, need see what is the importance of this '''


#print samplesales[samplesales.item_cnt_day==0]
''' this means that they have removed the data point having no sale that day , IMPORTANT TO INDUCE THAT DATA POINT '''

print samplesales.head()

#print hist(samplesales['item_category_id'],bins=85)
#print samplesales_inv['item_category_id']
#print hist(samplesales_inv['item_category_id'],bins=85)

print sales.groupby(['shop_id']).describe()