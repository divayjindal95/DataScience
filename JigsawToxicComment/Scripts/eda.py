import numpy as np
import pandas as pd

train= pd.read_csv("../data/train.csv")
test= pd.read_csv("../data/test.csv")

print train.info()
#print train.describe()
#print train.nunique()

cols=["toxic",
"severe_toxic",
"obscene",
"threat",
"insult",
"identity_hate"]

#for col in cols:
#    print col,"0: ",train[col].value_counts()[0]*1.0/len(train),"1: ",train[col].value_counts()[1]*1.0/len(train)

"""

toxic 0:  0.90415551698 1:  0.0958444830201
severe_toxic 0:  0.99000444943 1:  0.00999555056997
obscene 0:  0.947051782592 1:  0.0529482174079
threat 0:  0.99700446823 1:  0.00299553176956
insult 0:  0.950636393831 1:  0.049363606169
identity_hate 0:  0.991195141974 1:  0.00880485802558

"""

#print train.corr().to_string()
"""
                  toxic  severe_toxic   obscene    threat    insult  identity_hate
toxic          1.000000      0.308619  0.676515  0.157058  0.647518       0.266009
severe_toxic   0.308619      1.000000  0.403014  0.123601  0.375807       0.201600
obscene        0.676515      0.403014  1.000000  0.141179  0.741272       0.286867
threat         0.157058      0.123601  0.141179  1.000000  0.150022       0.115128
insult         0.647518      0.375807  0.741272  0.150022  1.000000       0.337736
identity_hate  0.266009      0.201600  0.286867  0.115128  0.337736       1.000000
"""

#print train[(train.toxic==0)&(train.severe_toxic==0)&(train.obscene==0)&(train.threat==0)&(train.insult==0)&(train.identity_hate==0)]
# 143346/159570  = 89% data is all zeroes

'''
for i in xrange(15):
    print train[(train.toxic == 0) & (train.severe_toxic == 0) & (train.obscene == 0) & (train.threat == 0) & (
    train.insult == 0) & (train.identity_hate == 0)].iloc[i][1]
    print "--------------------"

nothing : simple nice statements, with thanks,sorry,please,aplologize,lets,
'''

'''
for i in xrange(15):
    print train[(train.obscene==1)].iloc[i+13][1]
    print train[(train.obscene==1)].iloc[i+13]

severe_toxic: fuck you,motherfucker,cocksucker,

threat : die ,terror,cut dog,attack,Bad things will happen,killing,blocked,kill

identity_hate : particularly some community / segment focussed comment ( e.g jews,muslims,homo,fagot,gay,communist

not able to get for toxic,obscene,insult , very mixed they are 
'''


# length might help

train['lens'] = train['comment_text'].apply(lambda x:len(x.split(" ")))
'''
print train[['lens','toxic']].groupby(['toxic']).describe()
print train[['lens','severe_toxic']].groupby(['severe_toxic']).describe()
print train[['lens','obscene']].groupby(['obscene']).describe()
print train[['lens','threat']].groupby(['threat']).describe()
print train[['lens','insult']].groupby(['insult']).describe()
print train[['lens','identity_hate']].groupby(['identity_hate']).describe()
'''

#print train.groupby(['toxic','severe_toxic','obscene','threat','insult','identity_hate']).describe()['lens'].sort_values(['50%'])[['50%','count']]
'''
for i in xrange(2):
    print train[(train.toxic == 1) & (train.severe_toxic == 1) & (train.obscene == 1) & (train.threat == 0) & (
    train.insult == 0) & (train.identity_hate == 1)].iloc[i][1]
    print "--------------------"

'''


import nltk

txt = train.comment_text.str.lower()
print txt[0]
words = nltk.tokenize.word_tokenize(txt[0])
word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords)
print words_except_stop_dist