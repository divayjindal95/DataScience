import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from  sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


train_data = pd.read_csv("../data/train.csv")
train_data_len=len(train_data)
test_data=pd.read_csv("../data/test.csv")
test_data_len=len(test_data)

