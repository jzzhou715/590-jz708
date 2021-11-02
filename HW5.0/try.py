#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:41:46 2021

@author: Zhou
"""

import pandas as pd

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import re
from keras.models import Sequential 
from clean import TextData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from keras.datasets import imdb
from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder


# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)

onehot_encoder = OneHotEncoder(sparse=False)
# onehot_encoder(tt1.test_y)

df = pd.read_csv('clean_data.csv')

a = df['y']

a = array(a)
a=a.reshape(len(a), 1)

a=onehot_encoder.fit_transform(a)      

max_features = 10000
embed_dim = 10
maxlen = 1000 
lr = 0.001 
epochs = 20
batch_size = 100
verbose = 1
plot = True


model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='ReLU'))
model.add(layers.MaxPooling1D())
model.add(layers.Dense(3))
model.compile(optimizer=RMSprop(lr = lr), loss='categorical_crossentropy', metrics=['acc']) 

print(model.summary())



model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(3, activation='sigmoid'))
model.compile(optimizer = RMSprop(lr = lr), 
              loss = 'categorical_crossentropy',
              metrics = ['acc']) 
model.summary()

model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 

model.summary()            
