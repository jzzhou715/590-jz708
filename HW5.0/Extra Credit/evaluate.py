from keras import models
import pandas as pd
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


from train import TrainingTesting
           
if __name__ == '__main__':
    tt1 = TrainingTesting()
    tt1.Vectorize()
    tt1.split()
    tt1.CNN1D()
    tt1.SimpleRNN()
    tt1.evaluate_model('CNN1D')
    tt1.SimpleRNN('SimpleRNN')
    
    
