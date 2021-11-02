import pandas as pdimport numpy as npimport refrom keras import modelsfrom keras.models import Sequential from clean import TextDatafrom sklearn.feature_extraction.text import CountVectorizerfrom sklearn.model_selection import train_test_splitfrom sklearn.linear_model import LogisticRegressionfrom sklearn import metricsfrom sklearn.naive_bayes import MultinomialNBfrom nltk.stem import PorterStemmerfrom keras.datasets import imdbfrom keras import preprocessingimport numpy as npfrom keras.models import Sequential from keras import layersimport matplotlib.pyplot as pltfrom tensorflow.keras.optimizers import RMSpropfrom sklearn.preprocessing import OneHotEncoderfrom keras.callbacks import CSVLoggerclass TrainingTesting():      def __init__(self, dir = 'clean_data.csv'):        self.data = pd.read_csv(dir)            def stem(str_input):        '''        A stand-alone function that stems a sentence.        Parameters        ----------        str_input : str            A string variable to be stemmed.        Raises        ------        ValueError            Raise ValueError when more than 1 group in token pattern is captured.        Returns        -------        words : lst            Post stemming list.        '''        # codes partially taken from the logic of sklearn's CountVectorizer's source code        # so that all the other parameters would keep the same while being stemmed        token_pattern = re.compile(r"(?u)\b\w\w+\b")            if token_pattern.groups > 1:            raise ValueError(                "More than 1 capturing group in token pattern. Only a single "                "group should be captured."            )            words = token_pattern.findall(str_input)        ps = PorterStemmer()        words = [ps.stem(word) for word in words]        return words            def Vectorize(self, ana = 'word', sw = 'english', lc = True, bi = False, st = False, rare_word = 0):        '''        Process the predictors to prepare for model training        Parameters        ----------        ana : str, optional            Count Vectorizer parameter analyzer. The default is 'word'.        sw : str, optional            Count Vectorizer parameter stop_words. The default is 'english'.        lc : str, optional            Count Vectorizer parameter lowercase. The default is True.        bi : BOOL, optional            Count Vectorizer parameter binary. The default is False.        st: BOOL, optional            Count Vectorizer parameter stemmer. The default is False.        rare_word : int, optional            Set the threshold to remove typo. The default is 1.        Returns        -------        None.        '''        if st:            Vec = CountVectorizer(analyzer = ana,                                  stop_words = sw,                                  lowercase = lc,                                  binary = bi,                                  tokenizer = TrainingTesting.stem)        else:            Vec = CountVectorizer(analyzer = ana,                                  stop_words = sw,                                  lowercase = lc,                                  binary = bi)        data_X_t = Vec.fit_transform(self.data.X.tolist())                features = Vec.get_feature_names()                temp = pd.DataFrame(data_X_t.toarray(), columns = features)        temp = temp.loc[:, temp.sum(axis = 0) > rare_word]                temp['y'] = self.data.y                self.vec_data = temp                print(self.data.head)            def split(self, test_percent = 0.2, val_percet = 0.2):        '''        Split data into training and testing.        Parameters        ----------        test_percent : float, optional            Percentage split into testing set. The default is 0.2.        val_percet : float, optional            Percentage split into validation set from training set. The default is 0.2.        Returns        -------        None.        '''                onehot_encoder = OneHotEncoder(sparse=False)                train, test = train_test_split(self.vec_data, test_size = test_percent)                train, val = train_test_split(train, test_size = val_percet/(1-test_percent), random_state=1)                train_y = train["y"]                train_y = np.array(train_y)                train_y = train_y.reshape(len(train_y), 1)        self.train_y = onehot_encoder.fit_transform(train_y)                             self.train_X = train.drop(["y"], axis = 1)                test_y = test["y"]                test_y = np.array(test_y)                test_y = test_y.reshape(len(test_y), 1)        self.test_y = onehot_encoder.fit_transform(test_y)                    self.test_X = test.drop(["y"], axis = 1)                val_y = val["y"]                val_y = np.array(val_y)                val_y = val_y.reshape(len(val_y), 1)        self.val_y = onehot_encoder.fit_transform(val_y)                            self.val_X = val.drop(["y"], axis = 1)                    def report(history, title=''):        '''        A class method to generate plots        Parameters        ----------        history : keras_history            Data used to plot.        title : str, optional            Option to change plot title. The default is ''.        Returns        -------        None.        '''        epochs = range(1, len(history.history['loss']) + 1)        plt.figure()        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')        plt.title(title)        plt.legend()        plt.savefig('HISTORY-'+title+'.png')         plt.show()        plt.close()            def CNN1D(self,               max_features = 10000,               embed_dim = 10,               maxlen = 1000,               lr = 0.001,               epochs = 20,               batch_size = 100,               verbose = 1,              dropout_rate = 0.5,              plot = True,              save_model = True):        '''        A class method to train CNN 1D.        Parameters        ----------        max_features : int, optional            Parameter. The default is 10000.        embed_dim : int, optional            Parameter. The default is 10.        maxlen : int, optional            Parameter. The default is 1000.        lr : float, optional            Parameter. The default is 0.001.        epochs : int, optional            Parameter. The default is 20.        batch_size : int, optional            Parameter. The default is 100.        verbose : int, optional            Parameter. The default is 1.        dropout_rate : float, optional            Parameter. The default is 0.5.        plot : boolean, optional            Parameter. The default is True.        save_model : boolean, optional            Parameter. The default is True.        Returns        -------        None.        '''                model = Sequential()        model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))        model.add(layers.Conv1D(32, 7, activation='relu'))        model.add(layers.MaxPooling1D(5))        model.add(layers.Dropout(dropout_rate))        model.add(layers.Conv1D(32, 7, activation='relu'))        model.add(layers.GlobalMaxPooling1D())        model.add(layers.Dropout(dropout_rate))        model.add(layers.Dense(3))        model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc'])                 print(model.summary())         self.logCNN1D = CSVLogger('logCNN1D.txt', append=True, separator=';')                           history = model.fit(self.train_X, self.train_y,                            epochs = epochs,                            batch_size = batch_size,                             validation_data = (self.val_X, self.val_y),                            verbose = verbose,                            callbacks = self.logCNN1D)                self.history = history                        if plot:            TrainingTesting.report(history,title="CNN")        if save_model:            model.save('CNN1D.h5')                def SimpleRNN(self,                   max_features = 10,                   embed_dim = 5,                   maxlen = 1000,                   lr = 0.001,                   epochs = 20,                   batch_size = 10,                   verbose = 1,                  dropout_rate = 0.5,                  plot = True,                  save_model = True):        '''        A class method to train Simple RNN.        Parameters        ----------        max_features : int, optional            Parameter. The default is 10000.        embed_dim : int, optional            Parameter. The default is 10.        maxlen : int, optional            Parameter. The default is 1000.        lr : float, optional            Parameter. The default is 0.001.        epochs : int, optional            Parameter. The default is 20.        batch_size : int, optional            Parameter. The default is 100.        verbose : int, optional            Parameter. The default is 1.        dropout_rate : float, optional            Parameter. The default is 0.5.        plot : boolean, optional            Parameter. The default is True.        save_model : boolean, optional            Parameter. The default is True.        Returns        -------        None.        '''                        model = Sequential()         model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))        model.add(layers.SimpleRNN(32))        model.add(layers.Dropout(dropout_rate))        model.add(layers.Dense(3, activation='sigmoid'))        model.compile(optimizer = RMSprop(lr = lr),                       loss = 'categorical_crossentropy',                      metrics = ['acc'])         model.summary()                self.logSimpleRNN = CSVLogger('logSimpleRNN.txt', append=True, separator=';')                           history = model.fit(self.train_X, self.train_y,                            epochs = epochs,                            batch_size = batch_size,                             validation_data = (self.val_X, self.val_y),                            verbose = verbose,                            callbacks = self.logSimpleRNN)                self.history = history                if plot:            TrainingTesting.report(history,title="SimpleRNN")        if save_model:            model.save('SimpleRNN.h5')                def evaluate_model(self, model_type):        '''        A class method to evaluate model        Parameters        ----------        model_type : string            Flag to tell whcih model to evaluate.        Returns        -------        None.        '''        if model_type == 'SimpleRNN':            model = models.load_model('SimpleRNN.h5')            test = model.evaluate(self.test_X, self.test_y)            print('loss:', test[0])            print('accuracy:', test[1])        elif model_type == 'CNN1D':            model = models.load_model('CNN1D.h5')            test = model.evaluate(self.test_X, self.test_y)            print(model_type)            print('loss:', test[0])            print('accuracy:', test[1])                        if __name__ == '__main__':    tt1 = TrainingTesting()    tt1.Vectorize()    tt1.split()    tt1.CNN1D()    tt1.SimpleRNN()        