from termcolor import colored, cprint
import pandas as pd
import nltk
#nltk.download('stopwords')
from nltk import tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix



class KaggleAction:
    def __init__(self, train_filename, test_filename, groundtruth_filename=None):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.groundtruth_filename = groundtruth_filename
        self.predictor_labels = []
        self.target_label = None
        self.pipelines = {}
        self.fitted = None
        self.encodings = {}

    def _fix_na(self, data, label):
        is_numeric = data[label].dtype.kind not in ['O', 's', 'S', 'u']
        if is_numeric:
            data[label] = data[label].fillna(data[label].mean(), axis=0)
        else:
            data[label] = data[label].fillna(data[label].mode(), axis=0)
        return data

    def train(self):
        print("\nTRAINING...")
        train_data = pd.read_csv(self.train_filename)
        labels = train_data.columns
        self.predictor_labels = labels

        print('Features:')
        for i, label in enumerate(labels):
            print('  {}) {}'.format(i, label))

        idx = int(input('Select target feature [ENTER]: '))

        self.target_label = list(train_data.columns)[idx]
        self.predictor_labels = self.predictor_labels.drop(self.target_label)
        train_data = self._fix_na(train_data, self.target_label)
        target_data = train_data[[self.target_label]]

        self.encodings = {}
        steps = []
        for label in labels:
            print(label)
            train_data = self._fix_na(train_data, label)
            label_data = train_data[[label]]

            if train_data[label].dtype.kind in ['s', 'S', 'O', 'u']:
                encoding = pd.factorize(train_data[label])
                train_data[label] = encoding[0]
                if len(encoding[1]) >= 0.95 * len(train_data[label]):
                    train_data = train_data.drop(columns=label)
                    self.predictor_labels = self.predictor_labels.drop(label)
                else:
                    self.encodings[label] = encoding[1]
                """
                corpus = ' '.join(train_data[label].to_list())
                corpus = WhitespaceTokenizer().tokenize(text=corpus)
                vocabulary = set(corpus)

                count_step = (label, CountVectorizer(vocabulary=vocabulary))
                tfidf_step = ('tfidf', TfidfTransformer())
                #steps.append(('classifier', MLPClassifier()) )
                pipeline = Pipeline(steps=[count_step, tfidf_step], verbose=True)
                """
        #sgd_step = ('sgd', SGDClassifier())
        dtree_step = ('dtree', DecisionTreeClassifier())
        pipeline = Pipeline(steps=[dtree_step], verbose=True)
        self.fitted = pipeline.fit(X=train_data[self.predictor_labels], y=target_data)
        score = self.fitted.score(X=train_data[self.predictor_labels], y=target_data)
        cprint('SCORE:  {}'.format(score), attrs=['bold'])
        print()


    def test(self):
        print("\nTESTING...")

        test_data = pd.read_csv(self.test_filename)
        print(test_data.head())
        labels = test_data.columns
        for label in labels:
            if label not in self.predictor_labels:
                test_data = test_data.drop(columns=label)
            else:
                test_data = self._fix_na(test_data, label)

                if label in self.encodings:
                    encoding = pd.factorize(test_data[label])
                    test_data[label] = encoding[0]

        predictions = self.fitted.predict(test_data)

        #print(predictions)
        df = pd.DataFrame({labels[0]: test_data[labels[0]].to_list(), self.target_label: predictions})
        df.to_csv(path_or_buf='./results.csv', sep=',', index=False)

        if self.groundtruth_filename is not None:
            groundtruth = pd.read_csv(self.groundtruth_filename)
            print(groundtruth)
            print("\nConfusion Matrix :")
            matrix = confusion_matrix(y_true=groundtruth[self.target_label], y_pred=predictions)
            print(matrix)
            accuracy = accuracy_score(y_true=groundtruth[self.target_label], y_pred=predictions)
            print('Accuracy:  {}'.format(accuracy))

