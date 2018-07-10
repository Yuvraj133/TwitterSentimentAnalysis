"""Author:Yuvraj Singh Bisht and Rohit Kumar
Date: 1/30/18
Name: training.py
Function: This module is used for defining, training
and testing the SVM classifier and then dumping it as a pickle
file which can be reused for sentiment prediction.
Input: The training datasets from various sources.
Output: The result of tesing(recall,precision, F1-score etc) and
the trained classifier dumped in svmClassifier.pkl
"""

# Import required libraries

import csv
import sklearn.metrics
import sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline


# Generating the Training and testing vectors

def getdata():
    x = []
    y = []
    y = []
    y = []

    # Training data
    f = open(r'/home/yuvraj/PycharmProjects/twitter_analysis/training_test.csv', 'r', encoding='ISO-8859-1')
    reader = csv.reader(f)

    for row in reader:
        x.append(row[5])
        y.append(1 if (row[0] == '4') else 0)

    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(x, y, test_size=0.20,
                                                                                 random_state=42)
    return x_train, x_test, y_train, y_test


# Process Tweets (Stemming(root words)+Pre-processing)

def processtweets(x_train, x_test):
    x_train = [sentiment.stem(sentiment.cleantweets(tweet)) for tweet in x_train]
    x_test = [sentiment.stem(sentiment.cleantweets(tweet)) for tweet in x_test]
    return x_train, x_test


# SVM classifier

def classifier(x_train, y_train):
    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    svm_clf = svm.LinearSVC(C=0.1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(x_train, y_train)
    joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
    return vec_clf


# Main function

def main():
    x_train, x_test, y_train, y_test = getdata()
    x_train, x_test = processtweets(x_train, x_test)
    vec_clf = classifier(x_train, y_train)
    y_pred = vec_clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
