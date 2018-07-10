"""Author:Yuvraj Singh Bisht and Rohit Kumar
Date: 1/16/18
Name: NaiveBayes.py
Function: This module is used for defining, training
and testing the NaiveBayes classifier for sentiment prediction.
Input: The training dataset from source.
Output: The result of tesing and
the trained classifier
"""

# import sys
import csv
# import tweepy
import re
import nltk
# import string
# from nltk.classify import *
# from tweepy.streaming import StreamListener
# from tweepy import OAuthHandler
# from tweepy import Stream
# from nltk.corpus import stopwords
import nltk.classify.util


# look for 2 or more repetitions of character and replace with the character itself
def replacetwoormore(s):
    pattern = re.compile(r"(.)\1+", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# starting the function
def getstopwordlist(stopwordlistfilename):
    # read the stopwords file and build a list
    stopwords = []
    # stopwords.append('TWITTER_USER')
    stopwords.append('URL')

    fp = open(stopwordlistfilename, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopwords.append(word)
        line = fp.readline()
    fp.close()
    return stopwords


st = open('StopWords.txt', 'r')
stopwords = getstopwordlist('StopWords.txt')


# starting the function
def getfeaturevector(tweet):
    featurevector = []
    # split tweet into words
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replacetwoormore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        # ignore if it is a stop word
        if w in stopwords or val is None:
            continue
        else:
            featurevector.append(w.lower())
    return featurevector


# starting the function
def featureextraction():
    # Here I am reading the tweets one by one and process it
    f = open(r'/home/yuvraj/PycharmProjects/twitter_analysis/training_test.csv', 'r', encoding='ISO-8859-1')
    inptweets = csv.reader(f)
    tweets = []
    # inptweets = csv.reader(open('ua.csv', 'rb', encoding='ISO-8859-1'), delimiter=',', quotechar='|')

    for rowTweet in inptweets:
        sentiment = rowTweet[0]
        tweet = rowTweet[1]
        featurevector = getfeaturevector(tweet)
        tweets.append((featurevector, sentiment))
    # print "Printing the tweets con su sentiment"
    # print tweets
    # Here I am returning the tweets inside the array plus its sentiment
    return tweets


# end

tweets = featureextraction()


# print tweets

# Classifier
def get_words_in_tweets(tweets):
    all_words = []
    for (text, sentiment) in tweets:
        all_words.extend(text)
    return all_words


def get_word_features(wordlist):
    # This line calculates the frequency distrubtion of all words in tweets
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()

    # This prints out the list of all distinct words in the text in order
    # of their number of occurrences.
    return word_features


word_features = get_word_features(get_words_in_tweets(tweets))  # my list of many words


def extract_features(tweet):
    settweet = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in settweet)
    return features


# Here I am creating my Training set.
# I extract feature vector for all tweets in one shot

training_set = nltk.classify.apply_features(extract_features, tweets)
test_set = nltk.classify.apply_features(extract_features, tweets[:250])

# ****** Naive Bayes Classifier******************************************

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Accuracy
accuracy = nltk.classify.accuracy(classifier, training_set)

# Printing the accuracy
print(accuracy)

total = accuracy * 100
print('Naive Bayes Accuracy: %4.2f' % total)

# Accuracy Test Set
accuracyTestSet = nltk.classify.accuracy(classifier, test_set)

# Printing the accuracy for the test set
print(accuracyTestSet)

totalTest = accuracyTestSet * 100
print('\nNaive Bayes Accuracy with the Test Set: %4.2f' % totalTest)


