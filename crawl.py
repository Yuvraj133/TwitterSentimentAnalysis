"""Author:Yuvraj Singh Bisht and Rohit Kumar
Date: 1/11/18
Name: training.py
Function: This module is used for fetching tweets using tweepy.
Input: consumer_key, consumer_secret, access_token, access_token_secret.
Output: The result of crawling the tweets and is dumped in ua.csv .
"""

import tweepy
import csv


# input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# Open/Create a file to append data
csvFile = open('ua.csv', 'a')

# Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search, q="#siddaramaiah", count=100,
                           lang="en",
                           since="2018-04-09").items():
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

