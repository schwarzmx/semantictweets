#!/usr/bin/env python

import tweetstream as ts
import pymongo as pm
import sys

username = 'user'
password = 'pass'
words = []
location = ['-124.4','32.5','-66','47.5'] # from the US (california to maine) to try to restrict to English tweets

if len(sys.argv) < 3:
    sys.exit("Usage: " + sys.argv[0] + " <user> <password> [<space separated search terms>]" )
else:
    username = sys.argv[1]
    password = sys.argv[2]
    words = None

    if len(sys.argv) > 3:
        words = []
        for i in range(3,len(sys.argv)):
            words.append(sys.argv[i])

    print words

connection = pm.Connection()
db = connection.corpus

with ts.FilterStream(username, password, locations=location, track=words) as stream:
    for tweet in stream:
#        print tweet['user']['screen_name'] + ': ' + tweet['text'], "\n"
#        print "----" * 4
        db.tweets.save(tweet)
        
