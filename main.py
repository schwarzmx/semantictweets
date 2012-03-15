# -*- coding: utf-8 -*-

import sys
sys.path.append('./semantictweets')
sys.path.append('./lib')

from semtweets import *
from kmeans import *

# load the tweets from the corpus
print "Loading tweets from file..."
sem_tweets = SemanticTweets()

# run the model, this should take no more than 15 min 
# (tested on a 2 year old Mac)
print "Computing model... \nThis can take a while (10-15min), please sit back..."
clusters = sem_tweets.run_model()

# prepare clusters to save them in a file
c = 0
bar = '******************************************************************************************************************************************************\n'
output = bar
for cluster in clusters:
    output += "* Cluster: %d. Total tweets: %d\n" % (c, len(cluster.documents))

    for document in cluster.documents:
        output += "*   tweet: %s\n" % sem_tweets.tweets[document.index]

    output += bar 
    c += 1

# save results in a file
import os

file_path = './clusters.txt'

if os.path.exists(file_path):
    os.remove(file_path)

file = open(file_path, 'w')
file.write(output.encode('utf-8'))
file.close()
