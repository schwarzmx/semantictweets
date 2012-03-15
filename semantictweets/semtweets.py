# -*- coding: utf-8 -*-

import json
from vspace import *
from lsi import *
from kmeans import *

class SemanticTweets:
    """ This class is the application controller.
    It manages the models for indexing the tweets and clustering them.
    """

    corpus_path = "corpus/corpus.json"
    tweets = []

    def __init__(self, samplesize=50):
        self.load_tweets(samplesize)

    def load_tweets(self, samplesize):
        """ This method parses the corpus.json file.
        It selects 'samplesize' many tweets from the corpus.
        
        TODO: do the selection randomly
        """

        corpus = open(self.corpus_path, 'r')
        tweets_json = json.loads(corpus.read())
        corpus.close()

        self.tweets = [t['text'] for t in tweets_json]

    def run_model(self):
        """ This method is in charge of indexing the tweets
        and running the clustering method.
        """

        vector_space = VSpace(self.tweets)

        print vector_space.term_index

        lsi = LSI(vector_space.doc_vectors)
        lsi.compute_tfidf()
        lsi.rank_reduced_svd(k=500)  # empirically decide to how many dimensions to reduce

        kmeans = KMeans(lsi.sem_space)

        clusters = kmeans.cluster(k=50)

        # sort clusters by number of documents
        clusters = sorted(clusters, cmp=lambda x,y: cmp(len(y.documents),len(x.documents)))

        print "clusters: %d" % len(clusters)
        for cluster in clusters:
            print "\n\n\tdocuments: %d" % len(cluster.documents)

            for document in cluster.documents:
                #print "\t\tindex: %d" % document.index
                print "\t\ttweet: %s" % self.tweets[document.index]

        total = reduce(lambda x, y: x+y, [len(cluster.documents) for cluster in clusters])
        print "total documents: %d" % total
        

