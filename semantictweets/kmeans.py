# -*- coding: utf-8 -*-

from scipy import dot, mat, linalg, array, random, add
from math import *
from copy import copy

class KMeans:
    """ This class represents the K-Means clustering algorithm.
    It takes the semantic space from the LSI model and computes
    the clustering algorithm.
    """

    dimensions = -1
    documents = []

    def __init__(self, sem_space):

        rows, cols = sem_space.shape
        print "dimensions: %d" % rows
        self.dimensions = rows

        self.documents = [Document(sem_space[:, c], c) for c in xrange(0,cols)]


    def cluster(self, k=3, max_iter=10):
        """ Perform K-Means algorithm.
        K is the number of clusters. It is done for a max of
        max_iter iterations.
        """

        # create a set of k random clusters as seeds
        old_clusters = [None] * k # just a placeholder
        clusters = self.random_clusters(k)

        iter = 0
        while (iter < max_iter) and not (old_clusters == clusters):
            print "iteration %d..." % iter
            # assign new clusters to old clusters
            for i in xrange(0, k):
                old_clusters[i] = copy(clusters[i])
                clusters[i].documents = []

            # for each document
            for document in self.documents:

                # determine the cluster with the highest similarity
                similarities = [cosine_similarity(document, cluster) for cluster in old_clusters]
                max_index = array(similarities).argmax()

                # assign document to that cluster
                clusters[max_index].add(document)

            # update cluster means
            for cluster in clusters:
                cluster.update_centroid()
            
            iter += 1
            
        return clusters

    def random_clusters(self, k):
        clusters = []

        for i in range(0,k):
            clusters.append(Cluster(i,self.dimensions))

        return clusters

class Cluster:
    """ This class represents a cluster.
    Each cluster contains an N dimensional vector, where N is the
    number of documents.
    """

    # to keep a tag for each cluster
    index = -1

    # a list of documents attached to this cluster
    documents = []

    # the mean of the cluster
    centroid = None

    def __init__(self, tag, dimensions):
        self.index = tag
        self.centroid = array(10*random.standard_normal(dimensions))

    def add(self, document):
        self.documents.append(document)

    def update_centroid(self):
        """ Update the mean of the current cluster """

        total_docs = len(self.documents)

        if total_docs == 0:
            return

        new_centroid = self.centroid.copy()
        for document in self.documents:
            new_centroid = new_centroid + document.vector

        self.centroid = new_centroid / float(total_docs)

    def __eq__(self, other):
        """ Override rich comparison method.
        It compares the indices of the documents in the self.documents list.
        Returns True if the indices are the same in both documents lists, 
        False otherwise.
        """

        if not self or not other: #either one of them is null
            return False

        if len(self.documents) != len(other.documents):
            return False

        for i in xrange(0, len(self.documents)):
            if self.documents[i].index != other.documents[i].index:
                return False

        return True

class Document:

    # the internal model of the document
    vector = None 
    # for labeling the vectors
    index = -1  

    def __init__(self, vector, index):
        self.vector = vector.copy().T
        self.index = index

def cosine_similarity(document, cluster):
    """ Helper function for calculating cosine similarities between 
    a document vector and a cluster centroid.
    """
    num = dot(document.vector, cluster.centroid)
    den = linalg.norm(document.vector) * linalg.norm(cluster.centroid)

    return num / den


        
