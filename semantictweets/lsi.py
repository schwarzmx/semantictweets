# -*- coding: utf-8 -*-
from scipy import dot, mat, linalg, array
from math import *
from pprint import pprint

class LSI:
    """Latent Semantic Analysis (LSI) model.
    This class will use the vector space (VSpace) class for applying
    latent semantic analysis which will be used further by K-Means for 
    clustering.
    It makes use of SciPy/NumPy for computing various transformations.
    """

    vspace = None
    sigma_k = None
    U_k = None
    VT_k = None
    sem_space = None

    def __init__(self, vspace):
        """ Initialize with the vspace obtained in previous step """
        self.vspace = array(vspace)

    def compute_tfidf(self):
        """ Transforms the current space to a tf*idf matrix for 
        better results.
        """

        docs, terms = self.vspace.shape
        docs_total = len(self.vspace)
        
        for doc in xrange(0, docs): 
            # iterate over documents
            
            # count words in a doc
            words_in_doc = reduce(lambda x, y: x + y, self.vspace[doc])

            for term in xrange(0, terms):
                # iterate over terms in document

                if self.vspace[doc][term] != 0:

                    # in how many docs does the term appear?
                    term_occurrences = self.term_occurence(term)

                    term_freq = float(self.vspace[doc][term]) / float(words_in_doc)
                    inv_doc_freq = log(abs(docs_total / term_occurrences))
                    self.vspace[doc][term] = term_freq * inv_doc_freq
                
    def rank_reduced_svd(self, k=3):
        """ Computes the rank reduced SVD.
        This method computes the SVD of a truncated space for the K highest
        singular values of sigma.
        First, the SVD for the current vector space is computed, then
        the top k values of sigma are selected, then the semantic
        space is computed from the reduced sigma.
        """

        rows, cols = self.vspace.shape

        if k <= rows: # vspace must be truncatable, i.e. k > num_docs

            # compute the regular SVD
            U, s, VT = linalg.svd(self.vspace) 

            print "sigmas: " + " length : %d" % len(s)
            print str(s)
            # ...conveniently scipy already sorts the singular values
            # we just slice sigma
            self.sigma_k = s[0:k].copy() # numpy slices are by reference
            self.U_k = U[0:k].copy()
            self.VT_k = VT[0:k].copy()

            print str(self.sigma_k)

            print "Us: "
            print prettify(U)
            print prettify(self.U_k)
            print "VTs: "
            print prettify(VT)
            print prettify(self.VT_k)

#            # truncate sigma by making zero the lower values
#            for i in xrange(rows - k, rows):
#                s[i] = 0
#
#            # compute truncated matrix to obtain the semantic space
#            sem_space = dot(dot(U, linalg.diagsvd(s, len(self.vspace), len(VT))), VT)
#
#            self.vspace = sem_space
            

        else:
            raise Exception('K must be smaller than the number of documents!')



    
    def term_occurence(self, term):
        """ computes in how many documents the term appears """

        occurrences = 0

        docs, terms = self.vspace.shape

        for doc in xrange(0, docs):

            if self.vspace[doc][term] != 0:
                # the term is in the document
                occurrences += 1.0 # use floats for convenience

        return occurrences

    def __repr__(self,):
        """ pretty print """
        rep = ""
         
        rows,cols = self.vspace.shape
         
        for row in xrange(0,rows):
            rep += "["
         
            for col in xrange(0,cols):
                rep += "%+0.2f "%self.vspace[row][col]

            rep += "]\n"
         
        return rep

def prettify(array):
    rep = ""
         
    rows,cols = array.shape
     
    for row in xrange(0,rows):
        rep += "["
     
        for col in xrange(0,cols):
            rep += "%+0.2f "% array[row][col]

        rep += "]\n"
     
    return rep
