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

    # results from the rank-reduced SVD
    sigma_k = None
    T_k = None
    D_k = None

    # the semantic space is the representation we are going to use for
    # k-means, which is sigma_k * D_k
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
        We transpose the Vector Space obtained in the previous steps to obtain
        the usual form for LSI. Then the SVD is computed over the space
        to obtain the Terms, Singular and Document components (matrices).
        Then those are truncated to obtain the reduced space.
        An additional step is to compute the semantic space for the documents,
        which is multiplying Sigma_k * D_k, which will be used by K-Means to
        cluster the documents by similarity of terms.
        """

        rows, cols = self.vspace.T.shape

        if k <= rows: # vspace must be truncatable, i.e. k > num_docs

            # compute the regular SVD
            # use the transpose to get the form of X as it is usual
            # with document vectors and term rows
            U, s, VT = linalg.svd(self.vspace.T) 

            
            # one approach is to truncate the SVD and then multiplying
            # truncating has the advantage of being computationally simpler
            # we just slice sigma
            self.sigma_k = s[0:k].copy() # in numpy slices are by reference
            self.T_k = U[:,0:k].copy() # eliminate last columns
            self.D_k = VT[0:k].copy() # eliminate last rows

#            print "Sigmas: " 
#            print str(s)
#            print str(self.sigma_k)
#
#            print "Terms: "
#            print prettify(U)
#            print prettify(self.T_k)
#
#            print "Documents: "
#            print prettify(VT)
#            print prettify(self.D_k)

            S = linalg.diagsvd(self.sigma_k, k, k)
            reduced_space = dot(dot(self.T_k, S), self.D_k)

            self.sem_space = dot(S, self.D_k)

#            # the other approach is to zero out the lowest values of sigma
#            # conveniently scipy already sorts the singular values
#            # zeroing out the lowest values of sigma produces
#            for i in xrange(rows - k, rows):
#                s[i] = 0
#
#            # compute reduced matrix to obtain the rank-reduced vector space
#            S = linalg.diagsvd(s, len(self.vspace), len(VT))
#            reduced_space = dot(dot(U, S), VT)
#            self.sem_space = dot(S, VT)

            self.vspace = reduced_space
            

        else:
            raise Exception('K must be smaller than the number of terms!')



    
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
