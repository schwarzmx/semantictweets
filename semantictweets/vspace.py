# -*- coding: utf-8 -*-

class VSpace:
    """ This represents a document matrix in which documents are the columns 
    and each row is a term in the document. If the term exists in the document
    it has a non-zero value in the corresponding row.
    """

    # A list of documents
    doc_vectors = []

    # A mapping of <term, index> of all the words in the corpus
    term_index = []

    # A helper class for manipulating document strings
    tokenizer = None

    def __init__(self, docs=[]):
        self.documents = []
        self.tokenizer = Tokenizer()

        if len(docs) > 0:
            self.build_space(docs)

    def build_space(self, docs):
        """ Create the vector space for the current documents """
        self.term_index = self.get_word_index(docs)

        self.doc_vectors = [self.create_vector(document) for document in docs]

    def get_word_index(self, docs):
        # create a single string with all the documents
        docs_string = " ".join(docs)

        # get a nice list of terms without stop words
        vocabulary = self.tokenizer.tokenize(docs_string)

        doc_index = {}
        index = 0

        # create a map of every word to a specific dimension of the "full document" space
        for word in vocabulary:
            doc_index[word] = index
            index += 1

        return doc_index

    def create_vector(self, doc):
        """ create a term vector for the current document """

        # initialize in zero, use floats
        vector = [0.0] * len(self.term_index) 
        terms = self.tokenizer.tokenize(doc)

        for term in terms:
            # simple word count, use floats for LSI
            vector[self.term_index[term]] += 1.0 

        return vector

class Tokenizer:
    """ Helper class for tokenizing document space and removing stop words """

    corpus = None
    terms = []
    stop_words = []

    def __init__(self):

        # read stop words from file
        self.stop_words = open('stop_words.txt', 'r').read().split()

    def tokenize(self, docs_string):
        """ Tokenizer's most important method.
        It separates the whole corpus string in tokens and
        removes stop words.
        TODO: implement stemmer(?)
        """
        self.corpus = docs_string

        self.clean()

        self.terms = self.corpus.split(" ")

        self.remove_stop_words()

        self.remove_duplicates()

        return self.terms

    def clean(self):
        """ get rid of punctuation signs, convert to lower case, standardize spacing """
        self.corpus = self.corpus.replace(".", " ")
        self.corpus = self.corpus.replace(",", " ")
        self.corpus = self.corpus.lower()
        self.corpus = self.corpus.replace("\s+", " ")

    def remove_stop_words(self):
        self.terms = [term for term in self.terms if term not in self.stop_words]

    def remove_duplicates(self):
        """ remove duplicated terms in the list """
        from sets import Set
        self.terms = Set((term for term in self.terms))
