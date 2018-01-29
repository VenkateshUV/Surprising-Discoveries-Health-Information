import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models
import os
import re
import numpy as np

def tokenize(text):
    
    #Removing punctuation marks
    tokenizer = RegexpTokenizer(r'\w+')
    terms = tokenizer.tokenize(text)
        
    #Removing stop words by filtering the tokens and by checking if it is not in stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [term.lower() for term in terms if term.lower() not in stop_words]
    filtered_tokens = []
    
    # filtering only words by checking if it contains only alphabets
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def process_documents(path):

    # create English stop words list and remove stop words
    stop_words = set(stopwords.words('english'))

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    # list for tokenized documents in loop
    texts = []
    
    listing = os.listdir(path)
    
    count = 0
    for infile in listing:
        
        #Training the model with a set of 1000 documents
        count += 1
        if (count == 10001):
            break
        
        #Retrieving the contents of all the documents
        raw_doc = open(path + '\\' + infile, encoding="utf8").read()
        
        tokens = tokenize(raw_doc.lower())

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stop_words]
    
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word = dictionary, passes=20)
    
    print(ldamodel.print_topics(num_topics=10, num_words=10))
    
    #finding similarity between probability distribution in documents given the top topics in the corpus and 
    #a likelihood values assumed to be 0.7
    for i in range(1, len(corpus)):
        dense1 = gensim.matutils.sparse2full(ldamodel[corpus[i-1]], ldamodel.num_topics)
        dense2 = gensim.matutils.sparse2full(ldamodel[corpus[i]], ldamodel.num_topics)
        sim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
        print('Similarity : document {} and {} is {}'.format(i, i+1, sim))
        
# Call to process documents    
process_documents('C:\\PK\\Fall 17\\KDD\\FinalProject\\diabetes\\diabetes')
