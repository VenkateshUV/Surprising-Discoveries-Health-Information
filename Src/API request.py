#!/usr/bin/python
# -*- coding: utf-8 -*-
from requests import get
import nltk
import os
import glob
import codecs
import csv
import progressbar
import re
import pandas as pd
import nltk.data
#from phrase import PhraseVector

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"

def sss(s1, s2, type='relation', corpus='gigawords'):
    try:
        response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
        return float(response.text.strip())
    except:
        print 'Error in getting similarity for %s: %s' % ((s1,s2), response)
        return 0.0

location = "/Users/Nikhil/Documents/Courses/Knowledge Discovery in Databases/Final Project/diabetes/"
os.chdir("/Users/Nikhil/Documents/Courses/Knowledge Discovery in Databases/Final Project/diabetes/")
stopWords = set(nltk.corpus.stopwords.words('english'))
data = glob.glob("*.txt")

bar = progressbar.ProgressBar(max_value=len(data))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

FinalFile = open("../finalOutput.csv",'w')
outputFileWriter = csv.writer(FinalFile)

for j, file in enumerate(data):
    bar.update(j)
    openfile = codecs.open(file,encoding='utf-8')
    outputFile = open("../output2/"+str(file).replace('.txt','.csv'),'w')
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(["sentence1","sentence2","score"])
    fileData = openfile.read()
    sentences = tokenizer.tokenize(fileData)
    for i in range(1,len(sentences)):
        if ( i < (len(sentences)-2)):
            words1 = nltk.word_tokenize(sentences[i].strip('\n'))
            filtered_sentence1 = ' '.join([w for w in words1 if not w in stopWords])
            words2 = nltk.word_tokenize(sentences[i+1].strip('\n'))
            filtered_sentence2 = ' '.join([w for w in words2 if not w in stopWords])
            score  = sss(filtered_sentence1,filtered_sentence2)
            if(score != 0):
                csvWriter.writerow([sentences[i].encode("utf-8"),sentences[i+1].encode("utf-8"),score])
    outputFile.close()
    openfile.close()
    df = pd.read_csv("../output2/"+str(file).replace('.txt','.csv'))
    results = df['score'].describe()
    try:
        threshold = results['25%']
        scores = df['score']
        #print threshold
        totalCount = results['count']
        surpriseCount = 0
        index_pairs = ""
        for i in range(0,len(scores)):
            if (i < (len(scores) - 1)):
                if (scores[i] < threshold):
                    if (scores[i+1] < threshold):
                        surpriseCount += 1
                        index_pairs = index_pairs + str(i) + ',' + str(i + 1) + '|'
        if(surpriseCount != 0):
            surpriseScore = surpriseCount/totalCount
            outputFileWriter.writerow([str(file),totalCount,surpriseCount,index_pairs,surpriseScore,threshold])
    except KeyError:
        print results
FinalFile.close()
