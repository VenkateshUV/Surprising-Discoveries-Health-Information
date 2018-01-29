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
import gensim
from gensimphrase import PhraseVector
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.corpus import gutenberg
 
print dir(gutenberg)
print gutenberg.fileids()
 
text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id) 
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
 
tokenizer = PunktSentenceTokenizer(trainer.get_params())

tokenizer._params.abbrev_types.add('dr')
location = "/Users/Nikhil/Documents/Courses/Knowledge Discovery in Databases/Final Project/diabetes/"
os.chdir("/Users/Nikhil/Documents/Courses/Knowledge Discovery in Databases/Final Project/diabetes 3/")
stopWords = set(nltk.corpus.stopwords.words('english'))
data = glob.glob("*.txt")
bar = progressbar.ProgressBar(max_value=len(data))
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


#tokenizer.tokenize(text)
FinalFile = open("../output-Google.csv",'w')
outputFileWriter = csv.writer(FinalFile)
outputFileWriter.writerow(["fileName","totalCount","surpriseCount","index_pairs","surpriseScore","threshold"])
for j, file in enumerate(data):
    bar.update(j)
    openfile = codecs.open(file,encoding='utf-8')
    outputFile = open("../output2/"+str(file).replace('.txt','.csv'),'w')
    csvWriter = csv.writer(outputFile)
    #print outputFile
    csvWriter.writerow(["sentence1","sentence2","score"])
    fileData = openfile.read()
    sentences = tokenizer.tokenize(fileData)
    for i in range(1,len(sentences)):
        if ( i < (len(sentences)-2)):
            #sentence1 = re.sub(r"/\[MAD\\s*([^\n\r].*)END MAD\]/",'',sentences[i].strip('\n'))
            words1 = nltk.word_tokenize(sentences[i].strip('\n'))
            #sentence2 = re.sub(r"\[MAD\s*([^\n\r].*) END MAD]",'',sentences[i+1].strip('\n'))
            filtered_sentence1 = ' '.join([w for w in words1 if not w in stopWords])
            words2 = nltk.word_tokenize(sentences[i+1].strip('\n'))
            filtered_sentence2 = ' '.join([w for w in words2 if not w in stopWords])
            #print sentences[i]
            #print sentences[i+1]
            phraseVector1 = PhraseVector(filtered_sentence1)
            phraseVector2 = PhraseVector(filtered_sentence2)
            similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
            #print similarityScore
            if(similarityScore != 0):
                csvWriter.writerow([sentences[i].encode("utf-8"),sentences[i+1].encode("utf-8"),similarityScore])
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
                        index_pairs = index_pairs + str(i+2) + ',' + str(i + 3) + '|'
        if(surpriseCount != 0 and threshold < 0.4):
            surpriseScore = surpriseCount/totalCount
            outputFileWriter.writerow([str(file),totalCount,surpriseCount,index_pairs,surpriseScore,threshold])
            os.remove(file)
    except KeyError:
        continue;
FinalFile.close()
