# -*- coding: utf-8 -*-
import os
import zipfile
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import re
import json
import numpy as np
import gensim
import h5py
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk

data_source_path = r"./xml"
temp_words_path = r"./temp_data"
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*','”','“','’',"‘","'",'"']
wordEngStop = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()



def readfile(path):
    f = open(path,'r')
    s = f.read()
    finalwords = []
    
    mtext = s
    mtext = mtext.lower().strip().decode(errors="ignore")
    mtext = re.sub(r'-', r' ', mtext)
    mtext = re.sub(r'([0-9]+),([0-9]+)', r'\1\2', mtext)
    mtext = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", mtext)
    mtext = re.sub(r"\'s", " \'s", mtext)
    mtext = re.sub(r"\'ve", " \'ve", mtext)
    mtext = re.sub(r"n\'t", " n\'t", mtext)
    mtext = re.sub(r"\'re", " \'re", mtext)
    mtext = re.sub(r"\'d", " \'d", mtext)
    mtext = re.sub(r"\'ll", " \'ll", mtext)
    mtext = re.sub(r",", " , ", mtext)
    mtext = re.sub(r"!", " ! ", mtext)
    mtext = re.sub(r"\(", " \( ", mtext)
    mtext = re.sub(r"\)", " \) ", mtext)
    mtext = re.sub(r"\?", " \? ", mtext)
    mtext = re.sub(r"\s{2,}", " ", mtext)
    
    words = WordPunctTokenizer().tokenize(mtext)
    for word in words:
        if not word in english_punctuations and not word in wordEngStop and word != "" and word.isalpha():
            orig_stem = lemmatizer.lemmatize(word)
            finalwords.append(orig_stem)

    return finalwords

def find_words(start,end,target_path):
    all_words = {}
    for i in range(start,end):
        p = "{0}newsML.xml".format(i)
        fff = os.path.join(data_source_path,p)
        if not os.path.exists(fff):
            continue
        content = readfile(fff)
        
        for word in content:
            if word not in all_words.keys():
                all_words[word] = True
    pp = os.path.join(target_path,"words_from{0}_to_{1}.json".format(start,end))
    print(pp)
    with open(pp,"w") as fp:
        json.dump(all_words, fp)


def find_all_words():
    target_path = temp_words_path
    isExists = os.path.exists(target_path)
    if not isExists:
        os.makedirs(target_path)

    #for sample ID from 2286 to 810597
    lnums = [(30000+i*31200,30000+(i+1)*31200) for i in range(0,26)]+[(2286,30000)]+[(810000,810597)]
    print(lnums)
    p = Pool(30)
    results = []
    for i in range(len(lnums)):
        start,end = lnums[i]
        print("process{0} start. Range({1},{2})".format(i,start,end))
        results.append(p.apply_async(find_words,args=(start,end,target_path)))
        print("process{0} end".format(i))
    p.close()
    p.join()
    for r in results:
        print(r.get())



def allwords():
    tpath = temp_words_path
    words = {}
    ind = 0
    flist = os.listdir(tpath)
    for f in flist:
        ppath = os.path.join(tpath,f)
        with open(ppath, "r") as f1:
            simjson = json.load(f1)
            for i in simjson.keys():
                if i not in words.keys():
                    words[i] = ind
                    ind += 1
    lens = len(list(words.keys()))
    wembeddingwords = np.random.uniform(-1.0, 1.0, (lens, 50))
    word2vec_model = gensim.models.Word2Vec.load(r'./wiki.en.text.model')
    for key in words.keys():
        if key in word2vec_model:
            index = words[key]
            wembeddingwords[index, :] = word2vec_model[key]
    with open(r"./words.json", "w") as f:
        json.dump(words, f)
    f = h5py.File("./matrix_rcv1.h5", "w")
    f.create_dataset("data", data=wembeddingwords)
    f.close()

def classpro():
    tpath = r'./ReutersCorpusVolume1/Data/ReutersCorpusVolume1_Original/CD1/topic_codes.txt'
    labels = {}
    with open(tpath,"r") as f:
        lines = f.readlines()
        print(len(lines))
        for index,line in enumerate(lines[2:]):
            if line != '\n' and '\t' in line:
                labels[line.strip().split('\t')[0]] = index
        for k,v in labels.items():
            print(k,v)
    print(len(list(labels.keys())))
    with open(r'./classes.json','w') as f:
        json.dump(labels,f)


if __name__ == "__main__":
    find_all_words()
    allwords()
    classpro()
