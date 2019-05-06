# -*- coding: utf-8 -*-
import os
import string
import re
import os
import sys
import collections
import numpy as np
import codecs
import h5py
import json
import xml.etree.ElementTree as ET
from multiprocessing import Pool

PATH = os.path.dirname(os.path.realpath(__file__))
count = 1;
P = 1.581978

classes = None
class_nums = 103


with open(r'./classes.json', "r") as f3:
    classes = json.load(f3)
with open(r'./sim.json', "r") as f3:
    sim = json.load(f3)
        
data_source_path = "../xml2"

def count_words(s):
    global english_punctuations, wordEngStop, st
    tokenstr = []
    result = {}
                
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
    
    finalwords = []
    words = WordPunctTokenizer().tokenize(mtext)
    for word in words:
        if not word in english_punctuations and not word in wordEngStop and word != "" and word.isalpha():
            orig_stem = lemmatizer.lemmatize(word)
            tokenstr.append(orig_stem)
            result[orig_stem] = result.get(orig_stem, 0) + 1

    # sort
    result = collections.OrderedDict(sorted(result.items(), key=lambda x: (x[1], x[0]), reverse=True))
    wordslist = result.keys()
    assert len(set(tokenstr)) == len(wordslist)
    # 不重复的单词按照出现次数降序排列的list，第二个是按照出现顺序排列的单词词组
    return (wordslist, tokenstr)





def process(path,start,end,class_nums):
    _y = None
    weight = None
    flag = 0
    try:
        problem = np.load('../problem{0}_{1}.npy'.format(start,end))
    except:
        return
    problem = list(problem)
    tfpath = path
    for i in range(start,end):
#        print(i)
        if i in problem:
           continue
        one_hot_codes = np.zeros(class_nums)
        p = "{0}newsML.xml".format(i)
        fff = os.path.join(tfpath,p)
        if not os.path.exists(fff):
            continue
        xmlcont = ET.parse(fff)
        root = xmlcont.getroot()
        haha = []
        for neighbor in root.iter('title'):
            haha.append(neighbor.text)
        for neighbor in root.iter('headline'):
            haha.append(neighbor.text)
        for neighbor in root.iter('p'):
            haha.append(neighbor.text)

        topics = []
        for neighbor in root.iter('codes'):
            tclass = list(neighbor.attrib.values())
            # print(tclass)
            for lst in tclass:
                if 'topics' in lst:
                    for nn in neighbor.iter('code'):
                        topics.append(nn.attrib['code'])


          
        target = []
        for label in topics:
            one_hot_codes[classes[label]] = 1.0
            target.append(classes[label])
            
        _yxx = one_hot_codes
        _yxx = _yxx.reshape(1,-1)
        
        matrix = []    
        temp = []
        

        for t in target:        
            matrix.append(sim[t])       #把所有LABEL与LABEL"J"的相似度加入matrix
        if len(target)>0:                   #如果命中的LABEL数量大于0
            matrix = np.array(matrix).transpose(1,0)        #并行操作
            temp = matrix.max(1)                            
            temp = np.exp(-1*temp)*P
            for j in target:
                temp[j] = 1.
        else:
            temp = np.array([1.]*class_nums)
            
        
        temp =temp.reshape([1,-1])
        
        if flag == 0:
            _y = _yxx
            weight = temp
            flag = 1
        else:
            _y = np.concatenate((_y, _yxx), axis=0)
            weight = np.concatenate((weight, temp), axis=0)
    
    fpath = '../label_range{0}_{1}.npy'.format(start,end) 
    np.save(fpath, _y)
    #fpath = 'data/test/weight/weight_range{0}_{1}.npy'.format(start,end)
    #np.save(fpath, weight)

    
if __name__ == '__main__':
    
    #test ID 25993-810597
    lnums = [(30000+i*2000,30000+(i+1)*2000) for i in range(0,25)]#+[(24000,25993)]
    #lnums = [(25993, 30000)]
    p = Pool(30)
    results = []
    for i in range(len(lnums)):
        start,end = lnums[i]
        print("process{0} start. Range({1},{2})".format(i,start,end))
        results.append(p.apply_async(process,args=(data_source_path,start,end,class_nums)))
        print("process{0} end".format(i))
    p.close()
    p.join()
        
    
    print('Done!!!')
    
    
    
