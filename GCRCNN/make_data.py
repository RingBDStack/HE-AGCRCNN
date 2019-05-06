
# coding: utf-8

# In[5]:

import os
import string
import re
import sys
import collections
import numpy as np
import gensim
import codecs
import h5py
import json
import xml.etree.ElementTree as ET
from multiprocessing import Pool

# In[6]:

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


# In[8]:

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '”', '“', '’', "‘",
                        "'", '"']
wordEngStop = nltk.corpus.stopwords.words('english')
st = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

count = 1;

w_idnex,wdata = None,None
classes = None

data_source_path = "../xml2"


# In[97]:

def count_words(s):
    global english_punctuations, wordEngStop, st
    tokenstr = []
    result = {}
                
    mtext = s
    mtext = mtext.lower().strip()
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
    word2pos = {}
    pos2word = {}
    pos = 0
    for word in words:
        if not word in english_punctuations and not word in wordEngStop and word != "" and word.isalpha():
            orig_stem = lemmatizer.lemmatize(word)
            tokenstr.append(orig_stem)
            result[orig_stem] = result.get(orig_stem, 0) + 1
            pos2word[pos] = orig_stem
            try:
                word2pos[orig_stem].append(pos)
            except:
                word2pos[orig_stem] = [pos]
            pos += 1

    # sort
    result = collections.OrderedDict(sorted(result.items(), key=lambda x: (x[1], x[0]), reverse=True))
    wordslist = result.keys()
    assert len(set(tokenstr)) == len(wordslist)
    return (wordslist, tokenstr, word2pos, pos2word)


# In[98]:

def fill_table(TD_list, related_tables,target_width, qqueue):
    TD_list[0] = qqueue[0]
    count = 1

    while qqueue != [] and count < target_width:
        use_index = qqueue[0]
        del qqueue[0]
        use_list = related_tables[use_index]
        len1 = len(use_list)
        len2 = target_width - count
        if len1 >= len2:
            TD_list[count:] = use_list[:len2]
            assert len(TD_list) == target_width
            count = target_width
            break
        else:
            TD_list[count:count + len1] = use_list
            assert len(TD_list) == target_width
            count += len1
            for next_id in use_list:
                qqueue.append(next_id)
    for i in range(count, target_width):
        TD_list[i] = -1


# In[153]:

def test_text2matrix(_str, sliding_win=3, target_width=5):
    (wordslist, tokenwords, word2pos, pos2word) = count_words(_str)
    wlist = list(wordslist)
    word2id = {}
    for i in range(len(wlist)):
        word2id[wlist[i]] = i
    wordslist_length = len(wlist)
    if target_width > wordslist_length:
        raise ValueError("error")
    AM_table = [[0 for i in range(wordslist_length)] for j in range(wordslist_length)]
    for num in range(0, len(tokenwords) - sliding_win + 1):
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 1])] += 1
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num + 1])] += 1
    related_tables = {}
    for i in range(wordslist_length):
        related_tables[i] = [[index, num] for index, num in enumerate(AM_table[i]) if num > 0 and index != i]
        related_tables[i].sort(key=lambda x: x[1], reverse=True)
        related_tables[i] = [element[0] for element in related_tables[i]]
    TD_table = [[0 for i in range(target_width)] for j in range(wordslist_length)]
    for i in range(wordslist_length):
        fill_table(TD_table[i], related_tables,target_width, [i])
    
    TD_table = reorder(TD_table, word2pos, pos2word, wlist, word2id)
    return wordslist, TD_table


# In[100]:

def matrix_vector(wordslist, TD_table, target_width, word_vector_size):
    global wdata,w_idnex
    wlist = list(wordslist)
    TTD_table = np.zeros((word_vector_size, len(wlist), target_width), dtype=np.float32)
    
    for num_i in range(len(wlist)):
        for num_j in range(target_width):
            if TD_table[num_i][num_j] > -1:
                try:
                    aword = wlist[TD_table[num_i][num_j]]
                    wind = w_idnex[aword]
                    c_wordvector = wdata[wind]

                except:                                                                    
                    
                    aword = wlist[TD_table[num_i][num_j]]
                    c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
            else:
                c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
            TTD_table[:, num_i, num_j] = c_wordvector
    return (TTD_table)


# In[148]:

def reorder(table, word2pos, pos2word, wlist, word2id):
    sort_table = []
    topn, neighbor = np.array(table).shape
    for i in range(topn):
        tmp = []
        tmp += word2pos[wlist[table[i][0]]]
        length = len(tmp)
        t = []
        for j in range(1, neighbor):
            t += word2pos[wlist[table[i][j]]]
        index = np.random.randint(len(t), size = 20-length)
        t = np.array(t)
        t = list(t[index])
        tmp = tmp + t
        tmp.sort()
        for j in range(len(tmp)):
            tmp[j] = word2id[pos2word[tmp[j]]]
        sort_table.append(tmp)
    
    return np.array(sort_table)


# In[155]:

def process(path,start,end,slise_window, target_width, word_vector_size, words_limit,class_nums):
    flag = 0
    tfpath = path
    prob = []
    for i in range(start,end):
#        print(i)
        one_hot_codes = np.zeros(class_nums)
        p = "{0}newsML.xml".format(i)
        fff = os.path.join(tfpath,p)
        if not os.path.exists(fff):
            continue
        xmlcont = ET.parse(fff)
        root = xmlcont.getroot()
        content = ''
        for neighbor in root.iter('title'):
            content += (neighbor.text+' ')
        for neighbor in root.iter('headline'):
            content += (neighbor.text+' ')
        for neighbor in root.iter('p'):
            content += (neighbor.text+' ')

        topics = []
        for neighbor in root.iter('codes'):
            tclass = list(neighbor.attrib.values())
            for lst in tclass:
                if 'topics' in lst:
                    for nn in neighbor.iter('code'):
                        topics.append(nn.attrib['code'])
        try:
            (wordslist, TD_table) = test_text2matrix(content, slise_window, target_width)
        except:
            prob.append(i)
            continue
        
        TTD_table = matrix_vector(wordslist, TD_table, target_width, word_vector_size)
        shape0, shape1, shape2 = TTD_table.shape
        final_one_TTD = None
        if shape1 < words_limit:
            final_one_TTD = np.zeros((shape0, words_limit, shape2), dtype=np.float32)
            final_one_TTD[:, :shape1, :shape2] = TTD_table
        else:
            final_one_TTD = TTD_table[:, :words_limit, :shape2]
        final_one_TTD = final_one_TTD.reshape((1, word_vector_size, words_limit, target_width))
            
        if flag == 0:
            _X = final_one_TTD
            flag = 1
        else:
            _X = np.concatenate((_X, final_one_TTD), axis=0)
    
    # save ignored sample's id (process with problems)
    fpath = '../problem{0}_{1}.npy'.format(start,end)
    np.save(fpath, np.array(prob))
    fpath = '../range{0}_{1}.npy'.format(start,end)
    print(_X.shape)
    np.save(fpath, _X.transpose(0, 2, 3, 1))
    


if __name__ == '__main__':
    slise_window = 3

    target_width = 20

    word_vector_size = 50
    words_limit = 100
    class_nums = 103
                
    with open('./classes.json', 'r') as f3:
        classes = json.load(f3)
        
    with open('./words.json', 'r') as f3:
        w_idnex = json.load(f3)

    h5 = h5py.File('./matrix_rcv1.h5', 'r')
    wdata = h5['data'].value
              
    raw_path = data_source_path
    

    
    
    #test ID 25993-810597
    lnums = [(30000+i*2000,30000+(i+1)*2000) for i in range(0,25)]
#    lnums = [(0, 25994)]
    p = Pool(30)
    results = []
    for i in range(len(lnums)):
        start,end = lnums[i]
        print("process{0} start. Range({1},{2})".format(i,start,end))
        results.append(p.apply_async(process,args=(raw_path,start,end,slise_window,target_width,word_vector_size,words_limit,class_nums)))
        print("process{0} end".format(i))
    p.close()
    p.join()
    for r in results:
        print(r.get())
    
    
    
    
    print('Done!!!')


# In[ ]:



