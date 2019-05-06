# -*- coding: utf-8 -*-
import os
import nltk
import string
import re
import os
from nltk.corpus import wordnet as wn
import sys
import collections
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import gensim
import codecs
import h5py
import json
from multiprocessing import Pool
import xml.etree.ElementTree as ET

reload(sys)
sys.setdefaultencoding('utf-8')

PATH = os.path.dirname(os.path.realpath(__file__))
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '”', '“', '’', "‘",
                        "'", '"']
wordEngStop = nltk.corpus.stopwords.words('english')
st = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

count = 1;

w_idnex,wdata = None,None
classes = None

data_source_path = "./xml"
graph_output_path = "./test_h5"

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


# dfs填充
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


def test_text2matrix(_str, sliding_win=3, target_width=5):
    (wordslist, tokenwords) = count_words(_str)
    wlist = list(wordslist)
    wordslist_length = len(wlist)
    if target_width > wordslist_length:
        raise ValueError("图矩阵宽度大于词种类数量")
    # 统计词频
    AM_table = [[0 for i in range(wordslist_length)] for j in range(wordslist_length)]
    for num in range(0, len(tokenwords) - sliding_win + 1):
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 1])] += 1
        AM_table[wlist.index(tokenwords[num])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num + 2])] += 1
        AM_table[wlist.index(tokenwords[num + 1])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num])] += 1
        AM_table[wlist.index(tokenwords[num + 2])][wlist.index(tokenwords[num + 1])] += 1
    # 关联矩阵：每个单词关联的单词降序排列
    related_tables = {}
    for i in range(wordslist_length):
        related_tables[i] = [[index, num] for index, num in enumerate(AM_table[i]) if num > 0 and index != i]
        related_tables[i].sort(key=lambda x: x[1], reverse=True)
        related_tables[i] = [element[0] for element in related_tables[i]]
    TD_table = [[0 for i in range(target_width)] for j in range(wordslist_length)]
    # 第一个单词是它本身
    for i in range(wordslist_length):
        fill_table(TD_table[i], related_tables,target_width, [i])

    return wordslist, TD_table


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
                    print aword
                    c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
            else:
                c_wordvector = np.zeros((word_vector_size), dtype=np.float32)
            TTD_table[:, num_i, num_j] = c_wordvector
    return (TTD_table)


def process(path,start,end,slise_window, target_width, word_vector_size, words_limit,class_nums):
    _X = None
    _y = None
    flag = 0

    tfpath = path
    for i in range(start,end):
        one_hot_codes = np.zeros(class_nums)
        p = "{0}newsML.xml".format(i)
        fff = os.path.join(tfpath,p)
        if not os.path.exists(fff):
            continue
        xmlcont = ET.parse(fff)
        root = xmlcont.getroot()
        content = []
        for neighbor in root.iter('title'):
            content.append(neighbor.text)
        for neighbor in root.iter('headline'):
            content.append(neighbor.text)
        for neighbor in root.iter('p'):
            content.append(neighbor.text)

        topics = []
        for neighbor in root.iter('codes'):
            tclass = list(neighbor.attrib.values())
            # print(tclass)
            for lst in tclass:
                if 'topics' in lst:
                    for nn in neighbor.iter('code'):
                        topics.append(nn.attrib['code'])

        while None in content:
            content.remove(None)
        a =content
        try:
            (wordslist, TD_table) = test_text2matrix(a, slise_window, target_width)
        except:
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
            
            
        for label in topics:
            one_hot_codes[classes[label]] = 1.0
        _yxx = one_hot_codes
        _yxx = _yxx.reshape(1,-1)
            
        if flag == 0:
            _X = final_one_TTD
            _y = _yxx
            flag = 1
        else:
            _X = np.concatenate((_X, final_one_TTD), axis=0)
            _y = np.concatenate((_y, _yxx), axis=0)

    fpath = os.path.join(data_source_path,"range{0}_{1}.h5".format(start,end))
    print fpath
    f = h5py.File(fpath, "w")
    f.create_dataset("datax", data=_X)
    f.create_dataset("datay", data=_y)
    f.close()



    
if __name__ == '__main__':
    slise_window = 3
    # 目标宽度
    target_width = 10
    # 词向量长度
    word_vector_size = 50
    words_limit =96 
    class_nums = 103
                
    with open(r'./classes.json', "r") as f3:
        classes = json.load(f3)
        
    with open(r"./words.json", "r") as f3:
        w_idnex = json.load(f3)

    h5 = h5py.File(r"./matrix_rcv1.h5", 'r')
    wdata = h5['data'].value
              
    raw_path = data_source_path
    target_path = graph_output_path
    isExists = os.path.exists(target_path)
    if not isExists:
        os.makedirs(target_path)
    

    
    
    #test ID 25993-810597
    lnums = [(30000+i*31200,30000+(i+1)*31200) for i in range(0,26)]+[(25993,30000)]+[(810000,810597)]
    print(lnums)
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
    
    
    
