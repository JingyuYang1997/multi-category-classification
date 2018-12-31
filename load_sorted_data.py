import numpy as np
import re
from class_index import ClassIndex
from word_index import WordIndex
from char_index import CharIndex
from config import Config

config_obj = Config()
classindex = ClassIndex(opt=config_obj)
wordindex = WordIndex(opt=config_obj)
charindex = CharIndex(opt=config_obj)

'''载入word_data'''
def get_word_data(datasource):
    if datasource=='test':
        pattern='.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
    else:
        pattern='.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
    with open('./data/'+datasource+'_b.txt') as f:
        data = f.read()
        data = re.findall(pattern, data)[1:]
        inputs_pre = [(data_item[1] + ',' + data_item[3]).split(',') for data_item in data]
    if datasource=='train':
        keyword_sorted = wordindex.train_keyword_sorted
    if datasource=='valid':
        keyword_sorted = wordindex.valid_keyword_sorted
    if datasource=='test':
        keyword_sorted = wordindex.test_keyword_sorted

    '''根据tf-idf值为所有的word_data进行排序，tf-idf值高的word置前，'''
    '''除了顺序改变，完全保留原有文本的所有word信息（即允许词汇重复）'''
    inputs = []
    for input_index in range(len(inputs_pre)):
        input=[]
        for keyword in keyword_sorted[input_index]:
            keyword_count = inputs_pre[input_index].count(keyword)
            input += [keyword]*keyword_count
        inputs.append(input)

    '''将word_data由string转化为int编码形式'''
    data_words=[]
    for item in inputs:
        data_index = wordindex.word_to_idx(item)
        data_words.append(data_index)
    if datasource == 'test':
        return np.array(data_words)

    targets = [[data_item[4], data_item[5], data_item[6]] for data_item in data]
    label1=[]
    label2=[]
    label3=[]
    for item in targets:
        label_index = classindex.class_to_idx(item)
        label1.append(label_index[0])
        label2.append(label_index[1])
        label3.append(label_index[2])
    labels=[np.array(label1),np.array(label2),np.array(label3)]
    return np.array(data_words), labels


'''载入char_data'''
def get_char_data(datasource):
    if datasource == 'test':
        pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
    else:
        pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
    with open('./data/'+datasource+'_b.txt') as f:
        data = f.read()
        data = re.findall(pattern, data)[1:]
        inputs_pre = [(data_item[0] + ',' + data_item[2]).split(',') for data_item in data]
    if datasource == 'train':
        keychar_sorted = charindex.train_keychar_sorted
    if datasource == 'valid':
        keychar_sorted = charindex.valid_keychar_sorted
    if datasource == 'test':
        keychar_sorted = charindex.test_keychar_sorted

    '''根据tf-idf值为所有的char_data进行排序，tf-idf值高的cahr置前，'''
    '''除了顺序改变，完全保留原有文本的所有char信息（即允许词汇重复）'''
    inputs = []
    for input_index in range(len(inputs_pre)):
        input = []
        for keychar in keychar_sorted[input_index]:
            keychar_count = inputs_pre[input_index].count(keychar)
            input += [keychar] * keychar_count
        inputs.append(input)

    data_chars=[]
    for item in inputs:
        data_index = charindex.char_to_idx(item)
        data_chars.append(data_index)
    return np.array(data_chars)



