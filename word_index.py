import re
import numpy as np

'''词粒度信息整合'''
class WordIndex():
    def __init__(self, opt):
        self.train_keyword_sorted=[]
        self.valid_keyword_sorted = []
        self.test_keyword_sorted = []
        self.opt = opt

        '''建立三个文档出现的所有word的词典，并用int为其编码'''
        self.pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        self.pattern_test='.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        source_files=['./data/train_b.txt','./data/valid_b.txt','./data/test_b.txt']
        words = []
        for source_file in source_files:
            if source_file=='./data/test_b.txt':
                pattern=self.pattern_test
            else:
                pattern=self.pattern
            with open(source_file) as f:
                data = f.read()
                data = re.findall(pattern, data)[1:]
                title_words = [data_item[1] for data_item in data]
                describe_words = [data_item[3] for data_item in data]
            for i in range(len(data)):
                words += title_words[i].split(',')+describe_words[i].split(',')
        self.words = set(words)
        words = list(set(words))
        words.sort()                                                            # 这一步非常关键，这决定了test.py文件
        self.word_to_idx_dict = {word: i + 1 for i, word in enumerate(words)}   # 在运行时能否获得和训练集验证集一样的
                                                                                # 编码方式

        '''之前生成的保存有关键词信息的文档中载入关键词信息（已根据tf-idf值进行了排序）'''
        pattern_keyword = '(.*?)\n'
        with open('./keys_extract/train_keywords.txt') as f:
            data = f.read()
            data = re.findall(pattern_keyword, data)
            self.train_keyword_sorted = [data_item.split(',') for data_item in data]
        with open('./keys_extract/valid_keywords.txt') as f:
            data = f.read()
            data = re.findall(pattern_keyword, data)
            self.valid_keyword_sorted = [data_item.split(',') for data_item in data]
        with open('./keys_extract/test_keywords.txt') as f:
            data = f.read()
            data = re.findall(pattern_keyword, data)
            self.test_keyword_sorted = [data_item.split(',') for data_item in data]

    '''根据生成的word词典为数据进行编码'''
    def word_to_idx(self, inputs):
        outputs = []
        for item in inputs:
            outputs.append(self.word_to_idx_dict[item])
        if len(outputs) < self.opt.KEY_LEN:
            outputs += [0]*(self.opt.KEY_LEN-len(outputs))
        else:
            outputs = outputs[:self.opt.KEY_LEN]
        outputs = np.array(outputs, dtype=int)
        return outputs

