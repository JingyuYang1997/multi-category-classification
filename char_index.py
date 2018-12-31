import re
import numpy as np

'''字粒度信息整合'''
class CharIndex():
    def __init__(self, opt):
        self.train_keychar_sorted = []
        self.valid_keychar_sorted = []
        self.test_keychar_sorted = []
        self.opt = opt

        '''建立三个文档出现的所有char的字典，并用int为其编码'''
        self.pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        self.pattern_test='.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        source_files=['./data/train_b.txt','./data/valid_b.txt','./data/test_b.txt']
        chars = []
        for source_file in source_files:
            if source_file=='./data/test_b.txt':
                pattern=self.pattern_test
            else:
                pattern=self.pattern
            with open(source_file) as f:
                data = f.read()
                data = re.findall(pattern, data)[1:]
                title_chars = [data_item[0] for data_item in data]
                describe_chars = [data_item[2] for data_item in data]
            for i in range(len(data)):
                chars += title_chars[i].split(',')+describe_chars[i].split(',')
        self.chars = set(chars)
        chars = list(set(chars))
        chars.sort()
        self.char_to_idx_dict = {char: i+1 for i, char in enumerate(chars)}

        '''之前生成的保存有关键字信息的文档中载入关键词信息（已根据tf-idf值进行了排序）'''
        pattern_keychar = '(.*?)\n'
        with open('./keys_extract/train_keychars.txt') as f:
            data = f.read()
            data = re.findall(pattern_keychar, data)
            self.train_keychar_sorted = [data_item.split(',') for data_item in data]
        with open('./keys_extract/valid_keychars.txt') as f:
            data = f.read()
            data = re.findall(pattern_keychar, data)
            self.valid_keychar_sorted = [data_item.split(',') for data_item in data]
        with open('./keys_extract/test_keychars.txt') as f:
            data = f.read()
            data = re.findall(pattern_keychar, data)
            self.test_keychar_sorted = [data_item.split(',') for data_item in data]

    '''根据生成的char字典为数据进行编码'''
    def char_to_idx(self, inputs):
        outputs = []
        for item in inputs:
            outputs.append(self.char_to_idx_dict[item])
        if len(outputs) < self.opt.KEY_LEN:
            outputs += [0]*(self.opt.KEY_LEN-len(outputs))
        else:
            outputs = outputs[:self.opt.KEY_LEN]
        outputs = np.array(outputs, dtype=int)
        return outputs
